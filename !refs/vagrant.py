# src/vagrant.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import gc
import psutil

logger = logging.getLogger(__name__)

class Vagrant(nn.Module):
    def __init__(self, dispatcher, max_layers=5):
        super().__init__()
        self.dispatcher = dispatcher
        self.device = dispatcher.device
        self.embedding_layer = nn.Embedding(dispatcher.vocab_size, dispatcher.hidden_size, dtype=torch.float16)
        self.shortcut_embed = nn.Embedding(1000, dispatcher.hidden_size, dtype=torch.float16)
        self.block_size = 512
        self.hidden_size = dispatcher.hidden_size
        self.num_layers = min(dispatcher.num_layers, max_layers)
        self.num_attention_heads = dispatcher.num_attention_heads
        self.intermediate_size = dispatcher.intermediate_size
        self.key_dim = dispatcher.key_dim
        self.num_key_value_heads = dispatcher.num_key_value_heads
        self.vocab_size = dispatcher.vocab_size
        
        self.query_scorer = nn.Linear(self.hidden_size, 2, dtype=torch.float16)
        self.context_scorer = nn.Linear(self.hidden_size * 2, 1, dtype=torch.float16)
        self.block_scorer = nn.Linear(self.hidden_size, 1, dtype=torch.float16)
        self.layer_scorer = nn.Linear(self.hidden_size, self.num_layers, dtype=torch.float16)
        self.intermediate_proj = nn.Linear(self.intermediate_size, self.hidden_size, dtype=torch.float16)
        
        self.prev_context = None

    def classify_query(self, hidden_states):
        scores = self.query_scorer(hidden_states.mean(dim=1))  # [batch_size, 2]
        return "light" if scores[0, 0] > scores[0, 1] else "deep"

    def is_related(self, input_ids, prev_context):
        if prev_context is None:
            return False
        new_embed = self.embedding_layer(input_ids).mean(dim=1)
        combined = torch.cat([new_embed, prev_context.mean(dim=1)], dim=-1)
        score = self.context_scorer(combined).sigmoid()
        return score > 0.5

    def is_common(self, input_ids):
        return torch.all(input_ids < 1000)

    def forward(self, input_ids, max_steps=10):
        logger.info(f"Vagrant получил входные данные с формой: {input_ids.shape}")
        
        batch_size, seq_len = input_ids.shape
        if self.is_common(input_ids) and self.prev_context is None:
            hidden_states = self.shortcut_embed(input_ids)
        else:
            hidden_states = self.embedding_layer(input_ids).to(self.device)
        logger.info(f"Embeddings shape: {hidden_states.shape}")

        focus_level = self.classify_query(hidden_states)
        active_layers = self.select_layers(hidden_states, focus_level)

        for layer_idx in active_layers:
            process = psutil.Process()
            logger.info(f"RAM before layer {layer_idx}: {process.memory_info().rss / 1024**2:.2f} MB")
            logger.info(f"Vagrant processing layer {layer_idx}")
            
            if self.scan_layer(hidden_states, layer_idx):
                hidden_states = self.process_attention(hidden_states, layer_idx, focus_level)
                self.clear_memory()
                hidden_states = self.process_mlp(hidden_states, layer_idx)
                self.clear_memory()

        output = self.process_output(hidden_states)
        self.prev_context = hidden_states.detach()
        self.clear_memory()
        return output

    def select_layers(self, hidden_states, focus_level):
        layer_scores = self.layer_scorer(hidden_states.mean(dim=1))  # [batch_size, num_layers]
        layer_scores = layer_scores[0]  # Берем первый батч: [num_layers]
        if focus_level == "light":
            indices = layer_scores.argsort(descending=True)[:1]  # [1]
        else:
            indices = layer_scores.argsort(descending=True)[:3]  # [3]
        return indices.tolist()  # Плоский список, например, [4] или [4, 3, 0]

    def scan_layer(self, hidden_states, layer_idx):
        quick_q = self.apply_blocks(hidden_states, [f"{self.dispatcher.model_name}_layer{layer_idx}_self_attn_q_proj_weight_block0"], self.hidden_size)
        score = self.block_scorer(quick_q.mean(dim=1)).sigmoid()
        return score > 0.5

    def process_attention(self, hidden_states, layer_idx, focus_level):
        batch_size, seq_len, _ = hidden_states.shape
        
        q_blocks = self.select_blocks(hidden_states, f"{self.dispatcher.model_name}_layer{layer_idx}_self_attn_q_proj_weight", self.hidden_size)
        k_blocks = self.select_blocks(hidden_states, f"{self.dispatcher.model_name}_layer{layer_idx}_self_attn_k_proj_weight", self.key_dim)
        v_blocks = self.select_blocks(hidden_states, f"{self.dispatcher.model_name}_layer{layer_idx}_self_attn_v_proj_weight", self.key_dim)

        q = self.apply_blocks(hidden_states, q_blocks, self.hidden_size)
        k = self.apply_blocks(hidden_states, k_blocks, self.key_dim)
        v = self.apply_blocks(hidden_states, v_blocks, self.key_dim)

        head_dim = self.hidden_size // self.num_attention_heads
        key_head_dim = self.key_dim // self.num_key_value_heads
        heads_per_group = self.num_attention_heads // self.num_key_value_heads

        if focus_level == "light":
            seq_len = min(seq_len, 4)
            q, k, v = q[:, :seq_len], k[:, :seq_len], v[:, :seq_len]

        q = q.view(batch_size, seq_len, self.num_attention_heads, head_dim)
        q = q.view(batch_size, seq_len, self.num_key_value_heads, heads_per_group, head_dim)
        q = q.permute(0, 2, 3, 1, 4)

        k = k.view(batch_size, seq_len, self.num_key_value_heads, key_head_dim)
        k = k.permute(0, 2, 1, 3)

        v = v.view(batch_size, seq_len, self.num_key_value_heads, key_head_dim)
        v = v.permute(0, 2, 1, 3)

        scores = torch.einsum('bhgsd,bhqd->bhgsq', q, k) / (head_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        mask = attn_weights > 0.01
        attn_weights = attn_weights * mask

        v_expanded = v.unsqueeze(2).expand(-1, -1, heads_per_group, -1, -1)
        attn_output = torch.einsum('bhgsq,bhgsd->bhgsd', attn_weights, v_expanded)

        attn_output = attn_output.permute(0, 3, 1, 2, 4).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)

        o_blocks = self.select_blocks(attn_output, f"{self.dispatcher.model_name}_layer{layer_idx}_self_attn_o_proj_weight", self.hidden_size)
        hidden_states = self.apply_blocks(attn_output, o_blocks, self.hidden_size)

        del q, k, v, scores, attn_weights, v_expanded, attn_output
        return hidden_states

    def process_mlp(self, hidden_states, layer_idx):
        gate_blocks = self.select_blocks(hidden_states, f"{self.dispatcher.model_name}_layer{layer_idx}_mlp_gate_proj_weight", self.intermediate_size)
        up_blocks = self.select_blocks(hidden_states, f"{self.dispatcher.model_name}_layer{layer_idx}_mlp_up_proj_weight", self.intermediate_size)
        
        gate = self.apply_blocks(hidden_states, gate_blocks, self.intermediate_size)
        up = self.apply_blocks(hidden_states, up_blocks, self.intermediate_size)
        mlp_output = gate * up
        
        down_blocks = self.select_blocks(mlp_output, f"{self.dispatcher.model_name}_layer{layer_idx}_mlp_down_proj_weight", self.hidden_size)
        hidden_states = self.apply_blocks(mlp_output, down_blocks, self.hidden_size)

        del gate, up, mlp_output
        return hidden_states

    def process_output(self, hidden_states):
        output_blocks = self.select_blocks(hidden_states, f"{self.dispatcher.model_name}_output", self.vocab_size)
        subblocks = []
        for block in output_blocks:
            full_block = self.dispatcher.load_block(self.dispatcher.metadata[block])
            subblocks.extend(torch.split(full_block, self.block_size, dim=0)[:5])
        
        if not subblocks:
            logger.warning("Нет доступных субблоков для output, возвращаем нули")
            return torch.zeros(hidden_states.shape[0], hidden_states.shape[1], self.block_size * 5, dtype=torch.float16, device=self.device)
        
        combined_weights = torch.cat(subblocks, dim=0)
        logger.info(f"Объединенные веса выходных субблоков имеют форму: {combined_weights.shape}")
        output = torch.matmul(hidden_states, combined_weights.t())
        return output

    def select_blocks(self, input_data, prefix, output_size, num_blocks=1):
        if input_data.shape[-1] == self.intermediate_size:
            activations = self.intermediate_proj(input_data.mean(dim=1))
        else:
            activations = input_data[:, 0] if input_data.shape[1] > 0 else input_data.mean(dim=1)
        
        base_score = self.block_scorer(activations).squeeze(-1)
        
        available_blocks = [key for key in self.dispatcher.metadata.keys() if prefix in key]
        if not available_blocks:
            logger.warning(f"Нет блоков для префикса {prefix}")
            return []
        
        scores = torch.zeros(len(available_blocks), device=self.device, dtype=torch.float16)
        for i, key in enumerate(available_blocks):
            block_idx = int(key.split("_block")[1])
            scores[i] = base_score.mean() + block_idx * 0.1
        
        top_indices = scores.argsort(descending=True)[:num_blocks]
        return [available_blocks[idx.item()] for idx in top_indices]

    def apply_blocks(self, input_data, block_keys, output_size):
        output = torch.zeros(input_data.shape[0], input_data.shape[1], output_size, dtype=torch.float16, device=self.device)
        for block_key in block_keys:
            block = self.dispatcher.load_block(self.dispatcher.metadata[block_key])
            block_out_size, block_in_size = block.shape
            block_idx = int(block_key.split("_block")[1])
            
            if block_out_size == output_size:
                start_col = block_idx * block_in_size
                end_col = min(start_col + block_in_size, input_data.shape[-1])
                input_slice = input_data[..., start_col:end_col]
                output += torch.matmul(input_slice, block.t())
            else:
                start_row = block_idx * block_out_size
                end_row = min(start_row + block_out_size, output_size)
                output[..., start_row:end_row] = torch.matmul(input_data, block.t())
            
            del block
        return output

    def clear_memory(self):
        self.dispatcher.virtual_matrix.clear_cache()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()