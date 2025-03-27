import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import os
import gc
import logging
from gguf import GGUFReader

# Настройка логирования
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def format_size(bytes_size):
    if bytes_size >= 1024**3:
        return f"{bytes_size / (1024**3):.2f} GiB"
    elif bytes_size >= 1024**2:
        return f"{bytes_size / (1024**2):.2f} MiB"
    else:
        return f"{bytes_size / 1024:.2f} KiB"

class VirtualMatrix:
    def __init__(self, dispatcher):
        self.dispatcher = dispatcher
        self.device = dispatcher.device
        self.cache = {}
        self.block_size = 4096  # Размер блока для разбиения тензоров

    def get_block(self, tensor_name, block_idx):
        block_key = f"{tensor_name}_block{block_idx}"
        if block_key not in self.cache:
            self.cache[block_key] = self.dispatcher.load_tensor_block(tensor_name, block_idx)
        return self.cache[block_key]

    def embedding(self, input_ids, tensor_name="token_embd.weight"):
        batch_size, seq_len = input_ids.shape
        hidden_size = self.dispatcher.hidden_size
        output = torch.zeros(batch_size, seq_len, hidden_size, dtype=torch.float16, device=self.device)

        unique_tokens = torch.unique(input_ids)
        block_height = self.block_size

        for token in unique_tokens:
            token_id = token.item()
            if token_id >= self.dispatcher.vocab_size:
                continue
            block_idx = token_id // block_height
            local_idx = token_id % block_height
            block = self.get_block(tensor_name, block_idx)
            if local_idx < block.shape[0]:
                token_embedding = block[local_idx]
                mask = (input_ids == token)
                output[mask] = token_embedding.to(self.device)

        self.clear_cache()
        logger.info(f"Embeddings generated: {output.shape}, size: {format_size(output.nbytes)}")
        return output

    def linear(self, input, tensor_name, output_size, input_size, top_k=None):
        batch_size, seq_len, in_features = input.shape
        assert in_features == input_size, f"Input size mismatch: {in_features} != {input_size}"

        num_blocks = (output_size + self.block_size - 1) // self.block_size

        if top_k and tensor_name.startswith("output.weight"):
            # Грубая оценка на первых двух блоках
            coarse_logits = torch.zeros(batch_size, seq_len, min(8192, output_size), dtype=torch.float16, device=self.device)
            for block_idx in range(min(2, num_blocks)):
                block = self.get_block(tensor_name, block_idx)
                start_row = block_idx * self.block_size
                end_row = min(start_row + block.shape[0], output_size)
                coarse_logits[..., start_row:end_row] = torch.matmul(input, block.t())

            coarse_values, coarse_indices = torch.topk(coarse_logits, k=top_k, dim=-1)
            output = torch.zeros(batch_size, seq_len, top_k, dtype=torch.float16, device=self.device)

            for b in range(batch_size):
                for s in range(seq_len):
                    token_ids = coarse_indices[b, s].cpu().numpy()
                    for token_id in token_ids:
                        block_idx = token_id // self.block_size
                        block = self.get_block(tensor_name, block_idx)
                        local_idx = token_id % self.block_size
                        if local_idx < block.shape[0]:
                            output[b, s, token_ids.tolist().index(token_id)] = torch.matmul(
                                input[b, s:s+1], block[local_idx:local_idx+1].t()
                            )
            self.clear_cache()
            logger.info(f"Top-k logits: {output.shape}, indices: {coarse_indices.shape}")
            return output, coarse_indices
        else:
            output = torch.zeros(batch_size, seq_len, output_size, dtype=torch.float16, device=self.device)
            for block_idx in range(num_blocks):
                block = self.get_block(tensor_name, block_idx)
                start_row = block_idx * self.block_size
                end_row = min(start_row + block.shape[0], output_size)
                output[..., start_row:end_row] = torch.matmul(input, block.t())
            self.clear_cache()
            logger.info(f"Linear output: {output.shape}, size: {format_size(output.nbytes)}")
            return output

    def clear_cache(self):
        total_size = sum(t.nbytes for t in self.cache.values()) if self.cache else 0
        self.cache.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info(f"Cache cleared, freed: {format_size(total_size)}")

class ModelDispatcher:
    def __init__(self, model_name, gguf_prefix, split_count, vocab_size, hidden_size, num_layers, num_attention_heads, intermediate_size, key_dim, num_key_value_heads):
        self.model_name = model_name
        self.gguf_prefix = gguf_prefix
        self.split_count = split_count
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.key_dim = key_dim
        self.num_key_value_heads = num_key_value_heads
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tensor_map = {}
        self.readers = {}

        for i in range(1, self.split_count + 1):
            gguf_path = f"{gguf_prefix}-{i:05d}-of-{self.split_count:05d}.gguf"
            if os.path.exists(gguf_path):
                reader = GGUFReader(gguf_path)
                self.readers[gguf_path] = reader
                for tensor in reader.tensors:
                    self.tensor_map[tensor.name] = (gguf_path, tensor)
        logger.info(f"Loaded {len(self.tensor_map)} tensors from {self.split_count} GGUF splits")

    def load_tensor_block(self, tensor_name, block_idx, block_size=4096):
        if tensor_name not in self.tensor_map:
            raise ValueError(f"Tensor {tensor_name} not found in GGUF splits")

        gguf_path, tensor = self.tensor_map[tensor_name]
        shape = tensor.shape
        dtype = torch.float16

        if tensor_name == "token_embd.weight":
            total_rows, total_cols = self.vocab_size, self.hidden_size
        elif "attn_q" in tensor_name or "attn_o" in tensor_name:
            total_rows, total_cols = self.hidden_size, self.hidden_size
        elif "attn_k" in tensor_name or "attn_v" in tensor_name:
            total_rows, total_cols = self.key_dim, self.hidden_size
        elif "ffn_gate" in tensor_name or "ffn_up" in tensor_name:
            total_rows, total_cols = self.intermediate_size, self.hidden_size
        elif "ffn_down" in tensor_name:
            total_rows, total_cols = self.hidden_size, self.intermediate_size
        elif tensor_name == "output.weight":
            total_rows, total_cols = self.vocab_size, self.hidden_size
        else:
            total_rows, total_cols = shape[0], shape[1]

        start_row = block_idx * block_size
        end_row = min(start_row + block_size, total_rows)

        if start_row >= total_rows:
            raise ValueError(f"Block {block_idx} out of range for {tensor_name}")

        if tensor.tensor_type == 12:  # Q4_K
            group_size = 32
            n_elements = total_rows * total_cols
            n_groups = n_elements // group_size
            raw_data = tensor.data

            q_data = np.frombuffer(raw_data[:n_elements // 2], dtype=np.uint8)
            meta_data = np.frombuffer(raw_data[n_elements // 2:], dtype=np.float16)
            d_values = meta_data[0::2]
            m_values = meta_data[1::2]

            q_values = np.zeros(n_elements, dtype=np.uint8)
            for i in range(len(q_data)):
                q_values[2*i] = q_data[i] & 0x0F
                q_values[2*i + 1] = q_data[i] >> 4

            result = np.zeros(n_elements, dtype=np.float16)
            for g in range(n_groups):
                d = d_values[g]
                m = m_values[g]
                for i in range(group_size):
                    idx = g * group_size + i
                    if idx < n_elements:
                        result[idx] = d * (q_values[idx] - m)

            full_tensor = result.reshape(total_rows, total_cols)
            block = full_tensor[start_row:end_row, :]
            block_tensor = torch.from_numpy(block).to(dtype).to(self.device)
        else:
            full_tensor = tensor.data.reshape(total_rows, total_cols)
            block = full_tensor[start_row:end_row, :]
            block_tensor = torch.from_numpy(block).to(dtype).to(self.device)

        logger.info(f"Loaded block {tensor_name}_block{block_idx}, shape: {block_tensor.shape}, size: {format_size(block_tensor.nbytes)}")
        return block_tensor

class MatrixModel(nn.Module):
    def __init__(self, dispatcher):
        super().__init__()
        self.dispatcher = dispatcher
        self.virtual_matrix = VirtualMatrix(dispatcher)
        self.vocab_size = dispatcher.vocab_size
        self.hidden_size = dispatcher.hidden_size
        self.num_layers = dispatcher.num_layers
        self.num_attention_heads = dispatcher.num_attention_heads
        self.intermediate_size = dispatcher.intermediate_size
        self.key_dim = dispatcher.key_dim
        self.num_key_value_heads = dispatcher.num_key_value_heads
        self.device = dispatcher.device

    def forward(self, input_ids, top_k=None):
        if not isinstance(input_ids, torch.Tensor):
            input_ids = torch.tensor(input_ids, dtype=torch.long, device=self.device)
        elif input_ids.dtype != torch.long:
            input_ids = input_ids.long()

        batch_size, seq_len = input_ids.shape
        hidden_states = self.virtual_matrix.embedding(input_ids)

        for layer_idx in range(self.num_layers):
            logger.info(f"Processing layer {layer_idx}")
            q = self.virtual_matrix.linear(hidden_states, f"blk.{layer_idx}.attn_q.weight", self.hidden_size, self.hidden_size)
            k = self.virtual_matrix.linear(hidden_states, f"blk.{layer_idx}.attn_k.weight", self.key_dim, self.hidden_size)
            v = self.virtual_matrix.linear(hidden_states, f"blk.{layer_idx}.attn_v.weight", self.key_dim, self.hidden_size)

            head_dim = self.hidden_size // self.num_attention_heads
            key_head_dim = self.key_dim // self.num_key_value_heads
            heads_per_group = self.num_attention_heads // self.num_key_value_heads

            q = q.view(batch_size, seq_len, self.num_attention_heads, head_dim)
            q = q.view(batch_size, seq_len, self.num_key_value_heads, heads_per_group, head_dim)
            q = q.permute(0, 2, 3, 1, 4)

            k = k.view(batch_size, seq_len, self.num_key_value_heads, key_head_dim)
            k = k.permute(0, 2, 1, 3)

            v = v.view(batch_size, seq_len, self.num_key_value_heads, key_head_dim)
            v = v.permute(0, 2, 1, 3)

            scores = torch.einsum('bhgsd,bhqd->bhgsq', q, k) / (head_dim ** 0.5)
            attn_weights = F.softmax(scores, dim=-1)

            v_expanded = v.unsqueeze(2).expand(-1, -1, heads_per_group, -1, -1)
            attn_output = torch.einsum('bhgsq,bhgsd->bhgsd', attn_weights, v_expanded)

            attn_output = attn_output.permute(0, 3, 1, 2, 4).contiguous()
            attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)
            hidden_states = self.virtual_matrix.linear(attn_output, f"blk.{layer_idx}.attn_o.weight", self.hidden_size, self.hidden_size)

            gate = self.virtual_matrix.linear(hidden_states, f"blk.{layer_idx}.ffn_gate.weight", self.intermediate_size, self.hidden_size)
            up = self.virtual_matrix.linear(hidden_states, f"blk.{layer_idx}.ffn_up.weight", self.intermediate_size, self.hidden_size)
            mlp_output = gate * up
            hidden_states = self.virtual_matrix.linear(mlp_output, f"blk.{layer_idx}.ffn_down.weight", self.hidden_size, self.intermediate_size)

        output = self.virtual_matrix.linear(hidden_states, "output.weight", self.vocab_size, self.hidden_size, top_k=top_k)
        if top_k:
            logits, indices = output
            logger.info(f"Final logits: {logits.shape}, indices: {indices.shape}")
            return logits, indices
        else:
            logits = output
            logger.info(f"Final logits: {logits.shape}")
            return logits, None

class VirtualSpace:
    def __init__(self, gguf_path, model_name="DeepSeek-R1-Distill-Qwen-1.5B", metadata_dir="/workspaces/Veector/data"):
        self.gguf_path = gguf_path
        self.model_name = model_name
        self.metadata_dir = Path(metadata_dir)
        self.dispatcher = None
        self.model = None
        self.tokenizer = None

    def switch_model(self, vocab_size=151936, hidden_size=1536, num_layers=28, num_attention_heads=12, intermediate_size=8960, key_dim=256, num_key_value_heads=2):
        self.dispatcher = ModelDispatcher(
            self.model_name, self.gguf_path, vocab_size, hidden_size, num_layers, num_attention_heads, intermediate_size, key_dim, num_key_value_heads
        )
        self.model = MatrixModel(self.dispatcher)
        logger.info(f"Switched to model: {self.model_name}")

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def forward(self, input_text, top_k=None):
        if not self.tokenizer:
            raise ValueError("Tokenizer not set")
        if not self.model:
            raise ValueError("Model not initialized")
        
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt").to(self.dispatcher.device)
        logits, indices = self.model(input_ids, top_k=top_k)
        return logits, indices

if __name__ == "__main__":
    from transformers import AutoTokenizer

    gguf_prefix = "/workspaces/Veector/data/blocks/DeepSeek-R1-Distill-Qwen-1.5B/DeepSeek-R1-Distill-Qwen-1.5B-split"
    split_count = 43
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-1.5B-Instruct")

    virtual_space = VirtualSpace(gguf_prefix, split_count=split_count)
    virtual_space.set_tokenizer(tokenizer)
    virtual_space.switch_model()

    input_text = "Привет, как дела?"
    logits, indices = virtual_space.forward(input_text, top_k=10)
    print(f"Logits shape: {logits.shape}, Indices shape: {indices.shape if indices is not None else 'None'}")