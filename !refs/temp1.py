import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import gc
import psutil
from collections import defaultdict
from typing import List, Tuple

logger = logging.getLogger(__name__)

class TokenTracker:
    """Отслеживание топовых токенов и их наследников."""
    def __init__(self, top_k: int = 10):
        self.top_k = top_k
        self.token_relevance = {}
        self.token_successors = defaultdict(list)

    def update(self, hidden_states: torch.Tensor, attn_weights: torch.Tensor, layer_idx: int, threshold: float) -> List[int]:
        relevance = torch.norm(hidden_states, dim=-1).squeeze(0)
        self.token_relevance[layer_idx] = relevance

        top_values, top_indices = torch.topk(relevance, k=min(self.top_k, relevance.size(0)))
        for idx in top_indices.tolist():
            attn_to_idx = attn_weights[:, :, idx, :].sum(dim=1).squeeze(0)
            k = min(self.top_k, attn_to_idx.size(0))
            successor_values, successor_indices = torch.topk(attn_to_idx, k=k)
            self.token_successors[idx].extend(successor_indices.tolist())
            logger.debug(f"Layer {layer_idx} attention heatmap:")
            logger.debug(attn_weights.squeeze().tolist())

        active_mask = relevance > threshold
        active_tokens = torch.nonzero(active_mask, as_tuple=True)[0].tolist()
        logger.info(f"Layer {layer_idx}: Top {self.top_k} tokens: {top_indices.tolist()}, Active tokens: {len(active_tokens)}")
        return active_tokens

    def get_active_tokens(self) -> List[int]:
        active_tokens = set()
        for token_idx, successors in self.token_successors.items():
            if self.token_relevance.get(token_idx, torch.tensor(0.0)) > min(self.token_relevance.values()):
                active_tokens.add(token_idx)
                active_tokens.update(successors)
        return sorted(list(active_tokens))

class Observer(nn.Module):
    def __init__(self, dispatcher, tokenizer, max_layers: int = 28, top_k: int = 10):
        super().__init__()
        self.dispatcher = dispatcher
        self.tokenizer = tokenizer  # Явно передаём токенизатор
        self.device = dispatcher.device
        self.hidden_size = dispatcher.hidden_size
        self.num_layers = min(dispatcher.num_layers, max_layers)
        self.num_attention_heads = dispatcher.num_attention_heads
        self.top_k = top_k
        self.temperature_decay = 0.95  # Новое: затухание температуры

        self.query_scorer = nn.Linear(self.hidden_size, 3, dtype=torch.float16)
        self.layer_scorer = nn.Linear(self.hidden_size, self.num_layers, dtype=torch.float16)
        self.room_scorer = nn.Linear(self.hidden_size, self.hidden_size // 256, dtype=torch.float16)
        self.attn_scorer = nn.Linear(self.hidden_size, 1, dtype=torch.float16)

        self.token_tracker = TokenTracker(top_k=self.top_k)
        self.block_cache = {}
        self.to(self.device)
        

    def classify_query(self, hidden_states: torch.Tensor) -> str:
        scores = self.query_scorer(hidden_states.mean(dim=1)).softmax(dim=-1)[0]
        return "light" if scores[0] > max(scores[1], scores[2]) else "medium" if scores[1] > scores[2] else "deep"    

    def select_layers(self, hidden_states: torch.Tensor, strategy: str) -> List[int]:
        scores = self.layer_scorer(hidden_states.mean(dim=1))[0]
        num_active = {"light": 3, "medium": 5, "deep": 7}[strategy]
        return scores.argsort(descending=True)[:num_active].tolist()

    def select_rooms(self, hidden_states: torch.Tensor) -> List[int]:
        scores = self.room_scorer(hidden_states.mean(dim=1))[0]
        num_rooms = self.hidden_size // 256
        return list(range(num_rooms))

    def apply_attention(self, hidden_states: torch.Tensor, layer_idx: int, active_rooms: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = hidden_states.shape
        mask = torch.triu(torch.ones(batch_size, 1, seq_len, seq_len, device=self.device, dtype=torch.float16), diagonal=1) * -1e9

        block_keys = [f"{self.dispatcher.model_name}_layer{layer_idx}_self_attn_{proj}_proj_weight_block0" for proj in ['q', 'k', 'v', 'o']]
        blocks = [self._load_block_with_cache(key) for key in block_keys]
        q_block, k_block, v_block, o_block = blocks

        key_dim = k_block.shape[0]
        room_size = key_dim
        num_rooms = self.hidden_size // key_dim

        q_rooms = [q_block[i * room_size:(i + 1) * room_size, :] for i in range(num_rooms)]
        k_rooms = [k_block for _ in range(num_rooms)]
        v_rooms = [v_block for _ in range(num_rooms)]
        o_rooms = [o_block[i * room_size:(i + 1) * room_size, :] for i in range(num_rooms)]

        q_outs, k_outs, v_outs = [], [], []
        for room_idx in active_rooms:
            q = torch.matmul(hidden_states, q_rooms[room_idx].t())
            k = torch.matmul(hidden_states, k_rooms[room_idx].t())
            v = torch.matmul(hidden_states, v_rooms[room_idx].t())
            q_outs.append(q)
            k_outs.append(k)
            v_outs.append(v)

        q = torch.cat(q_outs, dim=-1)
        k = torch.cat(k_outs, dim=-1)
        v = torch.cat(v_outs, dim=-1)

        total_dim = key_dim * len(active_rooms)
        head_dim = total_dim // self.num_attention_heads
        q = q.view(batch_size, seq_len, self.num_attention_heads, head_dim).permute(0, 2, 1, 3)
        k = k.view(batch_size, seq_len, self.num_attention_heads, head_dim).permute(0, 2, 1, 3)
        v = v.view(batch_size, seq_len, self.num_attention_heads, head_dim).permute(0, 2, 1, 3)

        scores = torch.matmul(q, k.transpose(-2, -1)).to(torch.float16) / (head_dim ** 0.5)
        scores = scores + mask
        attn_weights = F.softmax(scores, dim=-1).to(torch.float16)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, total_dim)

        o_active = torch.cat([o_rooms[i] for i in active_rooms], dim=0)
        o_out = torch.matmul(attn_output, o_active.to(self.device))

        return o_out, attn_weights

    def process_layer(self, hidden_states: torch.Tensor, layer_idx: int, threshold: float) -> torch.Tensor:
        active_rooms = self.select_rooms(hidden_states)
        attn_output, attn_weights = self.apply_attention(hidden_states, layer_idx, active_rooms)
        active_tokens = self.token_tracker.update(hidden_states, attn_weights, layer_idx, threshold)

        mask = torch.ones(hidden_states.shape[1], device=self.device, dtype=torch.float16)
        for idx in range(hidden_states.shape[1]):
            if idx not in active_tokens:
                mask[idx] = 0.0
        hidden_states = attn_output * mask.unsqueeze(0).unsqueeze(-1)
        return hidden_states

    def analyze_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = F.layer_norm(hidden_states, normalized_shape=[hidden_states.shape[-1]])
        scores = self.attn_scorer(hidden_states).sigmoid()
        return hidden_states * scores

    def _load_block_with_cache(self, block_key: str) -> torch.Tensor:
        if block_key not in self.block_cache:
            self.block_cache[block_key] = self.dispatcher.load_block(block_key)
        return self.block_cache[block_key]

    def dynamic_threshold(self, relevance: torch.Tensor) -> float:
        sorted_relevance = torch.sort(relevance, descending=True)[0]
        return sorted_relevance[int(len(sorted_relevance) * 0.4)].item() if len(sorted_relevance) > 0 else 0.0

    def beam_search(self, input_ids: torch.Tensor, beam_width: int = 3, 
                  max_length: int = 50, temperature: float = 0.6) -> Tuple[torch.Tensor, float]:
        """Улучшенный Beam Search с контролем температуры"""
        hidden_states = self.dispatcher.virtual_matrix.embedding(input_ids, f"{self.dispatcher.model_name}_embed")
        strategy = self.classify_query(hidden_states)
        active_layers = self.select_layers(hidden_states, strategy)
        
        beams = [(input_ids, 0.0)]
        eos_token_id = self.tokenizer.eos_token_id
        
        for step in range(max_length - input_ids.shape[1]):
            new_beams = []
            current_temp = temperature * (self.temperature_decay ** step)  # Затухание температуры
            
            for seq, log_prob in beams:
                next_hidden = self.dispatcher.virtual_matrix.embedding(seq, f"{self.dispatcher.model_name}_embed")
                
                for layer_idx in active_layers:
                    threshold = self.dynamic_threshold(torch.norm(next_hidden, dim=-1).squeeze(0))
                    next_hidden = self.process_layer(next_hidden, layer_idx, threshold)
                    next_hidden = self.analyze_states(next_hidden)
                
                logits = self._calculate_logits(next_hidden)
                probs = F.softmax(logits[:, -1, :] / current_temp, dim=-1)
                
                top_probs, top_ids = torch.topk(probs, k=beam_width)
                for prob, token_id in zip(top_probs[0], top_ids[0]):
                    if token_id == eos_token_id:
                        new_seq = seq
                    else:
                        new_seq = torch.cat([seq, token_id.unsqueeze(0).unsqueeze(0)], dim=1)
                    
                    new_log_prob = log_prob + torch.log(prob).item()
                    new_beams.append((new_seq, new_log_prob))
            
            beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]
            if beams[0][0][0, -1].item() == eos_token_id:
                break
        
        return beams[0][0], 1.0

    def forward(self, input_ids: torch.Tensor, temperature: float = 0.6, 
               max_length: int = 50, beam_width: int = None) -> Tuple[torch.Tensor, float]:
        """Основной цикл генерации с улучшенным контролем"""
        if beam_width and beam_width > 1:
            return self.beam_search(input_ids, beam_width=beam_width, 
                                   max_length=max_length, temperature=temperature)
        
        hidden_states = self.dispatcher.virtual_matrix.embedding(input_ids, f"{self.dispatcher.model_name}_embed")
        strategy = self.classify_query(hidden_states)
        active_layers = self.select_layers(hidden_states, strategy)
        
        generated_ids = input_ids.clone()
        eos_token_id = self.tokenizer.eos_token_id
        
        for step in range(max_length - input_ids.shape[1]):
            current_temp = temperature * (self.temperature_decay ** step)  # Затухание температуры
            next_token_hidden = hidden_states[:, -1, :]
            
            logits = self._calculate_logits(next_token_hidden)
            probs = F.softmax(logits / current_temp, dim=-1)
            
            next_token_id = torch.argmax(probs, dim=-1)
            confidence = probs.max().item()
            
            # Декодирование и логирование
            current_token = self.tokenizer.decode([next_token_id.item()])
            logger.info(f"Step {step+1}: Next token: {current_token} (ID: {next_token_id.item()}), "
                        f"Confidence: {confidence:.2f}, Temperature: {current_temp:.2f}")
            
            if confidence < 0.85 or next_token_id == eos_token_id:
                break
            
            generated_ids = torch.cat([generated_ids, next_token_id.unsqueeze(0).unsqueeze(0)], dim=1)
            new_hidden = self.dispatcher.virtual_matrix.embedding(next_token_id.unsqueeze(0), 
                                                                f"{self.dispatcher.model_name}_embed")
            hidden_states = torch.cat([hidden_states, new_hidden], dim=1)
        
        return generated_ids, confidence

    def _calculate_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Улучшенный расчет логитов с проверкой блоков"""
        output_blocks = sorted([key for key in self.dispatcher.metadata if "_output" in key],
                               key=lambda x: int(x.split("_block")[1]))
        
        logits = []
        for block_key in output_blocks:
            block = self._load_block_with_cache(block_key)
            if block is None:
                logger.warning(f"Блок {block_key} не загружен")
                continue
            block_logits = torch.matmul(hidden_states, block.t())
            logits.append(block_logits)
        
        return torch.cat(logits, dim=-1)

    def clear_memory(self):
        self.block_cache.clear()
        self.dispatcher.virtual_matrix.clear_cache()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info(f"Очистка памяти, RAM: {psutil.Process().memory_info().rss / 1024**2:.2f} MB")

    def reset(self):
        self.token_tracker = TokenTracker(top_k=self.top_k)
        self.block_cache.clear()