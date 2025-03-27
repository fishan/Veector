import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
from virtual_space import VirtualSpace
from model_manager import ModelManager
from core import Veector
import logging
import gc
import os

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x

class QuantizedModelTest:
    def __init__(self):
        self.veector = Veector(use_memory=False, ipfs_enabled=False)
        self.model_manager = ModelManager(self.veector, ipfs_enabled=False)
        self.tokenizer = AutoTokenizer.from_pretrained("/workspaces/Veector/data/models/DeepSeek-R1-Distill-Qwen-1.5B")
        
        self.virtual_space = VirtualSpace(
            veector=self.veector,
            use_ipfs=False,
            model_manager=self.model_manager,
            metadata_dir="/workspaces/Veector/data"
        )
        
        self.model_name = "DeepSeek-R1-Distill-Qwen-1.5B"
        self.vocab_size = 151936
        self.hidden_size = 1536
        self.num_layers = 28
        self.num_attention_heads = 12
        self.intermediate_size = 8960
        self.key_dim = 256
        self.num_key_value_heads = 2
        self.rms_norm_eps = 1e-6
        self.dtype = torch.float16
        self.max_length = 50
        self.temperature = 0.7
        self.top_k = 10
        
        self.virtual_space.switch_model(
            model_name=self.model_name,
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            key_dim=self.key_dim,
            num_key_value_heads=self.num_key_value_heads,
            split_prefix="DeepSeek-R1-Distill-Qwen-1.5B-split",
            split_count=43
        )
        logger.info(f"Переключено на модель: {self.model_name} в GGUF-режиме с 43 сплитами")
        
        self.device = torch.device("cpu")
        self.norm = RMSNorm(self.hidden_size, eps=self.rms_norm_eps).to(self.device).to(self.dtype)

    def process_layer(self, hidden_states, layer_idx, past_kv=None):
        logger.debug(f"Обработка слоя {layer_idx}, входная форма: {hidden_states.shape}")
        hidden_states = self.norm(hidden_states).to(self.dtype)
        
        q = self.virtual_space.virtual_matrix.linear(
            hidden_states, f"layers.{layer_idx}.attn_q",
            self.hidden_size, self.hidden_size
        ).to(self.dtype)
        logger.debug(f"Q для слоя {layer_idx}, форма: {q.shape}")
        
        k = self.virtual_space.virtual_matrix.linear(
            hidden_states, f"layers.{layer_idx}.attn_k",
            self.key_dim, self.hidden_size
        ).to(self.dtype)
        logger.debug(f"K для слоя {layer_idx}, форма: {k.shape}")
        
        v = self.virtual_space.virtual_matrix.linear(
            hidden_states, f"layers.{layer_idx}.attn_v",
            self.key_dim, self.hidden_size
        ).to(self.dtype)
        logger.debug(f"V для слоя {layer_idx}, форма: {v.shape}")
        
        batch_size, seq_len = hidden_states.shape[:2]
        head_dim = self.hidden_size // self.num_attention_heads
        key_head_dim = self.key_dim // self.num_key_value_heads
        heads_per_group = self.num_attention_heads // self.num_key_value_heads
        
        q = q.view(batch_size, seq_len, self.num_key_value_heads, heads_per_group, head_dim).permute(0, 2, 1, 3, 4)
        k = k.view(batch_size, seq_len, self.num_key_value_heads, key_head_dim).permute(0, 2, 1, 3)
        v = v.view(batch_size, seq_len, self.num_key_value_heads, key_head_dim).permute(0, 2, 1, 3)
        logger.debug(f"Q после reshape: {q.shape}, K: {k.shape}, V: {v.shape}")
        
        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
            logger.debug(f"Обновленные K: {k.shape}, V: {v.shape} с past_kv")
        
        attn_output = torch.zeros(batch_size, seq_len, self.num_key_value_heads, heads_per_group, head_dim, 
                                 device=self.device, dtype=self.dtype)
        for group_idx in range(self.num_key_value_heads):
            q_group = q[:, group_idx]
            k_group = k[:, group_idx]
            v_group = v[:, group_idx]
            scores = torch.matmul(q_group, k_group.transpose(-2, -1)) / (head_dim ** 0.5)
            attn_weights = torch.softmax(scores, dim=-1)
            attn_output[:, :, group_idx] = torch.matmul(attn_weights, v_group)
            logger.debug(f"Attn weights для группы {group_idx}: {attn_weights.shape}, output: {attn_output.shape}")
        
        attn_output = attn_output.permute(0, 3, 1, 2, 4).contiguous().view(batch_size, seq_len, self.hidden_size)
        attn_output = self.virtual_space.virtual_matrix.linear(
            attn_output, f"layers.{layer_idx}.attn_output",
            self.hidden_size, self.hidden_size
        ).to(self.dtype)
        logger.debug(f"Attn output после linear: {attn_output.shape}")
        hidden_states = hidden_states + attn_output
        
        hidden_states = self.norm(hidden_states).to(self.dtype)
        gate = self.virtual_space.virtual_matrix.linear(
            hidden_states, f"layers.{layer_idx}.ffn_gate",
            self.intermediate_size, self.hidden_size
        ).to(self.dtype)
        logger.debug(f"Gate для слоя {layer_idx}: {gate.shape}")
        
        up = self.virtual_space.virtual_matrix.linear(
            hidden_states, f"layers.{layer_idx}.ffn_up",
            self.intermediate_size, self.hidden_size
        ).to(self.dtype)
        logger.debug(f"Up для слоя {layer_idx}: {up.shape}")
        
        mlp_output = gate * up
        down = self.virtual_space.virtual_matrix.linear(
            mlp_output, f"layers.{layer_idx}.ffn_down",
            self.hidden_size, self.intermediate_size
        ).to(self.dtype)
        logger.debug(f"Down для слоя {layer_idx}: {down.shape}")
        
        hidden_states = hidden_states + down
        logger.info(f"Обработан слой {layer_idx}, форма hidden_states: {hidden_states.shape}")
        
        return hidden_states, (k, v)

    def process_input(self, text):
        try:
            prompt_template = "<｜User｜>{question}<｜Assistant｜>"
            input_text = prompt_template.format(question=text)
            input_ids = self.tokenizer.encode(input_text, return_tensors="pt", add_special_tokens=False).to(self.device).long()
            
            batch_size, seq_len = input_ids.shape
            logger.info(f"Input shape: {input_ids.shape}")
            
            with torch.no_grad():
                hidden_states = self.virtual_space.virtual_matrix.embedding(
                    input_ids, "token_embd"
                ).to(self.dtype)
                logger.info(f"Форма эмбеддингов: {hidden_states.shape}")
                
                past_kv_cache = [None] * self.num_layers
                full_input_ids = input_ids.clone()
                
                for layer_idx in range(self.num_layers):
                    hidden_states, past_kv = self.process_layer(hidden_states, layer_idx, past_kv_cache[layer_idx])
                    past_kv_cache[layer_idx] = past_kv
                
                generated_ids = input_ids.clone()
                for step in range(self.max_length):
                    hidden_states = self.norm(hidden_states).to(self.dtype)
                    logits, coarse_indices = self.virtual_space.virtual_matrix.linear(
                        hidden_states, "output",
                        self.vocab_size, self.hidden_size, top_k=self.top_k
                    ).to(self.dtype)
                    logger.debug(f"Logits: {logits.shape}, coarse_indices: {coarse_indices.shape}")
                    
                    next_token_logits = logits[:, -1, :] / self.temperature
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = coarse_indices[:, -1, torch.multinomial(probs, num_samples=1)[0]].unsqueeze(0).unsqueeze(0)
                    logger.debug(f"Следующий токен: {next_token}")
                    
                    generated_ids = личноtorch.cat([generated_ids, next_token], dim=-1)
                    full_input_ids = torch.cat([full_input_ids, next_token], dim=-1)
                    
                    hidden_states = self.virtual_space.virtual_matrix.embedding(
                        next_token, "token_embd"
                    ).to(self.dtype)
                    
                    for layer_idx in range(self.num_layers):
                        hidden_states, past_kv = self.process_layer(hidden_states, layer_idx, past_kv_cache[layer_idx])
                        past_kv_cache[layer_idx] = past_kv
                    
                    if next_token.item() == self.tokenizer.encode("<｜end▁of▁sentence｜>", add_special_tokens=False)[0]:
                        logger.debug("Обнаружен конец последовательности")
                        break
                
                output_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=False)
                logger.info(f"Ответ: {output_text}")
                return output_text
        
        except Exception as e:
            logger.error(f"Ошибка: {str(e)}")
            raise
        
        finally:
            self.virtual_space.clear_memory()
            gc.collect()

if __name__ == "__main__":
    tester = QuantizedModelTest()
    input_text = "Hello!"
    output = tester.process_input(input_text)
    print(f"Ответ модели: {output}")