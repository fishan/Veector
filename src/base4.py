import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
from virtual_space import VirtualSpace
from model_manager import ModelManager
from core import Veector
import logging
import gc
import psutil
import os

# Настройка логирования с сохранением в файл
log_file = "inference.log"
if os.path.exists(log_file):
    os.remove(log_file)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def get_memory_usage():
    """Получение текущего использования памяти."""
    process = psutil.Process()
    ram_mb = process.memory_info().rss / 1024**2
    gpu_mb = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
    return ram_mb, gpu_mb

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x

class FullModelTest:
    def __init__(self, skip_layers=None):
        self.veector = Veector(use_memory=False, ipfs_enabled=False)
        self.model_manager = ModelManager(self.veector, ipfs_enabled=False)
        self.tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
        self.virtual_space = VirtualSpace(
            veector=self.veector,
            use_ipfs=False,
            model_manager=self.model_manager
        )
        
        # Параметры модели
        self.model_name = "DeepSeek-R1-Distill-Qwen-1.5B"
        self.vocab_size = 151936
        self.hidden_size = 1536
        self.num_layers = 28
        self.skip_layers = skip_layers if skip_layers is not None else [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26]
        self.active_layers = self.num_layers - len(self.skip_layers)
        self.num_attention_heads = 12
        self.intermediate_size = 8960
        self.key_dim = 256
        self.num_key_value_heads = 2
        self.rms_norm_eps = 1e-6
        self.dtype = torch.float16
        self.max_length = 5
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
            num_key_value_heads=self.num_key_value_heads
        )
        logger.info(f"Переключено на модель: {self.model_name}, пропущенные слои: {self.skip_layers}, активных слоев: {self.active_layers}")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.norm = RMSNorm(self.hidden_size, eps=self.rms_norm_eps).to(self.device).to(self.dtype)

    def process_layer(self, hidden_states, layer_idx, past_kv=None, input_ids=None):
        if layer_idx in self.skip_layers:
            logger.info(f"Пропуск слоя {layer_idx}")
            # Если слой пропущен, возвращаем hidden_states и past_kv (или None, если кэша нет)
            k, v = past_kv if past_kv is not None else (None, None)
            return hidden_states, (k, v)
        
        hidden_states = self.norm(hidden_states).to(self.dtype)
        
        q = self.virtual_space.virtual_matrix.linear(
            hidden_states, f"{self.model_name}_layer{layer_idx}_self_attn_q_proj_weight", 
            self.hidden_size, self.hidden_size
        ).to(self.dtype)
        k = self.virtual_space.virtual_matrix.linear(
            hidden_states, f"{self.model_name}_layer{layer_idx}_self_attn_k_proj_weight", 
            self.key_dim, self.hidden_size
        ).to(self.dtype)
        v = self.virtual_space.virtual_matrix.linear(
            hidden_states, f"{self.model_name}_layer{layer_idx}_self_attn_v_proj_weight", 
            self.key_dim, self.hidden_size
        ).to(self.dtype)
        
        batch_size, seq_len = hidden_states.shape[:2]
        head_dim = self.hidden_size // self.num_attention_heads
        key_head_dim = self.key_dim // self.num_key_value_heads
        heads_per_group = self.num_attention_heads // self.num_key_value_heads
        
        q = q.view(batch_size, seq_len, self.num_key_value_heads, heads_per_group, head_dim).permute(0, 2, 1, 3, 4)
        k = k.view(batch_size, seq_len, self.num_key_value_heads, key_head_dim).permute(0, 2, 1, 3)
        v = v.view(batch_size, seq_len, self.num_key_value_heads, key_head_dim).permute(0, 2, 1, 3)
        
        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        
        attn_output = torch.zeros(batch_size, seq_len, self.num_key_value_heads, heads_per_group, head_dim, device=self.device, dtype=self.dtype)
        if layer_idx == 0:  # Логировать только для первого слоя
            attn_weights_log = {}
            input_ids_flat = input_ids[0] if input_ids.dim() == 2 else input_ids
            total_seq_len = k.shape[2]
            k_max = min(5, total_seq_len)
        
        for group_idx in range(self.num_key_value_heads):
            q_group = q[:, group_idx]
            k_group = k[:, group_idx]
            v_group = v[:, group_idx]
            scores = torch.matmul(q_group, k_group.transpose(-2, -1)) / (head_dim ** 0.5)
            attn_weights = torch.softmax(scores, dim=-1)
            attn_output[:, :, group_idx] = torch.matmul(attn_weights, v_group)
            
            if layer_idx == 0:  # Логировать только для первого слоя
                weights = attn_weights[0, -1]
                top_k_values, top_k_indices = torch.topk(weights, k=k_max, dim=-1)
                tokens = [self.tokenizer.decode([input_ids_flat[idx.item()].item()]) for idx in top_k_indices[0]]
                attn_weights_log[f"head_{group_idx}"] = {"tokens": tokens, "values": top_k_values.tolist()}
        
        if layer_idx == 0:
            logger.info(f"Слой {layer_idx} - Топ-{k_max} токенов внимания: {attn_weights_log}")
        
        attn_output = attn_output.permute(0, 3, 1, 2, 4).contiguous().view(batch_size, seq_len, self.hidden_size)
        attn_output = self.virtual_space.virtual_matrix.linear(
            attn_output, f"{self.model_name}_layer{layer_idx}_self_attn_o_proj_weight", 
            self.hidden_size, self.hidden_size
        ).to(self.dtype)
        hidden_states = hidden_states + attn_output
        
        hidden_states = self.norm(hidden_states).to(self.dtype)
        gate = self.virtual_space.virtual_matrix.linear(
            hidden_states, f"{self.model_name}_layer{layer_idx}_mlp_gate_proj_weight", 
            self.intermediate_size, self.hidden_size
        ).to(self.dtype)
        up = self.virtual_space.virtual_matrix.linear(
            hidden_states, f"{self.model_name}_layer{layer_idx}_mlp_up_proj_weight", 
            self.intermediate_size, self.hidden_size
        ).to(self.dtype)
        mlp_output = gate * up
        down = self.virtual_space.virtual_matrix.linear(
            mlp_output, f"{self.model_name}_layer{layer_idx}_mlp_down_proj_weight", 
            self.hidden_size, self.intermediate_size
        ).to(self.dtype)
        hidden_states = hidden_states + down
        
        del q, k, v, gate, up, mlp_output, down, attn_output
        gc.collect()
        torch.cuda.empty_cache()
        
        return hidden_states, (k, v)
    def process_input(self, text):
        try:
            # Формируем входной шаблон
            prompt_template = "<｜User｜>{question}<｜Assistant｜>"
            input_text = prompt_template.format(question=text)
            input_ids = self.tokenizer.encode(input_text, return_tensors="pt", add_special_tokens=False).to(self.device).long()
            if torch.any(input_ids >= self.vocab_size):
                raise ValueError(f"Токены превышают vocab_size ({self.vocab_size})")
            
            batch_size, seq_len = input_ids.shape
            logger.info(f"Input shape: {input_ids.shape}, tokens: {seq_len}")
            tokens = [self.tokenizer.decode([token.item()]) for token in input_ids[0]]
            logger.info(f"Входные токены: {tokens}")
            
            ram_before, gpu_before = get_memory_usage()
            logger.info(f"Память до инференса: RAM {ram_before:.2f} MB, GPU {gpu_before:.2f} MB")
            
            with torch.no_grad():
                hidden_states = self.virtual_space.virtual_matrix.embedding(
                    input_ids, f"{self.model_name}_embed"
                ).to(self.dtype)
                logger.info(f"Форма эмбеддингов: {hidden_states.shape}, тип: {hidden_states.dtype}")
                
                past_kv_cache = [None] * self.num_layers
                full_input_ids = input_ids.clone()
                
                # Начальная обработка всех слоёв с входным контекстом
                for layer_idx in range(self.num_layers):
                    logger.info(f"Обработка слоя {layer_idx} (начальный проход)")
                    hidden_states, past_kv = self.process_layer(hidden_states, layer_idx, past_kv_cache[layer_idx], full_input_ids)
                    past_kv_cache[layer_idx] = past_kv
                    ram_after, gpu_after = get_memory_usage()
                    logger.info(f"Память после слоя {layer_idx}: RAM {ram_after:.2f} MB, GPU {gpu_after:.2f} MB")
                
                # Начинаем генерацию с <｜Assistant｜>
                assistant_token_id = self.tokenizer.encode("<｜Assistant｜>", add_special_tokens=False)[0]
                generated_ids = torch.tensor([[assistant_token_id]], dtype=torch.long, device=self.device)
                hidden_states = self.virtual_space.virtual_matrix.embedding(
                    generated_ids, f"{self.model_name}_embed"
                ).to(self.dtype)
                seen_tokens = set([assistant_token_id])
                
                # Генерация
                for step in range(self.max_length - 1):  # -1, так как <｜Assistant｜> уже есть
                    hidden_states = self.norm(hidden_states).to(self.dtype)
                    output_blocks = sorted(
                        [k for k in self.virtual_space.dispatcher.metadata.keys() if k.startswith(f"{self.model_name}_output")],
                        key=lambda x: int(x.split("_block")[1])
                    )
                    logits = torch.zeros(
                        batch_size, hidden_states.shape[1], self.vocab_size, 
                        dtype=self.dtype, device=self.device
                    )
                    for block_key in output_blocks:
                        block = self.virtual_space.dispatcher.load_block(block_key).to(self.dtype)
                        block_out_size, _ = block.shape
                        block_idx = int(block_key.split("_block")[1])
                        start_row = block_idx * block_out_size
                        end_row = min(start_row + block_out_size, self.vocab_size)
                        logits[:, :, start_row:end_row] = torch.matmul(hidden_states, block.t())
                        del block
                    
                    next_token_logits = logits[:, -1, :] / self.temperature
                    top_k_logits, top_k_indices = torch.topk(next_token_logits, self.top_k, dim=-1)
                    probs = F.softmax(top_k_logits, dim=-1)
                    next_token_idx = torch.multinomial(probs, num_samples=1)
                    next_token = top_k_indices[0, next_token_idx[0]].unsqueeze(0).unsqueeze(0)
                    
                    while next_token.item() in seen_tokens and len(seen_tokens) < self.vocab_size:
                        next_token_idx = torch.multinomial(probs, num_samples=1)
                        next_token = top_k_indices[0, next_token_idx[0]].unsqueeze(0).unsqueeze(0)
                    seen_tokens.add(next_token.item())
                    
                    generated_ids = torch.cat([generated_ids, next_token], dim=-1)
                    full_input_ids = torch.cat([full_input_ids, next_token], dim=-1)
                    
                    new_hidden = self.virtual_space.virtual_matrix.embedding(
                        next_token, f"{self.model_name}_embed"
                    ).to(self.dtype)
                    hidden_states = new_hidden
                    
                    for layer_idx in range(self.num_layers):
                        hidden_states, past_kv = self.process_layer(hidden_states, layer_idx, past_kv_cache[layer_idx], full_input_ids)
                        past_kv_cache[layer_idx] = past_kv
                    
                    token_word = self.tokenizer.decode([next_token.item()])
                    logger.info(f"Шаг {step + 1}: Токен ID: {next_token.item()}, Слово: {token_word}")
                    
                    if next_token.item() == self.tokenizer.encode("<｜end▁of▁sentence｜>", add_special_tokens=False)[0]:
                        break
                
                output_tokens = [self.tokenizer.decode([token_id.item()]) for token_id in generated_ids[0]]
                output_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                
                logger.info(f"Итоговые токены (ID): {generated_ids.tolist()}")
                logger.info(f"Итоговые токены (слова): {output_tokens}")
                logger.info(f"Ответ: {output_text}")
                
                return output_text
        
        except Exception as e:
            logger.error(f"Ошибка: {str(e)}")
            raise
        
        finally:
            self.virtual_space.clear_memory()
            gc.collect()
            torch.cuda.empty_cache()
            ram_final, gpu_final = get_memory_usage()
            logger.info(f"Память после очистки: RAM {ram_final:.2f} MB, GPU {gpu_final:.2f} MB")

if __name__ == "__main__":
    tester = FullModelTest(skip_layers=[2, 4, 6, 7, 8, 10, 12, 13, 14, 16, 18, 19, 20, 22, 23, 24, 26])
    input_text = "Как дела?"
    output = tester.process_input(input_text)
    print(f"Ответ модели: {output}")