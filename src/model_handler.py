import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer
from model_manager import ModelManager
from virtual_space import ModelDispatcher, VirtualSpace
from core import Veector
import logging
import gc
import psutil
from pathlib import Path
from typing import List, Tuple

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

class TokenTracker:
    def __init__(self, vocab_size: int, max_active_tokens: int = 10, decay_rate: float = 0.9):
        self.vocab_size = vocab_size
        self.max_active_tokens = max_active_tokens
        self.decay_rate = decay_rate
        self.token_scores = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def update(self, token_scores: torch.Tensor, step: int) -> List[int]:
        token_scores = token_scores.cpu().detach()
        if step == 0:
            self.token_scores = {}
        
        for token_id in range(min(self.vocab_size, token_scores.shape[-1])):
            score = token_scores[token_id].item() if token_id < token_scores.shape[-1] else 0.0
            if score > 0 and not torch.isnan(torch.tensor(score)):
                self.token_scores[token_id] = self.token_scores.get(token_id, 0.0) * self.decay_rate + score
        
        sorted_tokens = sorted(self.token_scores.items(), key=lambda x: x[1], reverse=True)
        active_tokens = [token_id for token_id, _ in sorted_tokens[:self.max_active_tokens]]
        logger.debug(f"Step {step}: Active tokens: {active_tokens}")
        return active_tokens

class ModelHandler:
    def __init__(self, model_name: str, tensor_dir: str, vocab_size: int, hidden_size: int, num_layers: int):
        self.model_name = model_name
        self.tensor_dir = tensor_dir
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.attention_layers = list(range(num_layers))
        logger.info(f"Используем слои внимания: {self.attention_layers}")

        self.veector = Veector(use_memory=False, ipfs_enabled=False)
        self.model_manager = ModelManager(self.veector, ipfs_enabled=False, model_dir="/workspaces/Veector/data")
        self.model_manager.load_pre_split_model(model_name, tensor_dir, vocab_size, hidden_size, num_layers)

        tokenizer_path = Path(tensor_dir)
        if not tokenizer_path.exists():
            raise FileNotFoundError(f"Папка с моделью {tensor_dir} не найдена")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        logger.info("Токенизатор загружен из папки модели.")

        metadata_path = Path(tensor_dir) / f"{model_name}_metadata.json"
        self.dispatcher = ModelDispatcher(model_name, metadata_path, vocab_size, hidden_size, num_layers)
        self.token_tracker = TokenTracker(vocab_size)

        self.num_attention_heads = 12
        self.num_key_value_heads = 2
        self.head_dim = self.hidden_size // self.num_attention_heads  # 128
        self.key_head_dim = 256 // self.num_key_value_heads  # 128
        self.heads_per_group = self.num_attention_heads // self.num_key_value_heads  # 6

        self.log_memory("После инициализации")

    def log_memory(self, stage: str):
        process = psutil.Process()
        ram_usage = process.memory_info().rss / 1024**2
        logger.debug(f"Память на этапе '{stage}': {ram_usage:.2f} MB")
        if torch.cuda.is_available():
            logger.debug(f"GPU память: {torch.cuda.memory_allocated()/1024**2:.2f} MB выделено, "
                        f"{torch.cuda.memory_reserved()/1024**2:.2f} MB зарезервировано")

    def preprocess_text(self, text: str, max_length: int = 16) -> np.ndarray:
        PROMPT_TEMPLATE = "<｜User｜>{message}<｜Assistant｜>"
        formatted_text = PROMPT_TEMPLATE.format(message=text)
        inputs = self.tokenizer(formatted_text, return_tensors="np", max_length=max_length, truncation=True)
        input_ids = inputs["input_ids"]
        logger.debug(f"Preprocessed input_ids: {input_ids.tolist()}")
        return input_ids.astype(np.int32)

    def _embed_input(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        hidden_states = torch.zeros(batch_size, seq_len, self.hidden_size, device=self.device, dtype=torch.float16)
        logger.debug(f"Начало _embed_input с input_ids shape: {input_ids.shape}")
        
        try:
            embed_blocks = self.dispatcher.get_embedding_blocks(input_ids)
            logger.debug(f"Найдено {len(embed_blocks)} нужных блоков эмбеддингов: {sorted(embed_blocks)}")
        except Exception as e:
            logger.error(f"Ошибка в get_embedding_blocks: {e}")
            raise

        for block_key in sorted(embed_blocks, key=lambda x: int(x.split("_block")[1])):
            block_info = self.dispatcher.metadata[block_key]
            block = self.dispatcher.load_block(block_info)
            logger.debug(f"Загружен блок {block_key} с shape: {block.shape}")
            block_height = block.shape[0]
            start_idx = int(block_key.split("_block")[1]) * block_height
            end_idx = min(start_idx + block_height, self.vocab_size)
            
            mask = (input_ids >= start_idx) & (input_ids < end_idx)
            if mask.any():
                indices = input_ids[mask] - start_idx
                hidden_states[mask] = block[indices]
                logger.debug(f"Обновлён hidden_states для блока {block_key}, обработано токенов: {mask.sum().item()}")
            del block
            gc.collect()
        
        logger.debug(f"Embeddings shape: {hidden_states.shape}, min: {hidden_states.min().item()}, max: {hidden_states.max().item()}")
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        return hidden_states

    def _compute_attention_layer(self, hidden_states: torch.Tensor, layer_idx: int, past_kv: List[Tuple[torch.Tensor, torch.Tensor]] = None):
        batch_size, seq_len, _ = hidden_states.shape
        logger.debug(f"Слой {layer_idx}: hidden_states shape: {hidden_states.shape}")
        q_key = f"{self.model_name}_layer{layer_idx}_self_attn_q_proj_weight_block0"
        k_key = f"{self.model_name}_layer{layer_idx}_self_attn_k_proj_weight_block0"
        v_key = f"{self.model_name}_layer{layer_idx}_self_attn_v_proj_weight_block0"
        o_key = f"{self.model_name}_layer{layer_idx}_self_attn_o_proj_weight_block0"
        q_block = self.dispatcher.load_block(self.dispatcher.metadata.get(q_key, q_key))
        k_block = self.dispatcher.load_block(self.dispatcher.metadata.get(k_key, k_key))
        v_block = self.dispatcher.load_block(self.dispatcher.metadata.get(v_key, v_key))
        o_block = self.dispatcher.load_block(self.dispatcher.metadata.get(o_key, o_key))
        q = F.linear(hidden_states, q_block).view(batch_size, seq_len, self.num_attention_heads, self.head_dim)
        k = F.linear(hidden_states, k_block).view(batch_size, seq_len, self.num_key_value_heads, self.key_head_dim)
        v = F.linear(hidden_states, v_block).view(batch_size, seq_len, self.num_key_value_heads, self.key_head_dim)
        logger.debug(f"Слой {layer_idx}: q shape: {q.shape}, k shape: {k.shape}, v shape: {v.shape}")
        q = q.view(batch_size, seq_len, self.num_key_value_heads, self.heads_per_group, self.head_dim).permute(0, 2, 3, 1, 4)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        if past_kv and len(past_kv) > layer_idx and past_kv[layer_idx] is not None:
            past_k, past_v = past_kv[layer_idx]
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
            logger.debug(f"Слой {layer_idx}: Используем past_kv, k shape: {k.shape}, v shape: {v.shape}")
            kv_seq_len = k.shape[2]
            mask = torch.zeros(seq_len, kv_seq_len, device=self.device).bool()
            mask[:, :-seq_len] = True  # Маскируем всё, кроме текущей последовательности
        else:
            kv_seq_len = seq_len
            mask = torch.triu(torch.ones(seq_len, kv_seq_len, device=self.device), diagonal=1).bool()
        scores = torch.einsum('bngsd,bnqd->bngsq', q, k) / (self.head_dim ** 0.5)
        logger.debug(f"Слой {layer_idx}: scores min: {scores.min().item()}, max: {scores.max().item()}")
        scores = torch.clamp(scores, min=-1e4, max=1e4)
        scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(1).unsqueeze(1), float('-inf'))
        attn_weights = F.softmax(scores.to(torch.float32), dim=-1).to(hidden_states.dtype)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        attn_output = torch.einsum('bngsq,bnqd->bngsd', attn_weights, v)
        attn_output = attn_output.permute(0, 3, 1, 2, 4).contiguous().view(batch_size, seq_len, self.hidden_size)
        attn_output = F.linear(attn_output, o_block)
        del q_block, k_block, v_block, o_block, q, scores, attn_weights
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        return attn_output, (k, v)

    def _compute_gated_mlp(self, hidden_states: torch.Tensor, layer_idx: int) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        gate_key = f"{self.model_name}_layer{layer_idx}_mlp_gate_proj_weight_block0"
        up_key = f"{self.model_name}_layer{layer_idx}_mlp_up_proj_weight_block0"
        down_key = f"{self.model_name}_layer{layer_idx}_mlp_down_proj_weight_block0"

        if gate_key not in self.dispatcher.metadata or up_key not in self.dispatcher.metadata or down_key not in self.dispatcher.metadata:
            logger.warning(f"Gated MLP blocks not found for layer {layer_idx}, skipping")
            return hidden_states

        gate_block = self.dispatcher.load_block(self.dispatcher.metadata[gate_key])
        up_block = self.dispatcher.load_block(self.dispatcher.metadata[up_key])
        down_block = self.dispatcher.load_block(self.dispatcher.metadata[down_key])

        gate = F.linear(hidden_states, gate_block)
        up = F.linear(hidden_states, up_block)
        x = gate * F.silu(up)
        mlp_output = F.linear(x, down_block)

        del gate_block, up_block, down_block, gate, up, x
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        return mlp_output

    def _process_layers(self, hidden_states: torch.Tensor, past_kv: List[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        new_past_kv = [None] * self.num_layers if past_kv is None else past_kv.copy()
        batch_size, seq_len, _ = hidden_states.shape
        logger.debug(f"Обработка всех слоёв для hidden_states shape: {hidden_states.shape}")
        for layer_idx in self.attention_layers:
            logger.debug(f"Обрабатываем слой {layer_idx}")
            attn_output, kv = self._compute_attention_layer(hidden_states, layer_idx, new_past_kv[layer_idx])
            hidden_states = hidden_states + attn_output
            logger.debug(f"После внимания слой {layer_idx}: hidden_states min={hidden_states.min().item()}, max={hidden_states.max().item()}")
            mlp_output = self._compute_gated_mlp(hidden_states, layer_idx)
            hidden_states = hidden_states + mlp_output
            logger.debug(f"После Gated MLP слой {layer_idx}: hidden_states min={hidden_states.min().item()}, max={hidden_states.max().item()}")
            hidden_states = (hidden_states - hidden_states.mean(dim=-1, keepdim=True)) / (hidden_states.std(dim=-1, keepdim=True) + 1e-5)
            if torch.isnan(hidden_states).any():
                logger.error(f"NaN в hidden_states на слое {layer_idx}")
                hidden_states = torch.nan_to_num(hidden_states, nan=0.0)
            new_past_kv[layer_idx] = kv
            self.log_memory(f"После слоя {layer_idx}")
        return hidden_states, new_past_kv

    def _calculate_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = (hidden_states - hidden_states.mean(dim=-1, keepdim=True)) / (hidden_states.std(dim=-1, keepdim=True) + 1e-5)
        
        output_blocks = self.dispatcher.get_output_blocks()
        logits = torch.zeros(hidden_states.shape[0], hidden_states.shape[1], self.vocab_size, 
                            device=self.device, dtype=torch.float16)
        logger.debug(f"Загружаем {len(output_blocks)} выходных блоков")
        
        for block_key in sorted(output_blocks, key=lambda x: int(x.split("_block")[1].split(":")[0])):
            block_info = self.dispatcher.metadata[block_key]
            block = self.dispatcher.load_block(block_info)
            block_height = block.shape[0]
            start_idx = int(block_key.split("_block")[1].split(":")[0]) * 4096
            end_idx = min(start_idx + block_height, self.vocab_size)
            logits[:, :, start_idx:end_idx] = F.linear(hidden_states, block)
            del block
            gc.collect()
        
        logger.debug(f"Logits shape: {logits.shape}, min: {logits.min().item()}, max: {logits.max().item()}")
        return logits

    def generate(self, input_ids: np.ndarray, max_new_tokens: int = 5, top_k: int = None) -> Tuple[List[int], float]:
        with torch.no_grad():
            input_ids_torch = torch.from_numpy(input_ids).to(self.device)
            generated_ids = input_ids_torch.clone()
            # Добавляем <think> и обрабатываем весь входной промпт один раз
            think_tokens = self.tokenizer.encode("<think>", add_special_tokens=False)
            generated_ids = torch.cat([generated_ids, torch.tensor([think_tokens], device=self.device)], dim=1)
            hidden_states = self._embed_input(generated_ids)
            hidden_states, past_kv = self._process_layers(hidden_states)
            confidence = 0.0
            logger.debug(f"Исходные hidden_states shape: {hidden_states.shape}")
            logger.debug(f"Исходные past_kv shape: {[kv[0].shape for kv in past_kv]}")
            # Генерация новых токенов с использованием past_kv
            for step in range(max_new_tokens):
                logits = self._calculate_logits(hidden_states[:, -1:, :])  # Логиты только для последнего токена
                next_token_logits = logits[:, -1, :] / 0.6
                if top_k is not None:
                    top_k_values, top_k_indices = torch.topk(next_token_logits, top_k, dim=-1)
                    next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                    next_token_logits.scatter_(-1, top_k_indices, top_k_values)
                probs = F.softmax(next_token_logits, dim=-1)
                probs = torch.nan_to_num(probs, nan=0.0)
                if probs.sum() <= 0:
                    logger.warning(f"Step {step}: Все вероятности нулевые, задаём равномерное распределение")
                    probs = torch.ones_like(probs) / self.vocab_size
                logger.debug(f"Step {step}: Первые 10 вероятностей: {probs[0, :10]}")
                logger.debug(f"Step {step}: Top 10 probs: {torch.topk(probs[0], 10).values}")
                active_tokens = self.token_tracker.update(probs[0], step)
                mask = torch.ones_like(probs, dtype=torch.bool)
                if active_tokens:
                    mask[0, active_tokens] = False
                probs[mask] *= 0.5
                probs = probs / probs.sum()
                next_token = torch.multinomial(probs, num_samples=1).view(1, 1)
                confidence += probs[0, next_token].item()
                generated_ids = torch.cat([generated_ids, next_token], dim=1)
                new_hidden = self._embed_input(next_token)  # Эмбеддинг только для нового токена
                new_hidden, past_kv = self._process_layers(new_hidden, past_kv)  # Прогоняем только новый токен с past_kv
                hidden_states = torch.cat([hidden_states, new_hidden], dim=1)  # Добавляем к общему состоянию
                logger.debug(f"Step {step}: Сгенерирован токен {next_token.item()}, Prob: {probs[0, next_token].item():.6f}")
                logger.debug(f"Step {step}: Обновленные hidden_states shape: {hidden_states.shape}")
                logger.debug(f"Step {step}: Обновленные past_kv shape: {[kv[0].shape for kv in past_kv]}")
                if next_token.item() == self.tokenizer.eos_token_id:
                    logger.info(f"Генерация завершена на шаге {step}: достигнут EOS токен")
                    break
                self.log_memory(f"После шага {step} генерации")
            self.clear_memory()
            return generated_ids[0].cpu().tolist(), confidence

    def clear_memory(self):
        self.model_manager.virtual_space.matrix_models.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        self.log_memory("После очистки памяти")

def chat():
    handler = ModelHandler(
        model_name="DeepSeek-R1-Distill-Qwen-1.5B",
        tensor_dir="/workspaces/Veector/data/blocks/DeepSeek-R1-Distill-Qwen-1.5B",
        vocab_size=151936,
        hidden_size=1536,
        num_layers=28
    )
    print("\n🤖 Чат с блочной моделью активен! Введи 'выход' для завершения.")
    while True:
        user_input = input("Ты: ")
        if user_input.lower() == "выход":
            print("🤖 Чат завершён.")
            break

        print("🤖 Генерирую ответ...")
        input_ids = handler.preprocess_text(user_input)
        try:
            generated_ids, confidence = handler.generate(input_ids)
            output_text = handler.tokenizer.decode(generated_ids, skip_special_tokens=True)
            print(f"🤖: {output_text} (Confidence: {confidence:.6f})")
        except Exception as e:
            print(f"⚠️ Ошибка генерации: {e}")
            logger.exception("Подробности ошибки:")

if __name__ == "__main__":
    chat()