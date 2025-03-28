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

    def update(self, token_scores: torch.Tensor, layer_idx: int) -> List[int]:
        token_scores = token_scores.cpu().detach()
        if layer_idx == 0:
            self.token_scores = {}
        
        for token_id in range(min(self.vocab_size, token_scores.shape[-1])):
            score = token_scores[token_id].item() if token_id < token_scores.shape[-1] else 0.0
            if score > 0:
                self.token_scores[token_id] = self.token_scores.get(token_id, 0.0) * self.decay_rate + score
        
        sorted_tokens = sorted(self.token_scores.items(), key=lambda x: x[1], reverse=True)
        active_tokens = [token_id for token_id, _ in sorted_tokens[:self.max_active_tokens]]
        logger.debug(f"Layer {layer_idx}: Active tokens: {active_tokens}")
        return active_tokens

    def get_active_tokens(self) -> List[int]:
        sorted_tokens = sorted(self.token_scores.items(), key=lambda x: x[1], reverse=True)
        return [token_id for token_id, _ in sorted_tokens[:self.max_active_tokens]]

class ModelHandler:
    def __init__(self, model_name: str, tensor_dir: str, vocab_size: int, hidden_size: int, num_layers: int):
        self.model_name = model_name
        self.tensor_dir = tensor_dir
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.attention_layers = list(range(num_layers))  # [0, 1, 2, ..., 27]
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

        k_key = f"{self.model_name}_layer0_self_attn_k_proj_weight_block0"
        if k_key in self.dispatcher.metadata:
            k_block = self.dispatcher.load_block(self.dispatcher.metadata[k_key])
            self.key_dim = k_block.shape[0]
            logger.info(f"Установлен key_dim={self.key_dim} на основе блока {k_key}: {k_block.shape}")
        else:
            self.key_dim = hidden_size // self.num_key_value_heads
            logger.warning(f"Блок {k_key} не найден, используется key_dim={self.key_dim} по умолчанию")

        self.head_dim = self.hidden_size // self.num_attention_heads
        self.key_head_dim = self.key_dim // self.num_key_value_heads
        self.heads_per_group = self.num_attention_heads // self.num_key_value_heads

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
        inputs = self.tokenizer(formatted_text, return_tensors="np", max_length=max_length, truncation=True, padding="max_length")
        input_ids = inputs["input_ids"]
        # Проверка входных токенов
        logger.debug(f"Preprocessed input_ids: {input_ids.tolist()}")
        logger.debug(f"Input shape: {input_ids.shape}, Number of tokens: {input_ids.shape[1]}")
        return input_ids.astype(np.int32)

    def _embed_input(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        hidden_states = torch.zeros(batch_size, seq_len, self.hidden_size, device=self.device, dtype=torch.float16)
        embed_blocks = self.dispatcher.get_embedding_blocks(input_ids)
        logger.debug(f"Загружаем {len(embed_blocks)} блоков эмбеддингов")
        
        for block_key in embed_blocks:
            block_info = self.dispatcher.metadata[block_key]
            block = self.dispatcher.load_block(block_info)
            block_height = block.shape[0]
            start_idx = int(block_key.split("_block")[1]) * block_height
            end_idx = start_idx + block_height
            
            mask = (input_ids >= start_idx) & (input_ids < end_idx)
            if mask.any():
                indices = input_ids[mask] - start_idx
                hidden_states[mask] = block[indices]
            del block
            gc.collect()
        
        self.log_memory(f"После загрузки эмбеддингов для {input_ids.shape}")
        return hidden_states

    def _apply_attention(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logger.debug(f"Запуск внимания для слоёв: {self.attention_layers}")
        for layer_idx in self.attention_layers:
            if layer_idx >= self.num_layers or layer_idx < 0:
                raise ValueError(f"Слой {layer_idx} вне диапазона [0, {self.num_layers-1}]")
            
            logger.debug(f"Обрабатываем слой {layer_idx}")
            q_key = f"{self.model_name}_layer{layer_idx}_self_attn_q_proj_weight_block0"
            k_key_prefix = f"{self.model_name}_layer{layer_idx}_self_attn_k_proj_weight_block"
            v_key_prefix = f"{self.model_name}_layer{layer_idx}_self_attn_v_proj_weight_block"
            o_key = f"{self.model_name}_layer{layer_idx}_self_attn_o_proj_weight_block0"

            q_block = self.dispatcher.load_block(self.dispatcher.metadata.get(q_key, q_key))
            o_block = self.dispatcher.load_block(self.dispatcher.metadata.get(o_key, o_key))

            k_blocks = [k_key_prefix + str(i) for i in range(3)]
            v_blocks = [v_key_prefix + str(i) for i in range(3)]
            k_block_keys = [k for k in k_blocks if k in self.dispatcher.metadata]
            v_block_keys = [v for v in v_blocks if v in self.dispatcher.metadata]
            
            if not k_block_keys or not v_block_keys:
                raise ValueError(f"Не найдены блоки для k или v в слое {layer_idx}")

            k_block_list = [self.dispatcher.load_block(self.dispatcher.metadata[k]) for k in k_block_keys]
            v_block_list = [self.dispatcher.load_block(self.dispatcher.metadata[v]) for v in v_block_keys]
            
            if len(k_block_list) == 0 or len(v_block_list) == 0:
                raise ValueError(f"Не удалось загрузить блоки для слоя {layer_idx}: k_blocks={len(k_block_list)}, v_blocks={len(v_block_list)}")

            if len(k_block_list) == 1:
                k_full = k_block_list[0]
                v_full = v_block_list[0]
                logger.debug(f"Слой {layer_idx}: Используется один блок для k/v: {k_full.shape}")
            else:
                try:
                    k_full = self.dispatcher.assemble_tensor(k_block_keys, (self.key_dim, self.hidden_size))
                    v_full = self.dispatcher.assemble_tensor(v_block_keys, (self.key_dim, self.hidden_size))
                except ValueError as e:
                    logger.warning(f"Слой {layer_idx}: Ошибка сборки: {e}. Используем первый блок.")
                    k_full = k_block_list[0]
                    v_full = v_block_list[0]
                    self.key_dim = k_full.shape[0]

            self.key_head_dim = self.key_dim // self.num_key_value_heads
            self.heads_per_group = self.num_attention_heads // self.num_key_value_heads

            logger.debug(f"Слой {layer_idx}: q_block shape: {q_block.shape}, k_full shape: {k_full.shape}, "
                         f"v_full shape: {v_full.shape}, o_block shape: {o_block.shape}")

            batch_size, seq_len, _ = hidden_states.shape

            q = F.linear(hidden_states, q_block)
            k = F.linear(hidden_states, k_full)
            v = F.linear(hidden_states, v_full)

            logger.debug(f"Слой {layer_idx}: q shape: {q.shape}, k shape: {k.shape}, v shape: {v.shape}")

            q = q.view(batch_size, seq_len, self.num_key_value_heads, self.heads_per_group, self.head_dim)
            q = q.permute(0, 2, 3, 1, 4)
            k = k.view(batch_size, seq_len, self.num_key_value_heads, self.key_head_dim)
            k = k.permute(0, 2, 1, 3)
            v = v.view(batch_size, seq_len, self.num_key_value_heads, self.key_head_dim)
            v = v.permute(0, 2, 1, 3)

            scores = torch.einsum('bhgsd,bhqd->bhgsq', q, k) / (self.head_dim ** 0.5)
            mask = torch.triu(torch.ones(seq_len, seq_len, device=self.device), diagonal=1).bool()
            scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(1).unsqueeze(1), float('-inf'))
            attn_weights = F.softmax(scores, dim=-1)

            attn_output = torch.einsum('bhgsq,bhqd->bhgsd', attn_weights, v)
            attn_output = attn_output.permute(0, 3, 1, 2, 4).contiguous()
            attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)
            hidden_states = F.linear(attn_output, o_block)

            del q_block, k_full, v_full, o_block, q, k, v, scores, attn_weights, attn_output
            gc.collect()
            self.log_memory(f"После внимания для слоя {layer_idx}")

        logger.debug(f"Завершено применение внимания для слоёв: {self.attention_layers}")
        # Проверка скрытого состояния перед возвратом
        logger.debug(f"Hidden state shape: {hidden_states.shape}, min: {hidden_states.min().item()}, max: {hidden_states.max().item()}")
        return hidden_states

    def _calculate_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        output_blocks = self.dispatcher.get_output_blocks(top_k=38)
        logits = torch.zeros(hidden_states.shape[0], hidden_states.shape[1], self.vocab_size, 
                            device=self.device, dtype=torch.float16)
        logger.debug(f"Загружаем {len(output_blocks)} выходных блоков")
        
        for block_key in sorted(output_blocks, key=lambda x: int(x.split("_block")[1])):
            block_info = self.dispatcher.metadata[block_key]
            block = self.dispatcher.load_block(block_info)
            block_height = block.shape[0]
            start_idx = int(block_key.split("_block")[1]) * block_height
            end_idx = min(start_idx + block_height, self.vocab_size)
            logger.debug(f"Загружаем блок {block_key}: start_idx={start_idx}, end_idx={end_idx}, block_shape={block.shape}")
            # Проверка весов блока
            logger.debug(f"Block {block_key} min: {block.min().item()}, max: {block.max().item()}")
            logits[:, :, start_idx:end_idx] = F.linear(hidden_states, block)
            del block
            gc.collect()
        
        logger.debug(f"Logits shape: {logits.shape}, min: {logits.min().item()}, max: {logits.max().item()}")
        self.log_memory("После вычисления логитов")
        return logits

    def generate(self, input_ids: np.ndarray) -> Tuple[List[int], float]:
        with torch.no_grad():
            input_ids_torch = torch.from_numpy(input_ids).to(self.device)
            generated_ids = input_ids_torch.clone()
            hidden_states = self._embed_input(input_ids_torch)
            hidden_states = self._apply_attention(hidden_states)
            confidence = 0.0

            think_tokens = self.tokenizer.encode("<think>\n", add_special_tokens=False)
            for token in think_tokens:
                generated_ids = torch.cat([generated_ids, torch.tensor([[token]], device=self.device)], dim=1)
            hidden_states = self._embed_input(generated_ids)
            hidden_states = self._apply_attention(hidden_states)

            for step in range(5):
                logits = self._calculate_logits(hidden_states)
                next_token_logits = logits[:, -1, :]

                logits_scaled = next_token_logits - next_token_logits.max(dim=-1, keepdim=True)[0]
                logits_scaled = logits_scaled / 0.6
                probs = F.softmax(logits_scaled, dim=-1)
                logger.debug(f"Step {step}: Первые 10 вероятностей: {probs[0, :10]}")
                logger.debug(f"Step {step}: Top 10 probs: {torch.topk(probs[0], 10).values}")

                active_tokens = self.token_tracker.update(probs[0], step)
                for token_id in range(self.vocab_size):
                    if token_id not in active_tokens:
                        probs[0, token_id] *= 0.5

                next_token = torch.multinomial(probs[0], num_samples=1)
                confidence += probs[0, next_token].item()

                generated_ids = torch.cat([generated_ids, next_token.unsqueeze(0)], dim=1)
                logger.debug(f"Step {step}: Сгенерирован токен {next_token.item()}, Prob: {probs[0, next_token].item():.6f}, Confidence {confidence:.6f}")
                del hidden_states
                hidden_states = self._embed_input(generated_ids)
                hidden_states = self._apply_attention(hidden_states)

                if next_token.item() == self.tokenizer.eos_token_id:
                    logger.info(f"Генерация завершена на шаге {step}: достигнут EOS токен")
                    break

                self.log_memory(f"После шага {step} генерации")

            del hidden_states
            self.clear_memory()
            return generated_ids[0].cpu().tolist(), confidence

    def clear_memory(self):
        self.model_manager.virtual_space.matrix_models.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        self.log_memory("После очистки памяти")

if __name__ == "__main__":
    handler = ModelHandler(
        model_name="DeepSeek-R1-Distill-Qwen-1.5B",
        tensor_dir="/workspaces/Veector/data/blocks/DeepSeek-R1-Distill-Qwen-1.5B",
        vocab_size=151936,
        hidden_size=1536,
        num_layers=28
    )

    input_text = "Hallo!"  # Изменил на "Hallo!" как в твоих логах
    input_ids = handler.preprocess_text(input_text)
    generated_ids, confidence = handler.generate(input_ids)
    output_text = handler.tokenizer.decode(generated_ids, skip_special_tokens=True)
    print(f"Generated: {output_text}, Confidence: {confidence:.6f}")