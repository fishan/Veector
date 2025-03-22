import logging
from pathlib import Path
import torch
import os
import gc
import numpy as np
from gguf import GGUFReader
import time

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def format_size(bytes_size):
    """Преобразует размер в байтах в читаемый формат (MiB, GiB)."""
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

    def get_block(self, block_key):
        if block_key not in self.cache:
            if block_key in self.dispatcher.tensor_map:
                start_time = time.time()
                self.cache[block_key] = self.dispatcher.load_gguf_tensor(block_key)
                logger.info(f"Кэширован блок {block_key}, форма: {self.cache[block_key].shape}, "
                           f"размер: {format_size(self.cache[block_key].nbytes)}, "
                           f"время загрузки: {(time.time() - start_time)*1000:.2f} мс")
            else:
                raise ValueError(f"Блок {block_key} не найден")
        return self.cache[block_key]

    def embedding(self, input_ids, prefix):
        batch_size, seq_len = input_ids.shape
        hidden_size = self.dispatcher.hidden_size
        logger.debug(f"Формирование эмбеддингов, input_ids: {input_ids.shape}, hidden_size: {hidden_size}")
        
        start_time = time.time()
        output = torch.zeros(batch_size, seq_len, hidden_size, dtype=torch.float16, device=self.device)
        unique_tokens = torch.unique(input_ids)
        logger.debug(f"Уникальные токены: {unique_tokens}")
        
        emb_tensor = self.get_block(f"{prefix}.weight")
        logger.debug(f"Загружен {prefix}.weight, форма: {emb_tensor.shape}")
        
        if emb_tensor.shape[0] == hidden_size and emb_tensor.shape[1] == self.dispatcher.vocab_size:
            emb_tensor = emb_tensor.t()
            logger.debug(f"Транспонирован {prefix}.weight в {emb_tensor.shape}")
        elif emb_tensor.shape[0] != self.dispatcher.vocab_size or emb_tensor.shape[1] != hidden_size:
            raise ValueError(f"Неправильная форма {prefix}.weight: {emb_tensor.shape}, ожидается [151936, 1536]")

        embed_time = time.time()
        for token in unique_tokens:
            token_id = token.item()
            if token_id < emb_tensor.shape[0]:
                token_embedding = emb_tensor[token_id]
                mask = (input_ids == token)
                output[mask] = token_embedding.to(self.device)
                logger.debug(f"Эмбеддинг для token_id {token_id}, форма: {token_embedding.shape}")
        
        total_time = (time.time() - start_time) * 1000
        embed_time = (time.time() - embed_time) * 1000
        logger.info(f"Эмбеддинги сгенерированы за {total_time:.2f} мс, "
                   f"время обработки токенов: {embed_time:.2f} мс ({len(unique_tokens)} токенов, "
                   f"{len(unique_tokens) / (embed_time / 1000):.2f} токенов/с), "
                   f"размер: {format_size(output.nbytes)}")
        self.clear_cache()
        return output

    def linear(self, input, prefix, output_size, input_size, top_k=None):
        batch_size, seq_len, in_features = input.shape
        assert in_features == input_size, f"Input size mismatch: {in_features} != {input_size}"
        logger.debug(f"Линейный слой, input: {input.shape}, output_size: {output_size}, input_size: {input_size}, top_k: {top_k}")
        
        start_time = time.time()
        output = torch.zeros(batch_size, seq_len, output_size, dtype=torch.float16, device=self.device)
        block_height = 4096
        num_blocks = (output_size + block_height - 1) // block_height
        
        if prefix == "output":
            tensor_name = "output.weight"
        elif prefix.startswith("layers."):
            parts = prefix.split('.')
            if len(parts) >= 3:
                layer_idx = parts[1]
                component = parts[2]
                if component == "attn":
                    component = "attn_q"
                elif component == "ffn":
                    component = "ffn_gate"
                tensor_name = f"blk.{layer_idx}.{component}.weight"
            else:
                raise ValueError(f"Неправильный формат prefix: {prefix}")
        else:
            raise ValueError(f"Неизвестный формат prefix: {prefix}")
        
        logger.debug(f"Запрос тензора: {tensor_name}")
        
        if prefix == "output" and top_k:
            coarse_logits = torch.zeros(batch_size, seq_len, min(8192, output_size), dtype=torch.float16, device=self.device)
            for block_idx in range(min(2, num_blocks)):
                start_row = block_idx * block_height
                end_row = min(start_row + block_height, output_size)
                block = self.dispatcher.load_gguf_tensor(tensor_name, start_row, end_row)
                coarse_logits[..., start_row:end_row] = torch.matmul(input, block.t())
            coarse_values, coarse_indices = torch.topk(coarse_logits, k=top_k, dim=-1)
            output = torch.zeros(batch_size, seq_len, top_k, dtype=torch.float16, device=self.device)
            
            for b in range(batch_size):
                for s in range(seq_len):
                    token_ids = coarse_indices[b, s].cpu().numpy()
                    for token_id in token_ids:
                        block_idx = token_id // block_height
                        start_row = block_idx * block_height
                        end_row = min(start_row + block_height, output_size)
                        block = self.dispatcher.load_gguf_tensor(tensor_name, start_row, end_row)
                        local_idx = token_id % block_height
                        if local_idx < block.shape[0]:
                            output[b, s, token_ids.tolist().index(token_id)] = torch.matmul(
                                input[b, s:s+1], block[local_idx:local_idx+1].t()
                            )
            logger.info(f"Генерация с top_k завершена за {(time.time() - start_time)*1000:.2f} мс, "
                       f"output: {output.shape}, coarse_indices: {coarse_indices.shape}, "
                       f"размер output: {format_size(output.nbytes)}")
            return output, coarse_indices
        else:
            for block_idx in range(num_blocks):
                start_row = block_idx * block_height
                end_row = min(start_row + block_height, output_size)
                block = self.dispatcher.load_gguf_tensor(tensor_name, start_row, end_row)
                output[..., start_row:end_row] = torch.matmul(input, block.t())
            logger.info(f"Линейный слой завершен за {(time.time() - start_time)*1000:.2f} мс, "
                       f"output: {output.shape}, размер: {format_size(output.nbytes)}")
            return output

    def clear_cache(self):
        total_size = sum(t.nbytes for t in self.cache.values()) if self.cache else 0
        self.cache.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info(f"Кэш очищен, освобождено: {format_size(total_size)}")

class ModelDispatcher:
    def __init__(self, model_name, split_prefix=None, split_count=None, vocab_size=151936, hidden_size=1536, 
                 num_layers=28, num_attention_heads=12, intermediate_size=8960, key_dim=256, num_key_value_heads=2, 
                 base_dir=None):
        self.model_name = model_name
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.key_dim = key_dim
        self.num_key_value_heads = num_key_value_heads
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.base_dir = base_dir or f"/workspaces/Veector/data/blocks/{model_name}"
        self.tensor_map = {}
        self.reader_cache = {}
        
        if split_prefix and split_count:
            full_split_prefix = os.path.join(self.base_dir, split_prefix)
            self.split_prefix = full_split_prefix
            self.split_count = split_count
            self._build_tensor_map()
            logger.info(f"Создана карта тензоров для {split_count} GGUF-сплитов модели {model_name} из {full_split_prefix}, "
                       f"всего тензоров: {len(self.tensor_map)}")
        else:
            raise ValueError("Необходимо указать split_prefix и split_count для GGUF-режима")

    def _build_tensor_map(self):
        start_time = time.time()
        total_size = 0
        for i in range(1, self.split_count + 1):
            file_path = f"{self.split_prefix}-{i:05d}-of-{self.split_count:05d}.gguf"
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Файл {file_path} не найден")
            reader = GGUFReader(file_path)
            file_size = os.path.getsize(file_path)
            total_size += file_size
            for tensor in reader.tensors:
                self.tensor_map[tensor.name] = (file_path, tensor)
            del reader
        gc.collect()
        logger.info(f"Карта тензоров построена за {(time.time() - start_time)*1000:.2f} мс, "
                   f"общий размер сплитов: {format_size(total_size)}")

    def load_gguf_tensor(self, tensor_name, start_row=None, end_row=None):
        if tensor_name not in self.tensor_map:
            raise ValueError(f"Тензор {tensor_name} не найден в GGUF-сплитах")
        
        file_path, tensor = self.tensor_map[tensor_name]
        if file_path not in self.reader_cache:
            self.reader_cache[file_path] = GGUFReader(file_path)
        reader = self.reader_cache[file_path]
        
        start_time = time.time()
        logger.debug(f"Загрузка тензора {tensor_name} из {file_path}, тип: {tensor.tensor_type}, форма: {tensor.shape}")
        raw_block = tensor.data
        
        logger.debug(f"Размер raw_block: {len(raw_block)} байт, содержимое (первые 10): {raw_block[:10]}")
        
        if start_row is not None and end_row is not None:
            raw_block = raw_block[start_row:end_row, :]
            logger.debug(f"Вырезан блок: {start_row}-{end_row}, новый размер: {len(raw_block)} байт")
        
        if tensor.tensor_type == 12:  # GGUF_TYPE_Q4_K
            expected_shape = (self.vocab_size, self.hidden_size) if tensor_name == "token_embd.weight" else tensor.shape
            group_size = 32
            n_elements = expected_shape[0] * expected_shape[1]
            n_groups = n_elements // group_size
            expected_bytes = (n_elements // 2) + (n_groups * 4)
            
            logger.debug(f"Ожидаемый размер: {expected_bytes} байт, n_elements: {n_elements}, n_groups: {n_groups}")
            
            if len(raw_block) != expected_bytes:
                raise ValueError(f"Неверный размер данных для Q4_K: {len(raw_block)} байт, ожидается {expected_bytes}")
            
            # Деквантизация (оставим как есть для теста)
            q_data = np.frombuffer(raw_block[:n_elements // 2], dtype=np.uint8)
            meta_data = np.frombuffer(raw_block[n_elements // 2:], dtype=np.float16)
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
            
            result = result.reshape(expected_shape)
            result = torch.from_numpy(result).to(torch.float16)
        else:
            result = torch.from_numpy(raw_block.copy()).to(torch.float16)
        
        if tensor_name == "token_embd.weight" and result.shape[0] == self.hidden_size:
            result = result.t()
        
        load_time = (time.time() - start_time) * 1000
        logger.info(f"Деквантизован тензор {tensor_name}, форма: {result.shape}, "
                    f"размер: {format_size(result.nbytes)}, время: {load_time:.2f} мс")
        return result

class VirtualSpace:
    def __init__(self, veector, use_ipfs: bool = False, model_manager=None, metadata_dir: str = "/workspaces/Veector/data"):
        self.veector = veector
        self.use_ipfs = use_ipfs
        self.model_manager = model_manager
        self.metadata_dir = Path(metadata_dir)
        self.dispatcher = None
        self.virtual_matrix = None
        self.tokenizer = None

    def switch_model(self, model_name: str, vocab_size: int, hidden_size: int, num_layers: int, 
                     num_attention_heads: int, intermediate_size: int, key_dim: int, num_key_value_heads: int = 2,
                     split_prefix: str = None, split_count: int = None):
        start_time = time.time()
        self.dispatcher = ModelDispatcher(
            model_name, split_prefix=split_prefix, split_count=split_count,
            vocab_size=vocab_size, hidden_size=hidden_size, num_layers=num_layers,
            num_attention_heads=num_attention_heads, intermediate_size=intermediate_size,
            key_dim=key_dim, num_key_value_heads=num_key_value_heads,
            base_dir=self.metadata_dir / "blocks" / model_name
        )
        self.virtual_matrix = VirtualMatrix(self.dispatcher)
        self.dispatcher.virtual_matrix = self.virtual_matrix
        self.dispatcher.tokenizer = self.tokenizer
        logger.info(f"Переключено на модель {model_name} за {(time.time() - start_time)*1000:.2f} мс")

    def clear_memory(self):
        if self.virtual_matrix:
            self.virtual_matrix.clear_cache()
        if self.dispatcher and hasattr(self.dispatcher, 'reader_cache'):
            total_size = sum(os.path.getsize(f) for f in self.dispatcher.reader_cache.keys())
            self.dispatcher.reader_cache.clear()
            logger.info(f"Reader cache очищен, освобождено: {format_size(total_size)}")
        gc.collect()
        logger.info("Память очищена в VirtualSpace")

if __name__ == "__main__":
    from transformers import AutoTokenizer
    from core import Veector
    from model_manager import ModelManager

    veector = Veector(use_memory=False, ipfs_enabled=False)
    model_manager = ModelManager(veector, ipfs_enabled=False)
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    
    virtual_space = VirtualSpace(veector, use_ipfs=False, model_manager=model_manager)
    virtual_space.tokenizer = tokenizer
    
    virtual_space.switch_model(
        model_name="DeepSeek-R1-Distill-Qwen-1.5B",
        vocab_size=151936,
        hidden_size=1536,
        num_layers=28,
        num_attention_heads=12,
        intermediate_size=8960,
        key_dim=256,
        num_key_value_heads=2,
        split_prefix="DeepSeek-R1-Distill-Qwen-1.5B-split",
        split_count=43
    )

    # Тестовый вызов
    input_ids = tokenizer.encode("Hello!", return_tensors="pt").to(virtual_space.dispatcher.device)
    embeddings = virtual_space.virtual_matrix.embedding(input_ids, "token_embd")
    logger.info(f"Тестовые эмбеддинги: {embeddings.shape}")

