import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import json
import os
import gc
import logging
from observatory import Observer  # Обновлённый импорт

# Настройка логирования
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class VirtualMatrix:
    def __init__(self, dispatcher):
        self.dispatcher = dispatcher
        self.device = dispatcher.device
        self.metadata = dispatcher.metadata
        self.cache = {}

    def get_block(self, block_key):
        if block_key not in self.cache:
            self.cache[block_key] = self.dispatcher.load_block(block_key)  # Передаём строку block_key
        return self.cache[block_key]

    def embedding(self, input_ids, prefix):
        batch_size, seq_len = input_ids.shape
        hidden_size = self.dispatcher.hidden_size
        output = torch.zeros(batch_size, seq_len, hidden_size, dtype=torch.float16, device=self.device)

        unique_tokens = torch.unique(input_ids)
        block_height = self.metadata[f"{prefix}_block0"]["shape"][0]

        for token in unique_tokens:
            block_idx = token.item() // block_height
            block_key = f"{prefix}_block{block_idx}"
            if block_key not in self.metadata:
                continue
            
            block = self.get_block(block_key)
            local_idx = token.item() % block_height
            token_embedding = block[local_idx]
            mask = (input_ids == token)
            output[mask] = token_embedding.to(self.device)
        
        self.clear_cache()
        return output

    def linear(self, input, prefix, output_size, input_size, top_k=None):
        batch_size, seq_len, in_features = input.shape
        assert in_features == input_size, f"Input size mismatch: {in_features} != {input_size}"
        
        block_keys = [k for k in self.metadata.keys() if k.startswith(prefix)]
        block_keys = sorted(block_keys, key=lambda x: int(x.split("_block")[1]))

        if prefix.endswith("_output") and top_k is not None:
            # Шаг 1: Грубая оценка на основе первых нескольких блоков
            coarse_logits = torch.zeros(batch_size, seq_len, 4096 * 2, dtype=torch.float16, device=self.device)  # Первые 2 блока (8192 токена)
            for block_key in block_keys[:2]:  # Используем только block0 и block1 для оценки
                block = self.get_block(block_key)
                block_out_size, block_in_size = block.shape
                block_idx = int(block_key.split("_block")[1])
                start_row = block_idx * block_out_size
                end_row = start_row + block_out_size
                coarse_logits[..., start_row:end_row] = torch.matmul(input, block.t())
                del block
                self.clear_cache()

            # Выбираем top_k кандидатов
            coarse_values, coarse_indices = torch.topk(coarse_logits, k=top_k, dim=-1)

            # Шаг 2: Уточняем логиты только для выбранных токенов
            output = torch.zeros(batch_size, seq_len, top_k, dtype=torch.float16, device=self.device)
            vocab_per_block = 4096  # Предполагаем, что большинство блоков по 4096 строк

            for b in range(batch_size):
                for s in range(seq_len):
                    token_ids = coarse_indices[b, s].cpu().numpy()  # Индексы кандидатов
                    for token_id in token_ids:
                        block_idx = token_id // vocab_per_block
                        block_key = f"{prefix}_block{block_idx}"
                        if block_key not in self.metadata:
                            continue
                        block = self.get_block(block_key)
                        local_idx = token_id % vocab_per_block
                        output[b, s, token_ids.tolist().index(token_id)] = torch.matmul(input[b, s:s+1], block[local_idx:local_idx+1].t())
                        del block
                        self.clear_cache()

            return output, coarse_indices  # Возвращаем логиты и индексы токенов
        else:
            output = torch.zeros(batch_size, seq_len, output_size, dtype=torch.float16, device=self.device)
            for block_key in block_keys:
                block = self.get_block(block_key)
                block_out_size, block_in_size = block.shape
                block_idx = int(block_key.split("_block")[1])

                if block_out_size == output_size:  # Сборка по столбцам
                    start_col = block_idx * block_in_size
                    end_col = min(start_col + block_in_size, input_size)
                    input_slice = input[..., start_col:end_col]
                    output += torch.matmul(input_slice, block.t())
                else:  # Сборка по строкам
                    start_row = block_idx * block_out_size
                    end_row = start_row + block_out_size
                    output[..., start_row:end_row] = torch.matmul(input, block.t())
            
            self.clear_cache()
            return output

    def clear_cache(self):
        self.cache.clear()
        gc.collect()

class ModelDispatcher:
    def __init__(self, model_name, metadata_path, vocab_size, hidden_size, num_layers, num_attention_heads, intermediate_size, key_dim, num_key_value_heads):
        self.model_name = model_name
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.key_dim = key_dim
        self.num_key_value_heads = num_key_value_heads
        with open(metadata_path, "r") as f:
            self.metadata = json.load(f)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cache = {}
        self.base_dir = f"/workspaces/Veector/data/blocks/{model_name}"
        self.block_cache = {}
        self.virtual_matrix = None  # Будет установлено позже
        self.tokenizer = None  # Будет установлено позже

    def get_embedding_blocks(self, input_ids):
        unique_tokens = torch.unique(input_ids)
        needed_blocks = set()
        embed_blocks = {k: v for k, v in self.metadata.items() if k.startswith(f"{self.model_name}_embed")}
        if not embed_blocks:
            raise ValueError("Нет блоков эмбеддингов в метаданных!")
        sample_block = list(embed_blocks.values())[0]
        block_height = sample_block["shape"][0]

        for token in unique_tokens:
            row_block = token.item() // block_height
            block_key = f"{self.model_name}_embed_block{row_block}"
            if block_key in self.metadata:
                needed_blocks.add(block_key)
        return needed_blocks

    def get_layer_blocks(self, layer_idx, component):
        blocks = set()
        for block_key, info in self.metadata.items():
            if f"layer{layer_idx}_{component}" in block_key:  # Исправлено на проверку по block_key
                blocks.add(block_key)
        return blocks

    def get_output_blocks(self, top_k=None):
        output_blocks = {k: v for k, v in self.metadata.items() if k.startswith(f"{self.model_name}_output")}
        if not output_blocks:
            raise ValueError("Нет блоков выходного слоя в метаданных!")
        
        total_blocks = len(output_blocks)
        sample_block = list(output_blocks.values())[0]
        block_height = sample_block["shape"][0]

        if top_k:
            num_blocks_needed = min((top_k + block_height - 1) // block_height, total_blocks)
        else:
            num_blocks_needed = total_blocks

        needed_blocks = {f"{self.model_name}_output_block{i}" for i in range(num_blocks_needed) if f"{self.model_name}_output_block{i}" in output_blocks}
        return needed_blocks

    def load_block(self, block_name):
        if block_name in self.block_cache:
            return self.block_cache[block_name]
        
        if block_name not in self.metadata:
            raise ValueError(f"Блок {block_name} не найден в метаданных")
        
        block_info = self.metadata[block_name]
        block_hash = block_info.get("hash")
        
        if block_hash and hasattr(self, 'ipfs') and self.ipfs:
            self.ipfs.get(block_hash)
            block = torch.load(f"{block_hash}.pt", map_location=self.device, weights_only=True)
        else:
            original_path = block_info["path"]
            block_filename = Path(original_path).name
            corrected_path = os.path.join(self.base_dir, block_filename)
            if not os.path.exists(corrected_path):
                raise FileNotFoundError(f"Файл блока {corrected_path} не найден")
            block = torch.load(corrected_path, map_location=self.device, weights_only=True)
        
        self.block_cache[block_name] = block
        logger.info(f"Loaded block {corrected_path} with shape {block.shape}")
        return block

    def assemble_tensor(self, block_keys, target_shape):
        if not block_keys:
            raise ValueError("Нет блоков для сборки тензора")

        sorted_keys = sorted(block_keys, key=lambda x: int(x.split("_block")[1]))
        blocks_info = [self.metadata[key] for key in sorted_keys]

        total_height = sum(info["shape"][0] for info in blocks_info)
        total_width = sum(info["shape"][1] for info in blocks_info)
        first_height = blocks_info[0]["shape"][0]
        first_width = blocks_info[0]["shape"][1]

        if total_height == target_shape[0] and all(info["shape"][1] == first_width for info in blocks_info):
            dim = 0
            if target_shape[1] != first_width:
                raise ValueError(f"Ширина блоков {first_width} не совпадает с целевой шириной {target_shape[1]}")
        elif total_width == target_shape[1] and all(info["shape"][0] == first_height for info in blocks_info):
            dim = 1
            if target_shape[0] != first_height:
                raise ValueError(f"Высота блоков {first_height} не совпадает с целевой высотой {target_shape[0]}")
        else:
            raise ValueError(f"Невозможно собрать тензор с целевой формой {target_shape} из блоков: "
                             f"суммарная высота={total_height}, ширина={total_width}")

        tensor = torch.zeros(target_shape, dtype=torch.float16, device=self.device)

        if dim == 0:
            current_row = 0
            for block_key in sorted_keys:
                block = self.load_block(block_key)
                block_height = block.shape[0]
                tensor[current_row:current_row + block_height, :] = block
                current_row += block_height
                del block
                gc.collect()
        else:
            current_col = 0
            for block_key in sorted_keys:
                block = self.load_block(block_key)
                block_width = block.shape[1]
                tensor[:, current_col:current_col + block_width] = block
                current_col += block_width
                del block
                gc.collect()

        logger.info(f"Assembled tensor with shape {tensor.shape} (dim={dim})")
        return tensor

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
        
        if torch.any(input_ids >= self.vocab_size):
            raise ValueError(f"Входные данные содержат значения, превышающие vocab_size ({self.vocab_size})")
        
        batch_size, seq_len = input_ids.shape
        hidden_states = self.virtual_matrix.embedding(input_ids, f"{self.dispatcher.model_name}_embed")
        logger.info(f"Embeddings shape: {hidden_states.shape}")

        for layer_idx in range(self.num_layers):
            logger.info(f"Processing layer {layer_idx}")

            q = self.virtual_matrix.linear(hidden_states, f"{self.dispatcher.model_name}_layer{layer_idx}_self_attn_q_proj_weight", self.hidden_size, self.hidden_size)
            k = self.virtual_matrix.linear(hidden_states, f"{self.dispatcher.model_name}_layer{layer_idx}_self_attn_k_proj_weight", self.key_dim, self.hidden_size)
            v = self.virtual_matrix.linear(hidden_states, f"{self.dispatcher.model_name}_layer{layer_idx}_self_attn_v_proj_weight", self.key_dim, self.hidden_size)

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
            hidden_states = self.virtual_matrix.linear(attn_output, f"{self.dispatcher.model_name}_layer{layer_idx}_self_attn_o_proj_weight", self.hidden_size, self.hidden_size)

            gate = self.virtual_matrix.linear(hidden_states, f"{self.dispatcher.model_name}_layer{layer_idx}_mlp_gate_proj_weight", self.intermediate_size, self.hidden_size)
            up = self.virtual_matrix.linear(hidden_states, f"{self.dispatcher.model_name}_layer{layer_idx}_mlp_up_proj_weight", self.intermediate_size, self.hidden_size)
            mlp_output = gate * up
            hidden_states = self.virtual_matrix.linear(mlp_output, f"{self.dispatcher.model_name}_layer{layer_idx}_mlp_down_proj_weight", self.hidden_size, self.intermediate_size)

        gc.collect()
        output = self.virtual_matrix.linear(hidden_states, f"{self.dispatcher.model_name}_output", self.vocab_size, self.hidden_size, top_k=top_k)
        
        if top_k is not None and top_k < self.vocab_size:
            logits, indices = output
            logger.info(f"Final logits shape: {logits.shape}, indices shape: {indices.shape}")
        else:
            logits = output
            indices = None
            logger.info(f"Final logits shape: {logits.shape}")
        
        return logits, indices

class VirtualSpace:
    def __init__(self, veector, use_ipfs: bool = False, model_manager=None, metadata_dir: str = "/workspaces/Veector/data"):
        """
        Инициализация VirtualSpace для виртуализации модели.
        
        :param veector: Экземпляр Veector.
        :param use_ipfs: Включение IPFS.
        :param model_manager: Экземпляр ModelManager для управления моделью.
        :param metadata_dir: Путь к директории с метаданными.
        """
        self.veector = veector
        self.use_ipfs = use_ipfs
        self.model_manager = model_manager  # Связь с ModelManager
        self.matrix_models = {}  # Хранилище виртуализированных моделей
        self.current_model = None
        self.metadata_dir = Path(metadata_dir)
        self.virtual_matrix = None
        self.dispatcher = None
        self.tokenizer = None  # Токенизатор устанавливается позже

    def switch_model(self, model_name: str, vocab_size: int, hidden_size: int, num_layers: int, num_attention_heads: int, intermediate_size: int, key_dim: int, num_key_value_heads: int = 2):
        """
        Переключение на модель с виртуализацией блоков.
        
        :param model_name: Название модели.
        :param vocab_size: Размер словаря.
        :param hidden_size: Размер скрытого слоя.
        :param num_layers: Количество слоёв.
        :param num_attention_heads: Количество голов внимания.
        :param intermediate_size: Размер промежуточного слоя.
        :param key_dim: Размер ключей.
        :param num_key_value_heads: Количество голов для ключей и значений.
        """
        metadata_path = self.metadata_dir / "blocks" / model_name / f"{model_name}_metadata.json"
        if not metadata_path.exists():
            raise ValueError(f"Метаданные для модели {model_name} не найдены в {metadata_path}")
        
        self.dispatcher = ModelDispatcher(model_name, metadata_path, vocab_size, hidden_size, num_layers, num_attention_heads, intermediate_size, key_dim, num_key_value_heads)
        self.virtual_matrix = VirtualMatrix(self.dispatcher)
        self.dispatcher.virtual_matrix = self.virtual_matrix
        self.dispatcher.tokenizer = self.tokenizer  # Устанавливаем токенизатор
        
        self.current_model = model_name
        self.matrix_models[model_name] = self.virtual_matrix  # Сохраняем виртуализированную матрицу
        logger.info(f"Переключено на модель: {model_name}")

    def get_matrix(self):
        """Возвращает текущую виртуализированную матрицу."""
        if not self.current_model:
            raise ValueError("Не выбрана активная модель")
        return self.virtual_matrix

    def clear_memory(self):
        """Очистка памяти."""
        if self.virtual_matrix:
            self.virtual_matrix.clear_cache()
        logger.info("Memory cleared in VirtualSpace")

if __name__ == "__main__":
    # Тестовый запуск
    from transformers import AutoTokenizer
    from core import Veector
    from model_manager import ModelManager

    veector = Veector(use_memory=False, ipfs_enabled=False)
    model_manager = ModelManager(veector, ipfs_enabled=False)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-1.5B-Instruct")
    
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
        num_key_value_heads=2
    )

    # Тест загрузки эмбеддингов
    input_ids = tokenizer.encode("Привет, как дела?", return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
    matrix = virtual_space.get_matrix()
    embeddings = matrix.embedding(input_ids, f"{virtual_space.current_model}_embed")
    print(f"Embeddings shape: {embeddings.shape}")