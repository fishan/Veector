import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import json
import os
import gc
import logging

logger = logging.getLogger(__name__)

class VirtualMatrix:
    def __init__(self, ipfs_client=None):
        self.ipfs = ipfs_client
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.matrices = {}

    def load_block(self, block_info):
        block_hash = block_info.get("hash")
        if self.ipfs and block_hash:
            self.ipfs.get(block_hash)
            return torch.load(f"{block_hash}.pt", map_location=self.device, weights_only=True)
        return torch.load(block_info["path"], map_location=self.device, weights_only=True)

class ModelDispatcher:
    def __init__(self, model_name, metadata_path, vocab_size, hidden_size, num_layers):
        self.model_name = model_name
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        with open(metadata_path, "r") as f:
            self.metadata = json.load(f)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.cache = {}
        self.base_dir = f"/workspaces/Veector/data/blocks/{model_name}"

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

    def get_layer_blocks(self, layer_idx):
        blocks = set()
        for block_key, info in self.metadata.items():
            if f"layer{layer_idx}_" in info["prefix"]:
                blocks.add(block_key)
        return blocks

    def get_output_blocks(self, top_k=None):
        output_blocks = {k: v for k, v in self.metadata.items() if k.startswith(f"{self.model_name}_output")}
        if not output_blocks:
            raise ValueError("Нет блоков выходного слоя в метаданных!")
        
        total_blocks = len(output_blocks)
        sample_block = list(output_blocks.values())[0]
        block_width = sample_block["shape"][0]  # Обычно это высота блока, например, 4096

        if top_k:
            num_blocks_needed = min((top_k + block_width - 1) // block_width, total_blocks)
        else:
            num_blocks_needed = total_blocks

        needed_blocks = {f"{self.model_name}_output_block{i}" for i in range(num_blocks_needed) 
                         if f"{self.model_name}_output_block{i}" in output_blocks}
        logger.debug(f"Найдено {len(needed_blocks)} выходных блоков из {total_blocks} для top_k={top_k}")
        return needed_blocks

    def load_block(self, block_info):
        block_hash = block_info.get("hash")
        if block_hash and hasattr(self, 'ipfs') and self.ipfs:
            self.ipfs.get(block_hash)
            return torch.load(f"{block_hash}.pt", map_location=self.device, weights_only=True)
        
        original_path = block_info["path"]
        block_filename = Path(original_path).name
        corrected_path = os.path.join(self.base_dir, block_filename)
        if not os.path.exists(corrected_path):
            raise FileNotFoundError(f"Файл блока {corrected_path} не найден")
        return torch.load(corrected_path, map_location=self.device, weights_only=True)

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
            dim = 0  # Сборка по строкам
            if target_shape[1] != first_width:
                raise ValueError(f"Ширина блоков {first_width} не совпадает с целевой шириной {target_shape[1]}")
        elif total_width == target_shape[1] and all(info["shape"][0] == first_height for info in blocks_info):
            dim = 1  # Сборка по столбцам
            if target_shape[0] != first_height:
                raise ValueError(f"Высота блоков {first_height} не совпадает с целевой высотой {target_shape[0]}")
        else:
            raise ValueError(f"Невозможно собрать тензор с целевой формой {target_shape} из блоков: "
                             f"суммарная высота={total_height}, ширина={total_width}")

        tensor = torch.zeros(target_shape, dtype=torch.float16, device=self.device)

        if dim == 0:
            current_row = 0
            for block_key in sorted_keys:
                block = self.load_block(self.metadata[block_key])
                block_height = block.shape[0]
                tensor[current_row:current_row + block_height, :] = block
                current_row += block_height
                del block
                gc.collect()
        else:
            current_col = 0
            for block_key in sorted_keys:
                block = self.load_block(self.metadata[block_key])
                block_width = block.shape[1]
                tensor[:, current_col:current_col + block_width] = block
                current_col += block_width
                del block
                gc.collect()

        return tensor

class MatrixModel(nn.Module):
    def __init__(self, dispatcher):
        super().__init__()
        self.dispatcher = dispatcher
        self.vocab_size = dispatcher.vocab_size
        self.hidden_size = dispatcher.hidden_size
        self.num_layers = dispatcher.num_layers
        self.device = dispatcher.device

    def forward(self, input_ids, top_k=None):
        if not isinstance(input_ids, torch.Tensor):
            input_ids = torch.tensor(input_ids, dtype=torch.long, device=self.device)
        elif input_ids.dtype != torch.long:
            input_ids = input_ids.long()
        
        if torch.any(input_ids >= self.vocab_size):
            raise ValueError(f"Входные данные содержат значения, превышающие vocab_size ({self.vocab_size})")
        
        batch_size, seq_len = input_ids.shape
        hidden_states = torch.zeros(batch_size, seq_len, self.hidden_size, dtype=torch.float16, device=self.device)
        
        embed_blocks = self.dispatcher.get_embedding_blocks(input_ids)
        for block_key in sorted(embed_blocks, key=lambda x: int(x.split("_block")[1])):
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

        for layer_idx in range(self.num_layers):
            layer_blocks = self.dispatcher.get_layer_blocks(layer_idx)
            # Пока пропускаем
            pass
        
        output_blocks = self.dispatcher.get_output_blocks(top_k=top_k)
        logits = torch.zeros(batch_size, seq_len, self.vocab_size, dtype=torch.float16, device=self.device)
        for block_key in sorted(output_blocks, key=lambda x: int(x.split("_block")[1])):
            block_info = self.dispatcher.metadata[block_key]
            block = self.dispatcher.load_block(block_info)
            block_height = block.shape[0]
            start_idx = int(block_key.split("_block")[1]) * block_height
            end_idx = start_idx + block_height
            
            logits[:, :, start_idx:end_idx] = F.linear(hidden_states, block)
            del block
            gc.collect()
        
        del hidden_states
        gc.collect()
        return logits

class VirtualSpace:
    def __init__(self, veector, use_ipfs=False, model_manager=None, metadata_dir="/workspaces/Veector/data"):
        self.veector = veector
        self.use_ipfs = use_ipfs
        self.model_manager = model_manager
        self.virtual_matrix = VirtualMatrix(self.veector.ipfs_client if use_ipfs else None)
        self.matrix_models = {}
        self.current_model = None
        self.metadata_dir = Path(metadata_dir)

    def switch_model(self, model_name, vocab_size, hidden_size, num_layers):
        metadata_path = self.metadata_dir / "blocks" / model_name / f"{model_name}_metadata.json"
        if not metadata_path.exists():
            raise ValueError(f"Метаданные для модели {model_name} не найдены в {metadata_path}")
        
        dispatcher = ModelDispatcher(model_name, metadata_path, vocab_size, hidden_size, num_layers)
        self.matrix_models[model_name] = MatrixModel(dispatcher)
        self.current_model = model_name
        print(f"Переключено на модель: {model_name}")

    def perform_inference(self, input_ids, top_k=None):
        if not self.current_model:
            raise ValueError("Не выбрана активная модель")
        
        model = self.matrix_models[self.current_model]
        if isinstance(input_ids, list):
            input_ids = torch.tensor(input_ids, dtype=torch.long, device=model.device)
        elif isinstance(input_ids, np.ndarray):
            input_ids = torch.from_numpy(input_ids).long().to(model.device)
        
        if torch.any(input_ids >= model.vocab_size):
            raise ValueError(f"Входные данные содержат значения, превышающие vocab_size ({model.vocab_size})")
        
        return model(input_ids, top_k=top_k)