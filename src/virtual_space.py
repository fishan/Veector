import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from core import Veector
from model_manager import ModelManager
from pathlib import Path
from collections import defaultdict

class VirtualMatrix:
    def __init__(self, ipfs_client=None, block_size=(512, 512)):
        """
        Виртуальная матрица для хранения моделей как 2D-матриц, разбитых на блоки.
        :param ipfs_client: Клиент IPFS для распределённого хранения.
        :param block_size: Размер блока (высота, ширина).
        """
        self.ipfs = ipfs_client
        self.block_size = block_size
        self.matrices = {}  # {model_name: {"matrix": tensor, "blocks": {coords: tensor}}}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def allocate_model(self, model_name, model):
        """
        Преобразование модели в 2D-матрицу и деление на блоки.
        :param model_name: Название модели.
        :param model: PyTorch-модель (nn.Module).
        """
        # "Расплющиваем" параметры модели в 1D-вектор
        flat_params = torch.cat([p.view(-1) for p in model.parameters()]).float()
        total_params = flat_params.numel()
        
        # Вычисляем размеры матрицы с padding
        width = self.block_size[1]
        height = (total_params + width - 1) // width  # Округляем вверх
        padded_height = ((height + self.block_size[0] - 1) // self.block_size[0]) * self.block_size[0]
        padded_size = padded_height * width
        
        # Создаём матрицу с padding
        padded_params = torch.zeros(padded_size, device="cpu")
        padded_params[:total_params] = flat_params
        matrix = padded_params.view(padded_height, width)

        # Делим на блоки
        blocks = {}
        for i in range(0, padded_height, self.block_size[0]):
            for j in range(0, width, self.block_size[1]):
                block = matrix[i:i + self.block_size[0], j:j + self.block_size[1]]
                coords = (i // self.block_size[0], j // self.block_size[1])
                block_name = f"{model_name}_row{coords[0]}_col{coords[1]}"
                block_hash = self._store_block(block_name, block)
                blocks[coords] = {"data": block, "hash": block_hash}

        self.matrices[model_name] = {"matrix": matrix, "blocks": blocks}
        print(f"Модель {model_name} преобразована в матрицу {matrix.shape} и разбита на {len(blocks)} блоков")

    def _store_block(self, block_name, block):
        """Сохранение блока в IPFS или локально."""
        block_path = f"data/blocks/{block_name}.pt"
        Path(block_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(block, block_path)
        if self.ipfs:
            return self.ipfs.add(block_path)["Hash"]
        return block_path

    def load_block(self, model_name, coords):
        """Загрузка блока по координатам."""
        if model_name not in self.matrices:
            raise ValueError(f"Модель {model_name} не найдена в матрице")
        block_info = self.matrices[model_name]["blocks"].get(coords)
        if not block_info:
            raise ValueError(f"Блок {coords} не найден для {model_name}")
        
        block_hash = block_info["hash"]
        if self.ipfs:
            self.ipfs.get(block_hash)
            return torch.load(f"{block_hash}.pt", map_location=self.device)
        return torch.load(block_hash, map_location=self.device)

class MatrixLoader:
    def __init__(self, virtual_matrix, max_cache_size=100):
        """
        Управление загрузкой блоков с вытеснением по LRU.
        :param virtual_matrix: Экземпляр VirtualMatrix.
        :param max_cache_size: Максимальное количество блоков в кэше.
        """
        self.matrix = virtual_matrix
        self.max_cache_size = max_cache_size
        self.loaded_blocks = {}  # {(model_name, coords): tensor}
        self.access_count = defaultdict(int)  # Счётчик доступа для LRU

    def get_block(self, model_name, coords):
        """Получение блока с кэшированием."""
        key = (model_name, coords)
        if key not in self.loaded_blocks:
            if len(self.loaded_blocks) >= self.max_cache_size:
                self._evict_least_used()
            self.loaded_blocks[key] = self.matrix.load_block(model_name, coords)
        self.access_count[key] += 1
        return self.loaded_blocks[key]

    def _evict_least_used(self):
        """Вытеснение наименее используемого блока."""
        lru_key = min(self.access_count.items(), key=lambda x: x[1])[0]
        del self.loaded_blocks[lru_key]
        del self.access_count[lru_key]

class MatrixModel(nn.Module):
    def __init__(self, original_model, matrix_loader):
        """
        Модель для инференса через виртуальную матрицу.
        :param original_model: Исходная PyTorch-модель для структуры.
        :param matrix_loader: Экземпляр MatrixLoader.
        """
        super().__init__()
        self.model = original_model
        self.loader = matrix_loader
        self.block_map = self._create_block_map()

    def _create_block_map(self):
        """Создание карты параметров → блоки."""
        block_map = defaultdict(list)
        offset = 0
        block_elements = self.loader.matrix.block_size[0] * self.loader.matrix.block_size[1]
        
        for param in self.model.parameters():
            param_size = param.numel()
            start_block = offset // block_elements
            end_block = (offset + param_size - 1) // block_elements
            for block_idx in range(start_block, end_block + 1):
                row = block_idx // (self.loader.matrix.matrices["current_model"]["matrix"].shape[1] // self.loader.matrix.block_size[1])
                col = block_idx % (self.loader.matrix.matrices["current_model"]["matrix"].shape[1] // self.loader.matrix.block_size[1])
                block_map[id(param)].append((row, col))
            offset += param_size
        return block_map

    def forward(self, input_ids):
        """Инференс через блочный доступ."""
        hidden_states = self._process_embeddings(input_ids)
        for layer in self.model.layers:  # Предполагаем структуру Qwen-1.5B
            hidden_states = self._apply_attention(layer, hidden_states)
            hidden_states = self._apply_mlp(layer, hidden_states)
        return self._apply_lm_head(hidden_states)

    def _process_embeddings(self, input_ids):
        embed_param = next(p for p in self.model.parameters())  # Предполагаем, что embeddings — первый параметр
        embed_blocks = self.block_map[id(embed_param)]
        embed_weight = torch.cat([self.loader.get_block("current_model", coords) for coords in embed_blocks], dim=0)
        embed_weight = embed_weight[:embed_param.numel()].view(embed_param.shape)
        return F.embedding(input_ids, embed_weight)

    def _apply_attention(self, layer, hidden_states):
        q_weight = layer.self_attn.q_proj.weight
        k_weight = layer.self_attn.k_proj.weight
        v_weight = layer.self_attn.v_proj.weight
        o_weight = layer.self_attn.o_proj.weight
        
        q_blocks = self.block_map[id(q_weight)]
        k_blocks = self.block_map[id(k_weight)]
        v_blocks = self.block_map[id(v_weight)]
        o_blocks = self.block_map[id(o_weight)]
        
        q = F.linear(hidden_states, self._reconstruct_param(q_weight, q_blocks))
        k = F.linear(hidden_states, self._reconstruct_param(k_weight, k_blocks))
        v = F.linear(hidden_states, self._reconstruct_param(v_weight, v_blocks))
        
        # RoPE (взято из Qwen)
        rotary_emb = layer.self_attn.rotary_emb(hidden_states.shape[1])
        q, k = rotary_emb(q), rotary_emb(k)
        
        attn_output = F.scaled_dot_product_attention(q, k, v)
        return F.linear(attn_output, self._reconstruct_param(o_weight, o_blocks)) + hidden_states

    def _apply_mlp(self, layer, hidden_states):
        gate_weight = layer.mlp.gate_proj.weight
        up_weight = layer.mlp.up_proj.weight
        down_weight = layer.mlp.down_proj.weight
        
        gate_blocks = self.block_map[id(gate_weight)]
        up_blocks = self.block_map[id(up_weight)]
        down_blocks = self.block_map[id(down_weight)]
        
        gate = F.linear(hidden_states, self._reconstruct_param(gate_weight, gate_blocks))
        up = F.linear(hidden_states, self._reconstruct_param(up_weight, up_blocks))
        down = F.linear(F.silu(gate) * up, self._reconstruct_param(down_weight, down_blocks))
        return down + hidden_states

    def _apply_lm_head(self, hidden_states):
        lm_head_weight = self.model.lm_head.weight
        lm_blocks = self.block_map[id(lm_head_weight)]
        return F.linear(hidden_states, self._reconstruct_param(lm_head_weight, lm_blocks))

    def _reconstruct_param(self, param, block_coords):
        """Сборка параметра из блоков."""
        blocks = [self.loader.get_block("current_model", coords) for coords in block_coords]
        flat_param = torch.cat(blocks, dim=0)[:param.numel()]
        return flat_param.view(param.shape)

class VirtualSpace:
    def __init__(self, veector, use_ipfs=False):
        """
        Виртуальное пространство для работы с множеством моделей.
        :param veector: Экземпляр Veector.
        :param use_ipfs: Использовать IPFS.
        """
        self.veector = veector
        self.use_ipfs = use_ipfs
        self.model_manager = ModelManager(self.veector, ipfs_enabled=use_ipfs)
        self.virtual_matrix = VirtualMatrix(self.veector.ipfs_client if use_ipfs else None)
        self.matrix_models = {}  # {model_name: MatrixModel}
        self.current_model = None

    def load_full_model_into_matrix(self, model_name, model_path):
        """Загрузка модели в виртуальную матрицу."""
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(model_path)
        self.model_manager.add_model(model_name, model_path)
        self.virtual_matrix.allocate_model(model_name, model)
        loader = MatrixLoader(self.virtual_matrix)
        self.matrix_models[model_name] = MatrixModel(model, loader)
        self.switch_model(model_name)
        print(f"Модель {model_name} загружена в VirtualSpace")

    def load_blocks_model_into_matrix(self, model_name, blocks_dir="data/blocks"):
        """Загрузка модели из предразложенных блоков."""
        from transformers import AutoModelForCausalLM
        # Загружаем только структуру модели (без весов)
        model = AutoModelForCausalLM.from_pretrained(model_name, ignore_mismatched_sizes=True)
        self.model_manager.add_model(model_name, blocks_dir)
        
        # Читаем блоки из директории
        blocks = {}
        for block_file in Path(blocks_dir).glob(f"{model_name}_row*_col*.pt"):
            coords = tuple(map(int, block_file.stem.split("_")[1:3:2]))  # rowX, colY
            blocks[coords] = {"path": str(block_file)}
        self.virtual_matrix.matrices[model_name] = {"blocks": blocks}
        import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from core import Veector
from model_manager import ModelManager
from pathlib import Path
from collections import defaultdict
import gc  # Добавлено для управления памятью

class VirtualMatrix:
    def __init__(self, ipfs_client=None, block_size=(1024, 1024)):  # Изменено: размер блока на (1024, 1024)
        """
        Виртуальная матрица для хранения моделей как 2D-матриц, разбитых на блоки.
        :param ipfs_client: Клиент IPFS для распределённого хранения.
        :param block_size: Размер блока (высота, ширина).
        """
        self.ipfs = ipfs_client
        self.block_size = block_size
        self.matrices = {}  # {model_name: {"matrix": tensor, "blocks": {coords: {"path": str, "hash": str}}}}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def allocate_model(self, model_name, model):
        """
        Преобразование модели в 2D-матрицу и деление на блоки.
        :param model_name: Название модели.
        :param model: PyTorch-модель (nn.Module).
        """
        # "Расплющиваем" параметры модели в 1D-вектор с float16 для экономии памяти
        flat_params = torch.cat([p.view(-1).to(dtype=torch.float16) for p in model.parameters()])
        total_params = flat_params.numel()
        
        # Вычисляем размеры матрицы с padding
        width = self.block_size[1]
        height = (total_params + width - 1) // width  # Округляем вверх
        padded_height = ((height + self.block_size[0] - 1) // self.block_size[0]) * self.block_size[0]
        padded_size = padded_height * width
        
        # Создаём матрицу с padding
        padded_params = torch.zeros(padded_size, dtype=torch.float16, device="cpu")  # Изменено: float16
        padded_params[:total_params] = flat_params
        matrix = padded_params.view(padded_height, width)
        del flat_params  # Очистка памяти
        gc.collect()

        # Делим на блоки
        blocks = {}
        for i in range(0, padded_height, self.block_size[0]):
            for j in range(0, width, self.block_size[1]):
                block = matrix[i:i + self.block_size[0], j:j + self.block_size[1]]
                coords = (i // self.block_size[0], j // self.block_size[1])
                block_name = f"{model_name}_row{coords[0]}_col{coords[1]}"
                block_hash = self._store_block(block_name, block)
                blocks[coords] = {"path": f"data/blocks/{block_name}.pt", "hash": block_hash}
                del block  # Очистка памяти после сохранения
                gc.collect()

        self.matrices[model_name] = {"matrix": None, "blocks": blocks}  # Изменено: matrix не храним в памяти
        print(f"Модель {model_name} преобразована в матрицу [{padded_height}, {width}] и разбита на {len(blocks)} блоков")
        del matrix  # Очистка памяти
        gc.collect()

    def _store_block(self, block_name, block):
        """Сохранение блока в IPFS или локально."""
        block_path = f"data/blocks/{block_name}.pt"
        Path(block_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(block, block_path)
        if self.ipfs:
            return self.ipfs.add(block_path)["Hash"]
        return block_path

    def load_block(self, model_name, coords):
        """Загрузка блока по координатам."""
        if model_name not in self.matrices:
            raise ValueError(f"Модель {model_name} не найдена в матрице")
        block_info = self.matrices[model_name]["blocks"].get(coords)
        if not block_info:
            raise ValueError(f"Блок {coords} не найден для {model_name}")
        
        block_hash = block_info["hash"]
        if self.ipfs and block_hash:
            self.ipfs.get(block_hash)
            return torch.load(f"{block_hash}.pt", map_location=self.device)
        return torch.load(block_info["path"], map_location=self.device)

class MatrixLoader:
    def __init__(self, virtual_matrix, max_cache_size=50):  # Изменено: уменьшен кэш до 50 для (1024, 1024)
        """
        Управление загрузкой блоков с вытеснением по LRU.
        :param virtual_matrix: Экземпляр VirtualMatrix.
        :param max_cache_size: Максимальное количество блоков в кэше.
        """
        self.matrix = virtual_matrix
        self.max_cache_size = max_cache_size
        self.loaded_blocks = {}  # {(model_name, coords): tensor}
        self.access_count = defaultdict(int)  # Счётчик доступа для LRU

    def get_block(self, model_name, coords):
        """Получение блока с кэшированием."""
        key = (model_name, coords)
        if key not in self.loaded_blocks:
            if len(self.loaded_blocks) >= self.max_cache_size:
                self._evict_least_used()
            self.loaded_blocks[key] = self.matrix.load_block(model_name, coords)
        self.access_count[key] += 1
        return self.loaded_blocks[key]

    def _evict_least_used(self):
        """Вытеснение наименее используемого блока."""
        lru_key = min(self.access_count.items(), key=lambda x: x[1])[0]
        del self.loaded_blocks[lru_key]
        del self.access_count[lru_key]

class MatrixModel(nn.Module):
    def __init__(self, original_model, matrix_loader):
        """
        Модель для инференса через виртуальную матрицу.
        :param original_model: Исходная PyTorch-модель для структуры.
        :param matrix_loader: Экземпляр MatrixLoader.
        """
        super().__init__()
        self.model = original_model
        self.loader = matrix_loader
        self.block_map = self._create_block_map()

    def _create_block_map(self):
        """Создание карты параметров → блоки."""
        block_map = defaultdict(list)
        offset = 0
        block_elements = self.loader.matrix.block_size[0] * self.loader.matrix.block_size[1]
        
        for param in self.model.parameters():
            param_size = param.numel()
            start_block = offset // block_elements
            end_block = (offset + param_size - 1) // block_elements
            for block_idx in range(start_block, end_block + 1):
                row = block_idx // (self.loader.matrix.matrices["current_model"]["blocks"].keys().__iter__().__next__()[1] + 1)  # Изменено: динамическая ширина
                col = block_idx % (self.loader.matrix.matrices["current_model"]["blocks"].keys().__iter__().__next__()[1] + 1)
                block_map[id(param)].append((row, col))
            offset += param_size
        return block_map

    def forward(self, input_ids):
        """Инференс через блочный доступ."""
        hidden_states = self._process_embeddings(input_ids)
        for layer in self.model.layers:  # Предполагаем структуру Qwen-1.5B
            hidden_states = self._apply_attention(layer, hidden_states)
            hidden_states = self._apply_mlp(layer, hidden_states)
        return self._apply_lm_head(hidden_states)

    def _process_embeddings(self, input_ids):
        embed_param = next(p for p in self.model.parameters())
        embed_blocks = self.block_map[id(embed_param)]
        embed_weight = torch.cat([self.loader.get_block("current_model", coords) for coords in embed_blocks], dim=0)
        embed_weight = embed_weight[:embed_param.numel()].view(embed_param.shape).to(self.loader.matrix.device)
        return F.embedding(input_ids, embed_weight)

    def _apply_attention(self, layer, hidden_states):
        q_weight = layer.self_attn.q_proj.weight
        k_weight = layer.self_attn.k_proj.weight
        v_weight = layer.self_attn.v_proj.weight
        o_weight = layer.self_attn.o_proj.weight
        
        q_blocks = self.block_map[id(q_weight)]
        k_blocks = self.block_map[id(k_weight)]
        v_blocks = self.block_map[id(v_weight)]
        o_blocks = self.block_map[id(o_weight)]
        
        q = F.linear(hidden_states, self._reconstruct_param(q_weight, q_blocks))
        k = F.linear(hidden_states, self._reconstruct_param(k_weight, k_blocks))
        v = F.linear(hidden_states, self._reconstruct_param(v_weight, v_blocks))
        
        rotary_emb = layer.self_attn.rotary_emb(hidden_states.shape[1])
        q, k = rotary_emb(q), rotary_emb(k)
        
        attn_output = F.scaled_dot_product_attention(q, k, v)
        return F.linear(attn_output, self._reconstruct_param(o_weight, o_blocks)) + hidden_states

    def _apply_mlp(self, layer, hidden_states):
        gate_weight = layer.mlp.gate_proj.weight
        up_weight = layer.mlp.up_proj.weight
        down_weight = layer.mlp.down_proj.weight
        
        gate_blocks = self.block_map[id(gate_weight)]
        up_blocks = self.block_map[id(up_weight)]
        down_blocks = self.block_map[id(down_weight)]
        
        gate = F.linear(hidden_states, self._reconstruct_param(gate_weight, gate_blocks))
        up = F.linear(hidden_states, self._reconstruct_param(up_weight, up_blocks))
        down = F.linear(F.silu(gate) * up, self._reconstruct_param(down_weight, down_blocks))
        return down + hidden_states

    def _apply_lm_head(self, hidden_states):
        lm_head_weight = self.model.lm_head.weight
        lm_blocks = self.block_map[id(lm_head_weight)]
        return F.linear(hidden_states, self._reconstruct_param(lm_head_weight, lm_blocks))

    def _reconstruct_param(self, param, block_coords):
        """Сборка параметра из блоков."""
        blocks = [self.loader.get_block("current_model", coords) for coords in block_coords]
        flat_param = torch.cat(blocks, dim=0)[:param.numel()]
        return flat_param.view(param.shape).to(self.loader.matrix.device)  # Изменено: перенос на device

class VirtualSpace:
    def __init__(self, veector, use_ipfs=False):
        """
        Виртуальное пространство для работы с множеством моделей.
        :param veector: Экземпляр Veector.
        :param use_ipfs: Использовать IPFS.
        """
        self.veector = veector
        self.use_ipfs = use_ipfs
        self.model_manager = ModelManager(self.veector, ipfs_enabled=use_ipfs)
        self.virtual_matrix = VirtualMatrix(self.veector.ipfs_client if use_ipfs else None, block_size=(1024, 1024))  # Изменено: (1024, 1024)
        self.matrix_models = {}
        self.current_model = None

    def load_full_model_into_matrix(self, model_name, model_path):
        """Загрузка полной модели в виртуальную матрицу."""
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)  # Изменено: float16
        self.model_manager.add_model(model_name, model_path)
        self.virtual_matrix.allocate_model(model_name, model)
        loader = MatrixLoader(self.virtual_matrix)
        self.matrix_models[model_name] = MatrixModel(model, loader)
        self.switch_model(model_name)
        if self.use_ipfs and self.veector.p2p_node:  # Добавлено: синхронизация через P2PNode
            self.veector.p2p_node.sync_model_blocks(model_name, "data/blocks")
        print(f"Модель {model_name} загружена в VirtualSpace")

    def load_blocks_model_into_matrix(self, model_name, blocks_dir="data/blocks"):
        """Загрузка модели из предразложенных блоков."""
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(model_name, ignore_mismatched_sizes=True, torch_dtype=torch.float16)  # Изменено: float16
        self.model_manager.add_model(model_name, blocks_dir)
        
        blocks = {}
        for block_file in Path(blocks_dir).glob(f"{model_name}_row*_col*.pt"):
            coords = tuple(map(int, block_file.stem.split("_")[1:3:2]))
            block_hash = self.veector.p2p_node.block_map.get(model_name, {}).get(coords) if self.use_ipfs else None  # Добавлено: получение хэша из P2PNode
            blocks[coords] = {"path": str(block_file), "hash": block_hash}
        self.virtual_matrix.matrices[model_name] = {"blocks": blocks}
        
        if self.use_ipfs and self.veector.p2p_node:  # Добавлено: синхронизация, если хэши отсутствуют
            self.veector.p2p_node.sync_model_blocks(model_name, blocks_dir)
            for coords in blocks:
                blocks[coords]["hash"] = self.veector.p2p_node.block_map.get(model_name, {}).get(coords)
        
        loader = MatrixLoader(self.virtual_matrix)
        self.matrix_models[model_name] = MatrixModel(model, loader)
        self.switch_model(model_name)
        print(f"Модель {model_name} загружена из блоков в {blocks_dir}")

    def switch_model(self, model_name):
        """Переключение активной модели."""
        if model_name not in self.matrix_models:
            raise ValueError(f"Модель {model_name} не найдена")
        self.current_model = model_name
        self.virtual_matrix.matrices["current_model"] = self.virtual_matrix.matrices[model_name]

    def perform_inference(self, input_ids):
        """Выполнение инференса для текущей модели."""
        if not self.current_model:
            raise ValueError("Не выбрана активная модель")
        return self.matrix_models[self.current_model](input_ids)

if __name__ == "__main__":
    from sync import P2PNode
    p2p_node = P2PNode("localhost", 5000, use_ipfs=True)
    p2p_node.start()
    veector = Veector(p2p_node=p2p_node)
    virtual_space = VirtualSpace(veector, use_ipfs=True)
    
    # Загрузка модели из блоков (предполагается, что блоки уже созданы в Colab)
    virtual_space.load_blocks_model_into_matrix("DeepSeek-R1-Distill-Qwen-1.5B", "data/blocks")
    
    # Пример инференса
    input_ids = torch.randint(0, 32768, (1, 512))
    output = virtual_space.perform_inference(input_ids)
    print(f"Вывод: {output.shape}")
    loader = MatrixLoader(self.virtual_matrix)
    self.matrix_models[model_name] = MatrixModel(model, loader)
    self.switch_model(model_name)
    print(f"Модель {model_name} загружена из блоков в {blocks_dir}")

    def switch_model(self, model_name):
        """Переключение активной модели."""
        if model_name not in self.matrix_models:
            raise ValueError(f"Модель {model_name} не найдена")
        self.current_model = model_name
        self.virtual_matrix.matrices["current_model"] = self.virtual_matrix.matrices[model_name]

    def perform_inference(self, input_ids):
        """Выполнение инференса для текущей модели."""
        if not self.current_model:
            raise ValueError("Не выбрана активная модель")
        return self.matrix_models[self.current_model](input_ids)

if __name__ == "__main__":
    from sync import P2PNode
    p2p_node = P2PNode("localhost", 5000, use_ipfs=True)
    p2p_node.start()
    veector = Veector(p2p_node=p2p_node)
    virtual_space = VirtualSpace(veector, use_ipfs=True)
    
    # Загрузка модели (в Colab)
    virtual_space.load_model_into_matrix("deepseek-r1-distill-qwen-1.5b", "path/to/model")
    
    # Пример инференса
    input_ids = torch.randint(0, 32768, (1, 512))
    output = virtual_space.perform_inference(input_ids)
    print(f"Вывод: {output.shape}")