import numpy as np
from veectordb import VeectorDB
import torch
import torch.nn as nn
import queue
import threading
import time
import random
from operations import (matrix_multiply, gradient_descent, softmax, matrix_determinant, matrix_eigenvalues, 
                        matrix_lu_decomposition, convolution, transpose, mean, std_dev, relu, sigmoid, 
                        exponential_smoothing, normalize, interpolate, self_attention, layer_normalization, 
                        multi_head_attention)
from memory import Memory
from evolution import Evolution
from model_manager import ModelManager
from sync import P2PNode
from tensors import create_tensor, validate_tensor, reshape_tensor, get_tensor_metadata

class NeuralStorage(nn.Module):
    def __init__(self, input_dim=16, hidden_dim=64, bottleneck_dim=32, activation_fn=nn.ReLU):
        super(NeuralStorage, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            activation_fn(),
            nn.Linear(hidden_dim, bottleneck_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim),
            activation_fn(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

class Veector:
    def __init__(self, db_path="vectordb.json", use_neural_storage=False, cache_size=1000, 
                 eviction_strategy="LRU", dropout_rate=0.0, use_memory=False, model_manager=None, p2p_node=None):
        self.db = VeectorDB(db_path)
        self.use_neural_storage = use_neural_storage
        self.neural_model = None
        self.max_coord = 0
        self.neural_embeddings = {}
        self.sync_queue = queue.Queue()
        self.cache = {}
        self.cache_size = cache_size
        self.eviction_strategy = eviction_strategy.upper()
        self.cache_access_count = {}
        self.cache_timestamps = {}
        self.dropout_rate = dropout_rate
        self.use_memory = use_memory
        self.memory = Memory()
        self.evolution = Evolution(self)
        self.model_manager = model_manager
        self.p2p_node = p2p_node

        if use_neural_storage:
            self._init_neural_storage()
        self._start_sync_thread()

        self.core = {
            (1, 0, 0): lambda x: np.sum(x),
            (1, 1, 1): lambda x: x[0] - x[1],
            (0, 1, 0): lambda x: x[0] * x[1],
            (0, 0, 1): lambda x: x[0] / x[1],
            (1, 0, 1): lambda x: np.sqrt(x[0]),
            (1, 1, 0): lambda x: np.power(x[0], x[1]),
            (1, 2, 0): lambda x: np.abs(x[0]),
            (1, 3, 0): lambda x: np.dot(x[0], x[1]) if len(x[0]) == len(x[1][0]) else None,
            (2, 1, 0): lambda x: np.sin(x[0]),
            (2, 1, 1): lambda x: np.cos(x[0]),
            (2, 2, 0): lambda x: np.tan(x[0]),
            (2, 2, 1): lambda x: 1 / np.tan(x[0]),
            (3, 1, 0): lambda x: np.log(x[0]),
            (3, 1, 1): lambda x: np.exp(x[0]),
            (4, 1, 0): lambda x: np.dot(x[0], x[1]),
            (4, 1, 1): lambda x: np.arccos(np.dot(x[0], x[1]) / (np.linalg.norm(x[0]) * np.linalg.norm(x[1]))),
            (2, 0, 0): lambda x: 1 if x[0] > x[1] else 0,
            (2, 0, 1): lambda x: 1 if x[0] == x[1] else 0,
            (2, 3, 0): lambda x: 1 if x[0] and x[1] else 0,
            (2, 3, 1): lambda x: 1 if x[0] or x[1] else 0,
            (2, 4, 0): lambda x: 1 if not x[0] else 0,
            (3, 0, 0): lambda x, t, f: t if x[0] else f,
            (4, 0, 0): lambda x, n: x[0] * n,
            (5, 0, 0): lambda x, *opts: opts[x[0]],
            (6, 0, 0): lambda x: max(x[0], 0),
            (11, 1, 0): lambda x: 1 / (1 + np.exp(-x[0])),
            (7, 0, 0): lambda x: print(f"Output: {x[0]}"),
            (8, 0, 0): lambda x: x,
            (9, 0, 0): lambda x: self._reason(x),
            (10, 1, 0): lambda x: self._dfs(x[0], x[1]),
            (50, 0, 0): matrix_multiply,
            (51, 0, 0): matrix_determinant,
            (52, 0, 0): matrix_eigenvalues,
            (53, 0, 0): convolution,
            (54, 0, 0): transpose,
            (55, 0, 0): mean,
            (56, 0, 0): std_dev,
            (57, 0, 0): sigmoid,
            (58, 0, 0): relu,
            (59, 0, 0): lambda x: self._dropout(x),
            (60, 0, 0): exponential_smoothing,
            (61, 0, 0): normalize,
            (62, 0, 0): interpolate,
            (70, 0, 0): self_attention,  # Добавляем self-attention
            (71, 0, 0): layer_normalization,  # Добавляем layer normalization
            (72, 0, 0): lambda x: multi_head_attention(x, num_heads=8),  # Добавляем multi-head attention
        }

    def _init_neural_storage(self):
        print("Инициализация нейронного хранилища")
        input_dim = self._get_max_input_dim()
        self.neural_model = NeuralStorage(input_dim=input_dim, activation_fn=nn.ReLU)
        self.neural_optimizer = torch.optim.Adam(self.neural_model.parameters(), lr=0.001)
        self.neural_loss = nn.MSELoss()
        self._train_neural_storage()

    def _get_max_input_dim(self):
        results = self.db.find_by_type("tensor_result")
        max_dim = 16
        for doc in results:
            result = doc["data"]
            if isinstance(result, np.ndarray):
                flat_len = len(result.flatten())
                max_dim = max(max_dim, flat_len)
        return max_dim

    def _train_neural_storage(self):
        results = self.db.find_by_type("tensor_result")
        if not results:
            print("Нет данных для обучения нейросети")
            return

        input_dim = self._get_max_input_dim()
        train_data = []
        for doc in results:
            result = doc["data"]
            if isinstance(result, (int, float)):
                data = np.array([float(result)] + [0] * (input_dim - 1))
            elif isinstance(result, list) and all(isinstance(x, (int, float)) for x in result):
                flat = np.array(result).flatten()
                data = np.pad(flat, (0, max(0, input_dim - len(flat))), mode='constant')[:input_dim]
            else:
                data = np.array([0] * input_dim)
            train_data.append(data)

        train_data = torch.tensor(train_data, dtype=torch.float32)

        for epoch in range(50):
            self.neural_optimizer.zero_grad()
            encoded, decoded = self.neural_model(train_data)
            loss = self.neural_loss(decoded, train_data)
            loss.backward()
            self.neural_optimizer.step()
            if epoch % 10 == 0:
                print(f"Эпоха {epoch + 1}, Loss: {loss.item()}")

    def _store_in_neural(self, result, doc_id):
        if not self.use_neural_storage or not self.neural_model:
            return

        input_dim = self._get_max_input_dim()
        if isinstance(result, (int, float, np.number)):
            data = np.array([float(result)] + [0] * (input_dim - 1))
        elif isinstance(result, np.ndarray):
            flat = result.flatten()
            data = np.pad(flat, (0, max(0, input_dim - len(flat))), mode='constant')[:input_dim]
        else:
            data = np.array([0] * input_dim)

        tensor_data = torch.tensor(data, dtype=torch.float32)
        encoded, _ = self.neural_model(tensor_data)
        self.neural_embeddings[doc_id] = encoded.detach().numpy()
        print(f"Сохранено в нейросеть: {doc_id} -> {encoded.detach().numpy()[:5]}...")

    def _retrieve_from_neural(self, doc_id):
        if not self.use_neural_storage or not self.neural_model or doc_id not in self.neural_embeddings:
            return None

        encoded = torch.tensor(self.neural_embeddings[doc_id], dtype=torch.float32)
        decoded = self.neural_model.decoder(encoded)
        return decoded.detach().numpy()

    def _start_sync_thread(self):
        def sync_worker():
            while True:
                peer_veector = self.sync_queue.get()
                self._sync_neural_blocking(peer_veector)
                self.sync_queue.task_done()

        t = threading.Thread(target=sync_worker, daemon=True)
        t.start()

    def _sync_neural_blocking(self, peer_veector):
        if not self.use_neural_storage or not peer_veector.use_neural_storage:
            return

        if not self.neural_model or not peer_veector.neural_model:
            return

        print("Синхронизация нейронных моделей (федеративное обучение)")
        self_data_count = len(self.db.find_by_type("tensor_result"))
        peer_data_count = len(peer_veector.db.find_by_type("tensor_result"))
        total_data = self_data_count + peer_data_count

        if total_data == 0:
            return

        state_dict = self.neural_model.state_dict()
        peer_state_dict = peer_veector.neural_model.state_dict()

        for key in state_dict:
            self_weight = self_data_count / total_data
            peer_weight = peer_data_count / total_data
            state_dict[key] = self_weight * state_dict[key] + peer_weight * peer_state_dict[key]

        self.neural_model.load_state_dict(state_dict)

    def sync_neural(self, peer_veector):
        self.sync_queue.put(peer_veector)

    def _reason(self, x):
        print(f"Reason input: {x}")

        if self.use_memory:
            cached_result = self.memory.retrieve(x)
            if cached_result is not None:
                print(f"Использована память для Reason: {x} -> {cached_result}")
                return cached_result

        if isinstance(x, (int, float, np.number)):
            if x > 10:
                result = np.log(x)
            else:
                result = x * 2

        elif isinstance(x, list):
            if len(x) == 1:
                result = self._reason(x[0])
            elif len(x) > 0 and isinstance(x[0], list):
                result = [self._reason(sub_x) for sub_x in x]
            else:
                result = None

        elif len(x) == 2:
            result = 1 if x[0] > x[1] else 0

        elif isinstance(x, np.ndarray):
            if x.ndim == 2 and x.shape[0] == x.shape[1]:
                result = np.dot(x, x)
            else:
                result = np.sum(x)

        else:
            result = x

        if self.use_memory:
            self.memory.store(x, result)
            print(f"Сохранено в памяти: {x} -> {result}")

        print(f"Reason result: {result}")
        return result

    def _next_coords(self):
        coords = max([key[1][0] for key in self.space.keys()] + [self.max_coord]) + 1
        self.max_coord = coords
        return coords

    def _dfs(self, graph, start):
        visited = set()
        result = []

        def dfs(node):
            if node not in visited:
                visited.add(node)
                result.append(node)
                for neighbor in graph.get(node, []):
                    dfs(neighbor)

        dfs(start)
        return result

    def _lru_cache_evict(self):
        if len(self.cache) >= self.cache_size:
            oldest_key = min(self.cache_timestamps, key=self.cache_timestamps.get)
            del self.cache[oldest_key]
            del self.cache_timestamps[oldest_key]
            if oldest_key in self.cache_access_count:
                del self.cache_access_count[oldest_key]

    def _lfu_cache_evict(self):
        if len(self.cache) >= self.cache_size:
            least_frequent_key = min(self.cache_access_count, key=self.cache_access_count.get)
            del self.cache[least_frequent_key]
            del self.cache_access_count[least_frequent_key]
            if least_frequent_key in self.cache_timestamps:
                del self.cache_timestamps[least_frequent_key]

    def _dropout(self, x):
        if self.dropout_rate > 0 and isinstance(x, np.ndarray):
            mask = (np.random.rand(*x.shape) < self.dropout_rate)
            x[mask] = 0
        return x

    def compute(self, tensor):
        if not validate_tensor(tensor):
            return tensor

        data_layer, data_coords, data, data_length = tensor[0]
        op_layer, op_coords, op, op_length = tensor[1]
        context = tensor[2]
        version = tensor[3]
        next_coords = tensor[4] if len(tensor) > 4 else []
        metadata = get_tensor_metadata(tensor)

        cache_key = (tuple(data_layer), tuple(data_coords), tuple(op))
        if cache_key in self.cache:
            self.cache_access_count[cache_key] = self.cache_access_count.get(cache_key, 0) + 1
            self.cache_timestamps[cache_key] = time.time()
            return self.cache[cache_key]

        if isinstance(data, list):
            data = [self.compute(d) if isinstance(d, list) else d for d in data]
        if len(data) == 1 and tuple(op) in [(2, 1, 0), (2, 1, 1), (2, 2, 0), (2, 2, 1)]:
            data = data
        elif len(data) == 1 and not isinstance(data[0], list):
            data = data[0]

        op_func = self.core.get(tuple(op), lambda x: x)

        if op == [3, 0, 0]:
            cond = self.compute(data[0]) if isinstance(data, list) else data
            true_val = self.compute(data[1])
            false_val = self.compute(data[2])
            result = op_func([cond], true_val, false_val)
        elif op == [4, 0, 0]:
            if isinstance(data, list) and len(data) > 1:
                result = op_func(data[0], data[1])
            else:
                result = op_func(data, 1)
        elif op == [5, 0, 0]:
            opts = [self.compute(opt) for opt in data[1:]]
            result = op_func(data[0], *opts)
        else:
            if isinstance(data, list) and tuple(op) not in [(2, 1, 0), (2, 1, 1), (2, 2, 0), (2, 2, 1)]:
                data = np.array(data)
            if self.dropout_rate > 0 and op != [59, 0, 0]:
                data = self._dropout(data)

            result = op_func(data)

        metadata = {"tensor": tensor, "coords": (data_layer, data_coords)}
        doc_id = self.db.insert("tensor_result", result, metadata)

        if self.use_neural_storage and self.neural_model:
            self._store_in_neural(result, doc_id)

        if self.p2p_node:
            self.p2p_node.sync_tensor(result, metadata)

        self.space[(tuple(data_layer), tuple(data_coords))] = doc_id

        if len(self.cache) >= self.cache_size:
            if self.eviction_strategy == "LRU":
                self._lru_cache_evict()
            elif self.eviction_strategy == "LFU":
                self._lfu_cache_evict()
            else:
                self._lru_cache_evict()

        self.cache[cache_key] = result
        self.cache_access_count[cache_key] = 1
        self.cache_timestamps[cache_key] = time.time()

        return result

    def add_to_space(self, tensor):
        layer, coords = tensor[0][0], tuple(tensor[0][1])
        doc_id = self.db.insert("tensor", tensor)
        self.space[(tuple(layer), coords)] = doc_id

    def evolve_tensor(self, tensor):
        return self.evolution.log_evolution(tensor, self)