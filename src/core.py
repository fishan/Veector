import numpy as np
from veectordb import VeectorDB
import torch
import torch.nn as nn
import queue
import threading
import time
import random
from qiskit import QuantumCircuit, Aer, execute  # Интеграция Qiskit для квантовых операций
from operations import (
    mod, floor, ceil, arcsin, arccos, arctan, xor, nand, nor, matrix_multiply,
    gradient_descent, softmax, matrix_determinant, matrix_eigenvalues, matrix_lu_decomposition,
    convolution, transpose, mean, std_dev, relu, leaky_relu, batch_norm, sigmoid,
    exponential_smoothing, normalize, interpolate, inverse, trace, random_uniform,
    random_normal, median, dropout, self_attention, layer_normalization,
    multi_head_attention, quantum_hadamard, quantum_pauli_x, quantum_cnot,
    quantum_measure, quantum_superposition, quantum_entanglement, causal_mask, masked_fill
)
from memory import Memory
from evolution import Evolution
from model_manager import ModelManager
from sync import P2PNode
from tensors import create_tensor, validate_tensor, reshape_tensor, get_tensor_metadata
import ipfshttpclient
import os

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
    def __init__(self, db_path="data/db/veectordb.json", use_neural_storage=False, cache_size=1000,
                 eviction_strategy="LRU", dropout_rate=0.0, use_memory=False, model_manager=None, p2p_node=None,
                 ipfs_address='/ip4/127.0.0.1/tcp/5001'):
        self.db = VeectorDB(db_path)
        self.use_neural_storage = use_neural_storage
        self.neural_model = None
        self.max_coord = 0
        self.space = {}
        self.neural_embeddings = {}
        self.sync_queue = queue.Queue()
        self.cache = {}
        self.cache_size = cache_size
        self.eviction_strategy = eviction_strategy.upper()
        self.cache_access_count = {}
        self.cache_timestamps = {}
        self.dropout_rate = dropout_rate
        self.use_memory = Memory() if use_memory else None  # Исправлено: инициализация Memory
        self.evolution = Evolution(self)
        self.model_manager = model_manager or ModelManager(self)  # Исправлено: дефолтный ModelManager
        self.p2p_node = p2p_node
        self.ipfs_client = ipfshttpclient.connect(addr=ipfs_address)
        self.models_dir = "data/models"
        self.tensors_dir = "data/tensors"

        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.tensors_dir, exist_ok=True)

        if use_neural_storage:
            self._init_neural_storage()
        self._start_sync_thread()

        self.core = {
            # Арифметика (0-9)
            (0, 0, 0): lambda x: np.sum(x, dtype=np.complex128),  # Поддержка комплексных чисел
            (0, 0, 1): lambda x: x[0] - x[1],
            (0, 1, 0): lambda x: x[0] * x[1],
            (0, 1, 1): lambda x: x[0] / x[1],
            (0, 2, 0): lambda x: np.sqrt(x[0], dtype=np.complex128),
            (0, 2, 1): lambda x: np.power(x[0], x[1], dtype=np.complex128),
            (0, 3, 0): lambda x: np.abs(x[0]),
            (0, 4, 0): lambda x: np.dot(x[0], x[1]) if x[0].shape[1] == x[1].shape[0] else None,
            (0, 5, 0): lambda x: mod(x[0], x[1]),
            (0, 6, 0): lambda x: floor(x[0]),
            (0, 6, 1): lambda x: ceil(x[0]),

            # Тригонометрия (1)
            (1, 0, 0): lambda x: np.sin(x[0], dtype=np.complex128),
            (1, 0, 1): lambda x: np.cos(x[0], dtype=np.complex128),
            (1, 1, 0): lambda x: np.tan(x[0], dtype=np.complex128),
            (1, 1, 1): lambda x: 1 / np.tan(x[0], dtype=np.complex128) if np.tan(x[0]) != 0 else np.nan,
            (1, 2, 0): lambda x: arcsin(x[0]),
            (1, 2, 1): lambda x: arccos(x[0]),
            (1, 3, 0): lambda x: arctan(x[0]),

            # Логика (2)
            (2, 0, 0): lambda x: 1 if x[0] > x[1] else 0,
            (2, 0, 1): lambda x: 1 if x[0] == x[1] else 0,
            (2, 1, 0): lambda x: 1 if x[0] and x[1] else 0,
            (2, 1, 1): lambda x: 1 if x[0] or x[1] else 0,
            (2, 2, 0): lambda x: 1 if not x[0] else 0,
            (2, 3, 0): lambda x: xor(x[0], x[1]),
            (2, 4, 0): nand,
            (2, 4, 1): nor,

            # Условные операции (3)
            (3, 0, 0): lambda x, t, f: t if x[0] else f,

            # Циклы (4)
            (4, 0, 0): lambda x, n: x[0] * n,

            # Рандом (5)
            (5, 1, 0): lambda x: random_uniform(x[0], x[1]),
            (5, 1, 1): lambda x: random_normal(x[0], x[1]),  # Исправлено: два аргумента
            (5, 2, 0): lambda x: median(x[0]),

            # Выбор (7)
            (7, 0, 0): lambda x, *opts: opts[int(x[0])] if opts else None,

            # Вывод (8)
            (8, 0, 0): lambda x: print(f"Output: {x[0]}"),

            # Идентичность (9)
            (9, 0, 0): lambda x: x[0],

            # Эволюция (10)
            (10, 0, 0): lambda x: self._reason(x),

            # Графовые операции (15)
            (15, 0, 0): lambda x: self._dfs(x[0], x[1]),

            # Статистика (16)
            (16, 0, 0): lambda x: np.mean(x[0], dtype=np.complex128),
            (16, 1, 0): lambda x: np.std(x[0], dtype=np.complex128),

            # Регуляризация (17)
            (17, 0, 0): lambda x: self._dropout(x[0]),

            # Активации (18)
            (18, 0, 0): lambda x: np.maximum(x[0], 0),
            (18, 1, 0): lambda x: 1 / (1 + np.exp(-x[0])),
            (18, 2, 0): softmax,
            (18, 3, 0): leaky_relu,

            # Сглаживание (19)
            (19, 0, 0): exponential_smoothing,

            # Нормализация (20)
            (20, 0, 0): normalize,
            (20, 1, 0): interpolate,

            # Матричные операции (30)
            (30, 0, 0): matrix_multiply,
            (30, 1, 0): matrix_determinant,
            (30, 2, 0): matrix_eigenvalues,
            (30, 3, 0): convolution,
            (30, 4, 0): transpose,
            (30, 5, 0): inverse,
            (30, 6, 0): trace,

            # Нейросетевые операции (40)
            (40, 0, 0): lambda x: self.model_manager.perform_inference("DeepSeek-R1-Distill-Qwen-1.5B", x),
            (40, 1, 0): layer_normalization,
            (40, 2, 0): lambda x: multi_head_attention(x, num_heads=8),
            (40, 3, 0): lambda x: dropout(x[0], rate=0.5),
            (40, 4, 0): batch_norm,

            # Квантовые операции (50) с шумом через Qiskit
            (50, 0, 0): lambda x: self._quantum_operation(x[0], "hadamard"),
            (50, 0, 1): lambda x: self._quantum_operation(x[0], "pauli_x"),
            (50, 1, 0): lambda x: self._quantum_operation([x[0], x[1]], "cnot"),
            (50, 2, 0): lambda x: self._quantum_operation(x[0], "measure"),
            (50, 3, 0): lambda x: self._quantum_operation(x[0], "superposition"),
            (50, 4, 0): lambda x: self._quantum_operation([x[0], x[1]], "entanglement"),
        }

    def _quantum_operation(self, data, op_type):
        """Выполнение квантовых операций через Qiskit с шумом."""
        if isinstance(data, list):
            num_qubits = len(data)
            initial_state = np.array(data, dtype=np.complex128).flatten()
        else:
            num_qubits = 1
            initial_state = np.array([data, 0], dtype=np.complex128) if np.isscalar(data) else data

        # Нормализация начального состояния
        initial_state = initial_state / np.linalg.norm(initial_state)

        # Создаём квантовую цепь
        qc = QuantumCircuit(num_qubits)
        qc.initialize(initial_state, range(num_qubits))

        # Применяем операцию
        if op_type == "hadamard":
            qc.h(0)
        elif op_type == "pauli_x":
            qc.x(0)
        elif op_type == "cnot" and num_qubits >= 2:
            qc.cx(0, 1)
        elif op_type == "measure":
            qc.measure_all()
        elif op_type == "superposition":
            qc.h(0)
        elif op_type == "entanglement" and num_qubits >= 2:
            qc.h(0)
            qc.cx(0, 1)

        # Добавляем квантовый шум (деполяризация)
        from qiskit.providers.aer.noise import NoiseModel, depolarizing_error
        noise_model = NoiseModel()
        error = depolarizing_error(0.05, num_qubits)  # 5% шум
        noise_model.add_all_qubit_quantum_error(error, ['h', 'x', 'cx'])

        # Симуляция
        simulator = Aer.get_backend('statevector_simulator')
        job = execute(qc, simulator, noise_model=noise_model)
        result = job.result().get_statevector()
        return np.array(result, dtype=np.complex128)

    def _apply_quantum_ops(self, op, data):
        """Устаревший метод, теперь используется _quantum_operation."""
        return data

    def _apply_neural_ops(self, op, data):
        """Устаревший метод, теперь операции в self.core."""
        return data

    def _next_coords(self):
        coords = max([key[1][0] for key in self.space.keys()] + [self.max_coord]) + 1
        self.max_coord = coords
        return [coords, coords, coords]

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
            if isinstance(result, (int, float, complex)):
                data = np.array([complex(result)] + [0] * (input_dim - 1), dtype=np.complex128)
            elif isinstance(result, list) and all(isinstance(x, (int, float, complex)) for x in result):
                flat = np.array(result, dtype=np.complex128).flatten()
                data = np.pad(flat, (0, max(0, input_dim - len(flat))), mode='constant')[:input_dim]
            else:
                data = np.array([0] * input_dim, dtype=np.complex128)
            train_data.append(np.real(data))  # Пока используем только вещественную часть

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
        if isinstance(result, (int, float, complex, np.number)):
            data = np.array([complex(result)] + [0] * (input_dim - 1), dtype=np.complex128)
        elif isinstance(result, np.ndarray):
            flat = result.flatten()
            data = np.pad(flat, (0, max(0, input_dim - len(flat))), mode='constant')[:input_dim]
        else:
            data = np.array([0] * input_dim, dtype=np.complex128)

        tensor_data = torch.tensor(np.real(data), dtype=torch.float32)  # Только вещественная часть
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
            cached_result = self.use_memory.retrieve(x)
            if cached_result is not None:
                print(f"Использована память для Reason: {x} -> {cached_result}")
                return cached_result
        
        if isinstance(x, (int, float, complex, np.number)):
            result = self._apply_rl_strategy(x)
        elif isinstance(x, list):
            result = self._evolve_program(x)
        elif isinstance(x, np.ndarray):
            result = self._optimize_tensor(x)
        else:
            result = self.evolution.evolve(x)

        if self.use_memory:
            self.use_memory.store(x, result, reward=self._calculate_reward(result, x))
        
        print(f"Reason result: {result}")
        return result

    def _apply_rl_strategy(self, x):
        """Обучение с подкреплением для числовых данных (заглушка)."""
        return x  # Пока без RL, можно добавить позже

    def _evolve_program(self, x):
        """Эволюция программ (заглушка)."""
        return x  # Пока без реализации

    def _optimize_tensor(self, x):
        """Оптимизация тензоров (заглушка)."""
        return x  # Пока без реализации

    def _calculate_reward(self, result, input_data):
        if isinstance(input_data, (int, float, complex)):
            return 1.0 if np.abs(result) > np.abs(input_data) else -1.0
        elif isinstance(input_data, np.ndarray):
            return -np.mean(np.abs(result - input_data) ** 2)
        return 0.0

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

    def _store_tensor_in_ipfs(self, tensor_data):
        """Сохранение тензора в IPFS."""
        try:
            res = self.ipfs_client.add(tensor_data.tobytes())
            return res['Hash']
        except Exception as e:
            print(f"Ошибка сохранения в IPFS: {e}")
            return None

    def _load_tensor_from_ipfs(self, ipfs_hash, shape, dtype=np.complex128):
        """Загрузка тензора из IPFS."""
        try:
            data = self.ipfs_client.cat(ipfs_hash)
            return np.frombuffer(data, dtype=dtype).reshape(shape)
        except Exception as e:
            print(f"Ошибка загрузки из IPFS: {e}")
            return None

    def _save_model_metadata(self, model_name, ipfs_hash):
        """Сохранение метаданных модели в veectordb."""
        model_metadata = {
            "name": model_name,
            "ipfs_hash": ipfs_hash,
            "location": "ipfs",
            "timestamp": time.time()
        }
        self.db.insert_model(model_name, model_metadata)
        print(f"Метаданные модели сохранены: {model_name} -> {ipfs_hash}")

    def load_model(self, model_name):
        """Загрузка метаданных модели."""
        model_metadata = self.db.get_model_metadata(model_name)
        if model_metadata:
            return model_metadata
        else:
            print(f"Модель {model_name} не найдена в базе данных.")
            return None

    def save_tensor(self, tensor, tensor_id, use_ipfs=True):
        """Сохранение тензора в IPFS или локально."""
        if use_ipfs:
            ipfs_hash = self._store_tensor_in_ipfs(tensor)
            if ipfs_hash:
                tensor_metadata = {
                    "tensor_id": tensor_id,
                    "ipfs_hash": ipfs_hash,
                    "shape": tensor.shape,
                    "dtype": str(tensor.dtype),
                    "location": "ipfs",
                    "timestamp": time.time()
                }
                self.db.insert_tensor(tensor_id, tensor_metadata)
                print(f"Тензор {tensor_id} сохранён в IPFS: {ipfs_hash}")
                return ipfs_hash
            else:
                print(f"Не удалось сохранить тензор {tensor_id} в IPFS.")
                return None
        else:
            tensor_path = os.path.join(self.tensors_dir, f"{tensor_id}.npy")
            np.save(tensor_path, tensor)
            tensor_metadata = {
                "tensor_id": tensor_id,
                "path": tensor_path,
                "shape": tensor.shape,
                "dtype": str(tensor.dtype),
                "location": "local",
                "timestamp": time.time()
            }
            self.db.insert_tensor(tensor_id, tensor_metadata)
            print(f"Тензор {tensor_id} сохранён локально: {tensor_path}")
            return tensor_path

    def load_tensor(self, tensor_id):
        """Загрузка тензора из IPFS или локального хранилища."""
        tensor_metadata = self.db.get_tensor_metadata(tensor_id)
        if not tensor_metadata:
            print(f"Тензор {tensor_id} не найден в базе данных.")
            return None

        if tensor_metadata["location"] == "ipfs":
            ipfs_hash = tensor_metadata["ipfs_hash"]
            shape = tensor_metadata["shape"]
            dtype = np.dtype(tensor_metadata["dtype"])
            tensor_data = self._load_tensor_from_ipfs(ipfs_hash, shape, dtype)
            if tensor_data is not None:
                print(f"Тензор {tensor_id} загружен из IPFS: {ipfs_hash}")
                return tensor_data
            else:
                print(f"Не удалось загрузить тензор {tensor_id} из IPFS.")
                return None
        elif tensor_metadata["location"] == "local":
            tensor_path = tensor_metadata["path"]
            try:
                tensor_data = np.load(tensor_path)
                print(f"Тензор {tensor_id} загружен локально: {tensor_path}")
                return tensor_data
            except Exception as e:
                print(f"Ошибка загрузки тензора из локального файла: {e}")
                return None
        else:
            print(f"Неизвестное местоположение тензора: {tensor_metadata['location']}")
            return None

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
                data = np.array(data, dtype=np.complex128)
            if self.dropout_rate > 0 and op != [59, 0, 0]:
                data = self._dropout(data)

            result = op_func(data)

        tensor_id = f"tensor_result_{time.time()}_{random.randint(1000, 9999)}"
        self.save_tensor(result, tensor_id, use_ipfs=True)
        metadata = {"tensor": tensor, "coords": (data_layer, data_coords), "tensor_id": tensor_id}

        if self.use_neural_storage and self.neural_model:
            self._store_in_neural(result, tensor_id)

        if self.p2p_node:
            sync_data = np.abs(result) if np.iscomplexobj(result) else result  # Только вещественная часть для синхронизации
            self.p2p_node.sync_tensor(sync_data, metadata)

        self.space[(tuple(data_layer), tuple(data_coords))] = tensor_id

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
        tensor_id = f"tensor_{time.time()}_{random.randint(1000, 9999)}"
        self.save_tensor(tensor[0][2], tensor_id, use_ipfs=True)  # Сохраняем только данные
        self.space[(tuple(layer), coords)] = tensor_id

    def evolve_tensor(self, tensor):
        return self.evolution.log_evolution(tensor, self)

    def generate_program_tensor(self, prompt, max_steps=5):
        return self.model_manager.generate_program_tensor(prompt, max_steps)

    def share_program(self, program_tensors):
        return self.model_manager.share_program(program_tensors)

    def improve_program(self, program_tensors, feedback_data, iterations=3):
        return self.model_manager.improve_program(program_tensors, feedback_data, iterations)

    def execute_program(self, program_tensors, input_data=None):
        return self.model_manager.execute_program(program_tensors, input_data)


if __name__ == "__main__":
    # Пример использования
    p2p_node = P2PNode("localhost", 5000, use_ipfs=True)
    p2p_node.start()
    veector = Veector(p2p_node=p2p_node, use_memory=True)
    
    # Тест квантовой операции (Hadamard)
    tensor = create_tensor([0], [0, 0, 0], [1, 0], 2, op=[50, 0, 0])
    result = veector.compute(tensor)
    print(f"Результат квантовой операции Hadamard: {result}")