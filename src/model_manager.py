# /workspaces/Veector/device/src/model_manager.py
import os
import torch
import torch.nn.functional as F
import numpy as np
from ipfshttpclient import connect
from pathlib import Path
from virtual_space import VirtualSpace
from qiskit import QuantumCircuit
from utils import parse_block_name  # Предполагается, что этот модуль существует

class ModelManager:
    def __init__(self, veector, block_size=(1024, 1024), ipfs_enabled=True, model_dir="../data/models"):
        """
        Менеджер моделей для работы с блочно-матричной архитектурой и квантовыми цепями.
        :param veector: Экземпляр ядра Veector.
        :param block_size: Размер блока матрицы (высота, ширина) — не используется, размеры из метаданных.
        :param ipfs_enabled: Включ-lact IPFS-хранилище.
        :param model_dir: Директория для локальных данных.
        """
        self.veector = veector
        self.ipfs_enabled = ipfs_enabled
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.virtual_space = VirtualSpace(veector, use_ipfs=ipfs_enabled, model_manager=self)
        self.quantum_circuits = {}
        self.p2p_node = veector.p2p_node if ipfs_enabled and veector.p2p_node else None

    def load_pre_split_model(self, model_name, tensor_dir, vocab_size=None, hidden_size=None, num_layers=None):
        """
        Загружает модель, предварительно разделённую на блоки, из директории.
        :param model_name: Название модели.
        :param tensor_dir: Путь к директории с блоками модели.
        :param vocab_size: Размер словаря (из config.json).
        :param hidden_size: Размер скрытого слоя (из config.json).
        :param num_layers: Количество слоёв (из config.json).
        """
        print(f"Проверка загрузки модели {model_name} из {tensor_dir}")
        block_files = list(Path(tensor_dir).glob(f"{model_name}_*_block*.pt"))  # Ищем файлы вида _blockN.pt
        print(f"Найдено {len(block_files)} файлов блоков в {tensor_dir}:")
        for block_file in block_files[:10]:  # Ограничим вывод первыми 10 файлами
            print(f" - {block_file.name}")
        if len(block_files) > 10:
            print(f" ... и еще {len(block_files) - 10} файлов")
        
        if not block_files:
            print(f"Ошибка: Файлы для модели {model_name} не найдены в {tensor_dir}")
            raise ValueError(f"Модель {model_name} не найдена")

        # Проверяем наличие метаданных
        metadata_path = os.path.join(tensor_dir, f"{model_name}_metadata.json")
        if not os.path.exists(metadata_path):
            print(f"Ошибка: Файл метаданных {metadata_path} не найден.")
            raise FileNotFoundError(f"Метаданные для {model_name} отсутствуют")

        # Используем переданные параметры из config.json
        if vocab_size is None or hidden_size is None or num_layers is None:
            raise ValueError("Все параметры (vocab_size, hidden_size, num_layers) должны быть переданы из config.json")

        # Переключаем VirtualSpace на модель
        self.virtual_space.switch_model(model_name, vocab_size, hidden_size, num_layers)
        print(f"Модель {model_name} загружена из {tensor_dir} с {len(block_files)} блоками")

    def perform_inference(self, model_name, input_data):
        """
        Выполняет инференс для указанной модели.
        :param model_name: Название модели.
        :param input_data: Входные данные (numpy массив или список).
        :return: Результат инференса (numpy массив).
        """
        if model_name not in self.virtual_space.matrix_models:
            raise ValueError(f"Модель {model_name} не загружена")
        input_tensor = torch.from_numpy(input_data).to(self.virtual_space.matrix_models[model_name].device)
        output = self.virtual_space.perform_inference(input_tensor)
        return output.detach().cpu().numpy() 

    def add_quantum_circuit(self, model_name, circuit):
        """Добавляет квантовую цепь для модели."""
        if not isinstance(circuit, QuantumCircuit):
            raise ValueError("circuit должен быть объектом QuantumCircuit")
        self.quantum_circuits[model_name] = circuit
        print(f"Квантовая цепь добавлена для модели {model_name}")

    def execute_quantum_circuit(self, model_name, input_state=None):
        """Выполняет квантовую цепь для модели."""
        if model_name not in self.quantum_circuits:
            raise ValueError(f"Квантовая цепь для {model_name} не найдена")
        
        from qiskit import execute
        from qiskit.providers.aer import Aer
        circuit = self.quantum_circuits[model_name]
        num_qubits = circuit.num_qubits

        if input_state is not None:
            input_state = np.array(input_state, dtype=np.complex128)
            if input_state.size != 2 ** num_qubits:
                raise ValueError(f"Размер входного состояния {input_state.size} не соответствует {2 ** num_qubits}")
            circuit.initialize(input_state / np.linalg.norm(input_state), range(num_qubits))

        simulator = Aer.get_backend('statevector_simulator')
        job = execute(circuit, simulator)
        result = job.result().get_statevector()
        return np.array(result, dtype=np.complex128)

if __name__ == "__main__":
    from core import Veector
    veector = Veector(use_memory=False, ipfs_enabled=False)
    manager = ModelManager(veector, ipfs_enabled=False)

    # Загрузка модели
    manager.load_pre_split_model(
        "DeepSeek-R1-Distill-Qwen-1.5B",
        "/workspaces/Veector/data/blocks/DeepSeek-R1-Distill-Qwen-1.5B",
        vocab_size=151936,
        hidden_size=1536,
        num_layers=28
    )

    # Тест инференса
    vocab_size = 151936
    max_sequence_length = 512
    batch_size = 1
    input_data = np.random.randint(0, vocab_size, (batch_size, max_sequence_length), dtype=np.int32)
    output = manager.perform_inference("DeepSeek-R1-Distill-Qwen-1.5B", input_data)
    print(f"Результат инференса: {output.shape}")

    # Тест квантовой цепи
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    manager.add_quantum_circuit("quantum_test", qc)
    result = manager.execute_quantum_circuit("quantum_test", input_state=[1, 0, 0, 0])
    print(f"Результат квантовой цепи: {result}")