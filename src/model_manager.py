import os
import torch
import torch.nn.functional as F
import numpy as np
from ipfshttpclient import connect
from pathlib import Path
from src.virtual_space import VirtualSpace
from src.tensors import create_tensor
from src.sync import P2PNode
from qiskit import QuantumCircuit  # Добавлено: для управления квантовыми цепями

class ModelManager:
    def __init__(self, veector, block_size=(1024, 1024), ipfs_enabled=True, model_dir="data/models"):
        """
        Менеджер моделей для работы с блочно-матричной архитектурой и квантовыми цепями.
        :param veector: Экземпляр ядра Veector.
        :param block_size: Размер блока матрицы (высота, ширина).
        :param ipfs_enabled: Включить IPFS-хранилище.
        :param model_dir: Директория для локальных данных.
        """
        self.veector = veector
        self.block_size = block_size
        self.ipfs_enabled = ipfs_enabled
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.model_space = {}  # { (model_name, layer, block_coords): tensor_info }
        self.tensor_metadata = {}  # Хранение метаданных тензоров
        self.quantum_circuits = {}  # { model_name: QuantumCircuit } — для квантовых цепей
        self.p2p_node = veector.p2p_node if ipfs_enabled and veector.p2p_node else None
        self.virtual_space = VirtualSpace(veector, use_ipfs=ipfs_enabled)

    def load_pre_split_model(self, model_name, tensor_dir):
        """
        Загрузка предварительно разбитой модели.
        :param model_name: Название модели.
        :param tensor_dir: Путь к директории с блоками тензоров.
        """
        tensor_dir = Path(tensor_dir)
        self.virtual_space.load_blocks_model_into_matrix(model_name, tensor_dir)
        if self.ipfs_enabled and self.p2p_node:
            self.p2p_node.sync_model_blocks(model_name, tensor_dir)

        # Заполняем model_space и tensor_metadata
        for coords, block_info in self.virtual_space.virtual_matrix.matrices.get(model_name, {}).get("blocks", {}).items():
            key = (model_name, 0, coords)  # layer=0 для плоской структуры
            self.model_space[key] = {
                "tensor_id": block_info["hash"],
                "shape": self.block_size,
                "data": None  # Данные загружаются по запросу
            }
            self.tensor_metadata[key] = {
                "role": "flat_model_block",
                "dependencies": [],
                "shape": self.block_size,
                "tensor_id": block_info["hash"]
            }
        print(f"Модель {model_name} загружена из {tensor_dir} с {len(self.model_space)} блоками")

    def get_block(self, model_name, layer_idx, coords):
        """
        Получение блока из кэша, IPFS или локального хранилища.
        :param model_name: Название модели.
        :param layer_idx: Индекс слоя (0 для плоской структуры).
        :param coords: Координаты блока (row, col).
        :return: Тензор блока (torch.Tensor).
        """
        key = (model_name, layer_idx, coords)
        if key not in self.model_space:
            raise ValueError(f"Блок {key} не найден в model_space")

        if self.model_space[key]["data"] is None:
            tensor_id = self.model_space[key]["tensor_id"]
            if self.ipfs_enabled and self.p2p_node and tensor_id:
                block = self.p2p_node._load_from_ipfs(tensor_id, self.block_size, dtype="float16")
                self.model_space[key]["data"] = block.numpy() if block is not None else None
            else:
                block = self.virtual_space.virtual_matrix.load_block(model_name, coords)
                self.model_space[key]["data"] = block.numpy() if block is not None else None

        data = self.model_space[key]["data"]
        return torch.from_numpy(data) if data is not None else None

    def perform_inference(self, model_name, input_data):
        """
        Выполнение инференса через VirtualSpace.
        :param model_name: Название модели.
        :param input_data: Входные данные (np.ndarray).
        :return: Результат инференса (np.ndarray).
        """
        if model_name not in self.virtual_space.matrix_models:
            self.load_pre_split_model(model_name, "data/blocks")
        
        input_tensor = (torch.from_numpy(input_data).long() 
                        if input_data.dtype in (np.int64, np.int32) 
                        else torch.from_numpy(input_data).float())
        output = self.virtual_space.perform_inference(input_tensor)
        return output.cpu().numpy()

    def update_parameters(self, model_name, learning_rate=1e-4):
        """
        Обновление параметров после обратного распространения.
        :param model_name: Название модели.
        :param learning_rate: Скорость обучения.
        """
        for key in self.model_space:
            m_name, layer, coords = key
            if m_name != model_name:
                continue
            tensor_info = self.model_space[key]
            if tensor_info["data"] is None:
                self.get_block(model_name, layer, coords)
            data_tensor = torch.from_numpy(tensor_info["data"])
            if data_tensor.requires_grad and data_tensor.grad is not None:
                updated_data = data_tensor - learning_rate * data_tensor.grad
                tensor_info["data"] = updated_data.detach().numpy()
                if self.ipfs_enabled and self.p2p_node:
                    self.p2p_node.sync_tensor(updated_data, {
                        "tensor_id": tensor_info["tensor_id"],
                        "model_name": model_name,
                        "coords": coords
                    })

    def save_model(self, model_name, output_dir):
        """
        Сохранение модели в директорию.
        :param model_name: Название модели.
        :param output_dir: Путь к директории для сохранения.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for key in self.model_space:
            m_name, layer, coords = key
            if m_name != model_name:
                continue
            if self.model_space[key]["data"] is None:
                self.get_block(model_name, layer, coords)
            tensor = torch.from_numpy(self.model_space[key]["data"])
            filename = f"{model_name}_layer_{layer}_block_{coords[0]}_{coords[1]}.pt"
            torch.save(tensor, output_dir / filename)
        print(f"Модель {model_name} сохранена в {output_dir}")

    def get_num_layers(self, model_name):
        """
        Получение количества слоёв (для плоской структуры всегда 1).
        :param model_name: Название модели.
        :return: Число слоёв.
        """
        return 1 if any(m_name == model_name for m_name, _, _ in self.model_space) else 0

    # Методы для квантовых цепей
    def add_quantum_circuit(self, model_name, circuit):
        """
        Добавление квантовой цепи для модели.
        :param model_name: Название модели.
        :param circuit: Объект QuantumCircuit из Qiskit.
        """
        if not isinstance(circuit, QuantumCircuit):
            raise ValueError("circuit должен быть объектом QuantumCircuit")
        self.quantum_circuits[model_name] = circuit
        print(f"Квантовая цепь добавлена для модели {model_name}")

    def execute_quantum_circuit(self, model_name, input_state=None):
        """
        Выполнение квантовой цепи.
        :param model_name: Название модели.
        :param input_state: Начальное состояние (np.ndarray с комплексными числами).
        :return: Результат выполнения (np.ndarray).
        """
        if model_name not in self.quantum_circuits:
            raise ValueError(f"Квантовая цепь для {model_name} не найдена")
        
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

    # Методы для работы с программами (из Veector)
    def generate_program_tensor(self, prompt, max_steps=5):
        """
        Генерация программного тензора на основе текстового запроса.
        :param prompt: Текстовый запрос.
        :param max_steps: Максимальное число шагов.
        :return: Программный тензор.
        """
        # Простая заглушка: генерируем тензор с операцией вывода
        data = np.array([prompt], dtype=object)
        return create_tensor(
            layer=[0],
            coords=[0, 0, 0],
            data=data,
            length=1,
            op=[8, 0, 0],  # Операция вывода
            metadata={"prompt": prompt, "steps": max_steps}
        )

    def share_program(self, program_tensors):
        """
        Поделиться программным тензором через P2PNode.
        :param program_tensors: Список тензоров программы.
        :return: Хэш IPFS или None.
        """
        if not self.ipfs_enabled or not self.p2p_node:
            print("P2PNode не доступен для совместного использования")
            return None
        
        program_data = np.array([t[0][2] for t in program_tensors], dtype=object)
        tensor_id = f"program_{int(time.time())}"
        return self.p2p_node.sync_tensor(program_data, {"tensor_id": tensor_id, "type": "program"})

    def improve_program(self, program_tensors, feedback_data, iterations=3):
        """
        Улучшение программы на основе обратной связи.
        :param program_tensors: Список тензоров программы.
        :param feedback_data: Данные обратной связи (np.ndarray).
        :param iterations: Число итераций улучшения.
        :return: Улучшенные тензоры.
        """
        # Простая заглушка: возвращаем исходные тензоры
        print(f"Улучшение программы на основе feedback_data shape={feedback_data.shape} за {iterations} итераций")
        return program_tensors

    def execute_program(self, program_tensors, input_data=None):
        """
        Выполнение программы, заданной тензорами.
        :param program_tensors: Список тензоров программы.
        :param input_data: Входные данные (опционально).
        :return: Результат выполнения.
        """
        results = []
        for tensor in program_tensors:
            result = self.veector.compute(tensor)
            results.append(result)
        return results

if __name__ == "__main__":
    from src.core import Veector
    from src.sync import P2PNode
    import time

    # Тест
    p2p_node = P2PNode("localhost", 5000, use_ipfs=True)
    veector = Veector(p2p_node=p2p_node)
    manager = ModelManager(veector)

    # Загрузка модели
    manager.load_pre_split_model("DeepSeek-R1-Distill-Qwen-1.5B", "data/blocks")

    # Инференс
    input_data = np.random.randint(0, 32768, (1, 512))
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

    # Тест программы
    program_tensor = manager.generate_program_tensor("Привет, мир!", max_steps=3)
    manager.share_program([program_tensor])
    result = manager.execute_program([program_tensor])
    print(f"Результат программы: {result}")