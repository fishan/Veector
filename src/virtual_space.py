import numpy as np
from core import Veector
from model_manager import ModelManager
from threading import Thread
from tensors import create_tensor, validate_tensor, reshape_tensor, get_tensor_metadata

class VirtualSpace:
    def __init__(self, veector, model_manager=None, use_ipfs=False):
        self.veector = veector
        self.space = self.veector.space
        self.db = self.veector.db
        self.model_manager = model_manager
        self.use_ipfs = use_ipfs
        self.tensor_metadata = {}

    def add_tensor(self, tensor, model_name=None, layer_idx=None, coords=None, role=None, dependencies=None):
        if model_name and layer_idx is not None and coords is not None:
            if self.model_manager is None:
                raise ValueError("ModelManager not initialized")
            tensor_name = f"{model_name}_layer{layer_idx}_coords{coords}"
            tensor_info = self.model_manager._store_tensor(model_name, tensor_name, tensor, layer_idx, coords)
            self.model_manager.model_space[(model_name, layer_idx, tuple(coords))] = tensor_info
            
            self.tensor_metadata[(model_name, layer_idx, tuple(coords))] = {
                "role": role or tensor_info["metadata"]["role"],
                "dependencies": dependencies or tensor_info["metadata"]["dependencies"],
                "shape": tensor.shape
            }
        else:
            self.veector.add_to_space(tensor)

    def get_tensor(self, layer, coords, model_name=None):
        if model_name and self.model_manager:
            cache_key = (model_name, layer, tuple(coords))
            if cache_key in self.veector.cache:
                return self.veector.cache[cache_key]
            tensor = self.model_manager.get_tensor(model_name, layer, coords)
            self.veector.cache[cache_key] = tensor
            return tensor
        doc_id = self.space.get((tuple(layer), tuple(coords)))
        if doc_id:
            doc = self.db.get(doc_id)
            return doc["data"] if doc else None
        return None

    def get_tensor_metadata(self, layer, coords, model_name=None):
        if model_name:
            return self.tensor_metadata.get((model_name, layer, tuple(coords)), {})
        return {}

    def execute(self, start_tensor, model_name=None):
        results = []
        result = self.veector.compute(start_tensor)
        results.append(result)

        next_coords = start_tensor[4] if len(start_tensor) > 4 else []
        if next_coords and all(isinstance(path, list) and len(path) == 2 for path in next_coords):
            threads = []
            thread_results = []

            def compute_path(layer, coords, res_list):
                tensor = self.get_tensor(layer, coords, model_name)
                if tensor and isinstance(tensor, list):
                    metadata = self.get_tensor_metadata(layer, coords, model_name)
                    if "dependencies" in metadata:
                        for dep_layer, dep_coords in metadata["dependencies"]:
                            dep_tensor = self.get_tensor(dep_layer, dep_coords, model_name)
                            if dep_tensor is None:
                                raise ValueError(f"Dependency {dep_layer, dep_coords} not found")
                    res = self.veector.compute(tensor)
                    if res not in res_list:
                        res_list.append(res)

            for path in next_coords:
                layer, coords = path
                t = Thread(target=compute_path, args=(layer, coords, thread_results))
                threads.append(t)
                t.start()

            for t in threads:
                t.join()

            results.extend(thread_results)
        return results

    def explore_generated(self, model_name=None):
        if model_name and self.model_manager:
            results = {}
            for (m_name, layer, coords), tensor_info in self.model_manager.model_space.items():
                if m_name == model_name:
                    tensor = self.get_tensor(layer, coords, model_name)
                    if tensor and isinstance(tensor, list):
                        result = self.veector.compute(tensor)
                        results[(tuple(layer), coords)] = result
            return results

        generated_tensors = self.db.find_by_type("tensor")
        results = {}

        def compute_recursive(tensor, visited=None):
            if visited is None:
                visited = set()

            if not isinstance(tensor, list) or len(tensor) < 4:
                return tensor

            layer, coords = tensor[0][0], tuple(tensor[0][1])
            key = (tuple(layer), coords)

            if key in visited:
                return None

            visited.add(key)
            result = self.veector.compute(tensor)
            results[key] = result
            next_coords = tensor[4] if len(tensor) > 4 else []
            if next_coords and all(isinstance(path, list) and len(path) == 2 for path in next_coords):
                for path in next_coords:
                    next_layer, next_coords_tuple = path
                    next_tensor = self.get_tensor(next_layer, next_coords_tuple, model_name)
                    if next_tensor and isinstance(next_tensor, list):
                        compute_recursive(next_tensor, visited)
            return result

        for tensor_doc in generated_tensors:
            tensor = tensor_doc["data"]
            compute_recursive(tensor)
        return results

    def analyze_deepseek(self, model_name, weights):
        for i, w in enumerate(weights):
            tensor = create_tensor([0], [i, 0, 0], w, len(w), op=[0, 1, 0], metadata={"model_name": model_name, "layer_idx": i})
            role = "unknown"
            dependencies = []
            self.add_tensor(tensor, model_name, i, [i, 0, 0], role, dependencies)

    def create_hybrid_space(self, hybrid_name, model_configs):
        for role, (model_name, layer_idx, coords) in model_configs.items():
            tensor = self.get_tensor(layer_idx, coords, model_name)
            if tensor is None:
                raise ValueError(f"Tensor for {role} not found in {model_name}")
            metadata = self.get_tensor_metadata(layer_idx, coords, model_name)
            self.add_tensor(tensor, hybrid_name, layer_idx, coords, role, metadata.get("dependencies", []))


if __name__ == "__main__":
    from sync import P2PNode

    # Инициализация с P2PNode и ModelManager
    p2p_node = P2PNode("localhost", 5000, use_ipfs=True)
    p2p_node.start()
    v = Veector(use_neural_storage=True, cache_size=500, dropout_rate=0.2, use_memory=True, p2p_node=p2p_node)
    model_manager = ModelManager(v, ipfs_enabled=True, p2p_node=p2p_node)
    vs = VirtualSpace(v, model_manager, use_ipfs=True)
    

    tensor1 = [
        [[0], [0, 0, 0], [np.pi / 2], 1],
        [[0], [0, 0, 0], [2, 1, 0], 1],
        [1, 0, 0],
        [0, 1, 0],
        [[0], [1, 0, 0]]
    ]

    tensor2 = [
        [[0], [1, 0, 0], [[1, 2], [3, 4]], 2],
        [[0], [1, 0, 0], [4, 1, 0], 1],
        [1, 0, 0],
        [0, 1, 0],
        [[0], [2, 0, 0]]
    ]

    tensor3 = [
        [[0], [2, 0, 0], [[[1, 2], [3, 4]], [[5, 6], [7, 8]]], 2],
        [[0], [2, 0, 0], [1, 3, 0], 1],
        [1, 0, 0],
        [0, 1, 0],
        [[0], [3, 0, 0]]
    ]

    tensor4 = [
        [[0], [3, 0, 0], [1], 1],
        [[0], [3, 0, 0], [2, 1, 0], 1],
        [1, 0, 0],
        [0, 1, 0],
        [[[0], [4, 0, 0]], [[0], [5, 0, 0]]]
    ]

    tensor5 = [
        [[0], [4, 0, 0], [2], 1],
        [[0], [4, 0, 0], [9, 0, 0], 1],
        [1, 0, 0],
        [0, 1, 0],
        []
    ]
    tensor6 = [  # Тест memory
        [[0], [5, 0, 0], [5], 1],
        [[0], [5, 0, 0], [9, 0, 0], 1],
        [1, 0, 0],
        [0, 1, 0],
        []
    ]
    operations = [
    (1, 0, 0),  # Сложение
    (0, 1, 0),  # Умножение
    (1, 1, 1)   # Вычитание
    ]
    data_list = [
        [5, 3],     # Данные для сложения
        [2, 4],     # Данные для умножения
        [10, 7]     # Данные для вычитания
    ]
    vs.add_tensor(tensor1)
    vs.add_tensor(tensor2)
    vs.add_tensor(tensor3)
    vs.add_tensor(tensor4)
    vs.add_tensor(tensor5)
    vs.add_tensor(tensor6)

    results = vs.execute(tensor1)
    print(f"Результаты (sin): {results}")

    results = vs.execute(tensor2)
    print(f"Результаты (dot): {results}")

    results = vs.execute(tensor3)
    print(f"Результаты (matrix): {results}")

    results = vs.execute(tensor4)
    print(f"Результаты (две ветви): {results}")

    results = vs.execute(tensor5)
    print(f"Результаты (reason): {results}")
    results = vs.execute(tensor6)  # Тест Memory
    print(f"Результаты (reason + memory): {results}")

    print("\nТест эволюции:")  # Тест эволюции
    evolved_result = vs.veector.evolve_tensor(tensor1)
    print(f"Эволюционированный результат: {evolved_result}")

    results = v.parallel_compute(operations, data_list)
    print("Результаты параллельных вычислений:", results)

    # Дополнительный тест с ModelManager и DeepSeek
    weights_deepseek = [np.random.rand(512, 512) for _ in range(2)]
    vs.analyze_deepseek("deepseek-7b", weights_deepseek)
    deepseek_tensor = create_tensor([0], [0, 0, 0], np.random.rand(1, 512), 512, op=[70, 0, 0], metadata={"model_name": "deepseek-7b"})
    results = vs.execute(deepseek_tensor, model_name="deepseek-7b")
    print(f"Результаты DeepSeek (self-attention): {results}")


