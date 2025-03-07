import numpy as np
from core import Veector
from threading import Thread

class VirtualSpace:
    def __init__(self, veector):
        self.veector = veector
        self.space = self.veector.space

    def add_tensor(self, tensor):
        self.veector.add_to_space(tensor)

    def get_tensor(self, layer, coords):
        doc_id = self.space.get((tuple(layer), tuple(coords)))
        if doc_id:
            doc = self.veector.db.get(doc_id)
            return doc["data"] if doc else None
        return None

    def execute(self, start_tensor):
        results = []
        result = self.veector.compute(start_tensor)
        results.append(result)
        
        next_coords = start_tensor[4] if len(start_tensor) > 4 else []
        if next_coords and all(isinstance(path, list) and len(path) == 2 for path in next_coords):
            threads = []
            thread_results = []
            def compute_path(layer, coords, res_list):
                tensor = self.get_tensor(layer, coords)
                if tensor and isinstance(tensor, list):
                    res = self.veector.compute(tensor)
                    if res not in res_list:  # Убираем дубли
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

    def explore_generated(self):
        """Рекурсивное выполнение всех сгенерированных тензоров"""
        generated_tensors = self.veector.db.find_by_type("tensor")
        results = {}
        
        def compute_recursive(tensor, visited=None):
            if visited is None:
                visited = set()
            if not isinstance(tensor, list) or len(tensor) < 4:
                return tensor  # Если не тензор, возвращаем как есть
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
                    next_tensor = self.get_tensor(next_layer, next_coords_tuple)
                    if next_tensor and isinstance(next_tensor, list):
                        compute_recursive(next_tensor, visited)
            return result
        
        for tensor_doc in generated_tensors:
            tensor = tensor_doc["data"]
            compute_recursive(tensor)
        return results

    def analyze_deepseek(self, weights):
        for i, w in enumerate(weights):
            tensor = [
                [[0], [i, 0, 0], w, len(w)],
                [[0], [i, 0, 0], [0, 1, 0], 1],
                [1, 0, 0],
                [0, 1, 0],
                [[1], [i+1, 0, 0]]
            ]
            self.add_tensor(tensor)

if __name__ == "__main__":
    v = Veector(use_neural_storage=True)
    vs = VirtualSpace(v)
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
    vs.add_tensor(tensor1)
    vs.add_tensor(tensor2)
    vs.add_tensor(tensor3)
    vs.add_tensor(tensor4)
    vs.add_tensor(tensor5)
    results = vs.execute(tensor1)
    print(f"Результаты (sin): {results}")
    results = vs.execute(tensor2)
    print(f"Результаты (dot): {results}")
    results = vs.execute(tensor3)
    print(f"Результаты (matrix): {results}")
    results = vs.execute(tensor4)
    print(f"Результаты (parallel): {results}")
    results = vs.execute(tensor5)
    print(f"Результаты (reason): {results}")
    
    tensors = v.db.find_by_type("tensor")
    print(f"Найдено тензоров: {len(tensors)}")
    results = v.db.find_by_type("tensor_result")
    print(f"Найдено результатов: {len(results)}")
    generated_results = vs.explore_generated()
    print(f"Результаты сгенерированных тензоров: {generated_results}")


    # Тест _retrieve_from_neural с MSE
    for doc in v.db.find_by_type("tensor_result"):
        doc_id = doc["id"]
        retrieved = v._retrieve_from_neural(doc_id)
        if retrieved is not None:
            original = doc["data"]
            if isinstance(original, (int, float, np.number)):
                orig_vector = np.array([float(original)] + [0] * (v._get_max_input_dim() - 1))
            elif isinstance(original, np.ndarray):
                flat = original.flatten()
                orig_vector = np.pad(flat, (0, max(0, v._get_max_input_dim() - len(flat))), mode='constant')
            else:
                orig_vector = np.array([0] * v._get_max_input_dim())
            mse = np.mean((retrieved - orig_vector) ** 2)
            print(f"Извлечено из нейросети {doc_id}: {retrieved[:5]}... (оригинал: {original}, MSE: {mse:.6f})")