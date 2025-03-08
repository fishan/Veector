import numpy as np
import torch
from transformers import AutoModelForCausalLM
from ipfshttpclient import connect
from tensors import create_tensor

class ModelManager:
    def __init__(self, veector, ipfs_enabled=True, p2p_node=None):
        self.veector = veector
        self.ipfs_enabled = ipfs_enabled
        self.p2p_node = p2p_node
        self.model_space = {}
        self.tensor_cache = {}

    def add_model(self, model_name, model_path):
        """
        Разделяет модель на тензоры и добавляет их в пространство.
        """
        print(f"Добавление модели: {model_name}")
        model = AutoModelForCausalLM.from_pretrained(model_path)
        state_dict = model.state_dict()

        layer_idx = 0
        for name, tensor in state_dict.items():
            tensor_np = tensor.cpu().numpy()
            coords = self.veector._next_coords()
            tensor_info = self._store_tensor(model_name, name, tensor_np, layer_idx, coords)
            self.model_space[(model_name, layer_idx, tuple(coords))] = tensor_info
            # Создаем тензор Veector-формата для каждого слоя
            veector_tensor = create_tensor(
                [layer_idx], coords, tensor_np, tensor_np.size,
                op=[50, 0, 0],  # Пример операции (matrix_multiply)
                metadata={"model_name": model_name, "layer_idx": layer_idx, "role": self._infer_tensor_role(name)}
            )
            self.veector.add_to_space(veector_tensor)
            layer_idx += 1

    def _store_tensor(self, model_name, tensor_name, tensor_np, layer_idx, coords):
        metadata = {
            "model_name": model_name,
            "tensor_name": tensor_name,
            "role": self._infer_tensor_role(tensor_name),
            "shape": tensor_np.shape,
            "dependencies": self._infer_dependencies(tensor_name, layer_idx),
        }

        if self.ipfs_enabled:
            ipfs_hash = self._store_in_ipfs(tensor_np)
            metadata["ipfs_hash"] = ipfs_hash
            tensor_data = None
        else:
            tensor_data = tensor_np

        tensor_info = {
            "data": tensor_data,
            "metadata": metadata
        }

        if self.p2p_node:
            self.p2p_node.sync_tensor(tensor_np, metadata)

        return tensor_info

    def _store_in_ipfs(self, tensor_np):
        client = connect()
        ipfs_hash = client.add_bytes(tensor_np.tobytes())
        return ipfs_hash

    def _load_from_ipfs(self, ipfs_hash, shape):
        client = connect()
        tensor_data = client.cat(ipfs_hash)
        return np.frombuffer(tensor_data, dtype=np.float32).reshape(shape)

    def _infer_tensor_role(self, tensor_name):
        if "self_attn" in tensor_name:
            return "attention_weights"
        elif "layer_norm" in tensor_name:
            return "layer_normalization"
        elif "mlp" in tensor_name or "dense" in tensor_name:
            return "feed_forward"
        else:
            return "unknown"

    def _infer_dependencies(self, tensor_name, layer_idx):
        dependencies = []
        if "self_attn" in tensor_name:
            dependencies.append((layer_idx, "layer_norm"))
        return dependencies

    def get_tensor(self, model_name, layer_idx, coords):
        key = (model_name, layer_idx, tuple(coords))
        if key in self.tensor_cache:
            return self.tensor_cache[key]

        tensor_info = self.model_space.get(key)
        if tensor_info is None:
            raise ValueError(f"Tensor at {key} not found")

        if tensor_info["data"] is None and "ipfs_hash" in tensor_info["metadata"]:
            tensor_np = self._load_from_ipfs(tensor_info["metadata"]["ipfs_hash"], tensor_info["metadata"]["shape"])
            tensor_info["data"] = tensor_np

        self.tensor_cache[key] = tensor_info["data"]
        return tensor_info["data"]

    def perform_inference(self, model_name, input_tensor, max_layers=32):
        current_tensor = input_tensor
        for layer_idx in range(max_layers):
            coords = None
            for k in self.model_space.keys():
                if k[0] == model_name and k[1] == layer_idx:
                    coords = k[2]
                    break
            if coords is None:
                continue

            tensor_info = self.model_space.get((model_name, layer_idx, coords))
            if tensor_info is None:
                continue

            role = tensor_info["metadata"]["role"]
            tensor_data = self.get_tensor(model_name, layer_idx, coords)

            if role == "attention_weights":
                current_tensor = self.veector.compute(
                    create_tensor([layer_idx], coords, [current_tensor, current_tensor, current_tensor], 3, op=[70, 0, 0])
                )
            elif role == "feed_forward":
                current_tensor = self.veector.compute(
                    create_tensor([layer_idx], coords, current_tensor, 1, op=[50, 0, 0])
                )
            elif role == "layer_normalization":
                current_tensor = self.veector.compute(
                    create_tensor([layer_idx], coords, current_tensor, 1, op=[71, 0, 0])
                )
        return current_tensor