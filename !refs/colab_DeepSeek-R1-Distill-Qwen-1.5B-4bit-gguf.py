# Google Colab: save_whole_model_to_vib.py
# !pip install torch numpy gguf

import torch
import numpy as np
from gguf import GGUFReader
import os
import json

# Скачиваем модель (замени путь на доступный в Colab)
gguf_path = "/content/DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf"
# Если модели нет, нужно загрузить её, например, с Hugging Face или вручную
# !wget <URL-to-GGUF-file> -O {gguf_path}

output_dir = "/content/blocks/DeepSeek-R1-Distill-Qwen-1.5B"
os.makedirs(output_dir, exist_ok=True)

reader = GGUFReader(gguf_path)
metadata = {}

def decode_tensor(tensor):
    shape = tensor.shape
    n_elements = shape[0] * shape[1]
    if tensor.tensor_type == 12:  # Q4_K
        n_groups = n_elements // 32
        q_bytes = n_elements // 2
        meta_bytes = n_groups * 4
        total_size = q_bytes + meta_bytes
        if total_size > len(tensor.data):
            raise ValueError(f"Недостаточно данных для декодирования {tensor.name}")
        q_data = np.frombuffer(tensor.data[:q_bytes], dtype=np.uint8)
        meta_data = np.frombuffer(tensor.data[q_bytes:q_bytes + meta_bytes], dtype=np.float16)
        d_values = meta_data[0::2]
        m_values = meta_data[1::2]
        q_values = np.zeros(n_elements, dtype=np.float16)
        for k in range(len(q_data)):
            q_values[2*k] = (q_data[k] & 0x0F)
            if 2*k + 1 < n_elements:
                q_values[2*k + 1] = (q_data[k] >> 4)
        block_data = np.zeros(n_elements, dtype=np.float16)
        for g in range(n_groups):
            d = d_values[g]
            m = m_values[g]
            for k in range(32):
                idx = g * 32 + k
                if idx < n_elements:
                    block_data[idx] = d * (q_values[idx] - m)
        return block_data.reshape(shape)
    else:
        itemsize = 2 if tensor.tensor_type == 1 else 4
        return np.frombuffer(tensor.data, dtype=np.float16 if tensor.tensor_type == 1 else np.float32).reshape(shape)

# Сохранение всей модели в один .vib
for tensor in reader.tensors:
    if len(tensor.shape) == 2:
        block = decode_tensor(tensor)
        block_name = tensor.name
        block_path = f"{output_dir}/{block_name}.vib"
        torch.save(torch.from_numpy(block), block_path)
        metadata[block_name] = {
            "tensor_name": tensor.name,
            "shape": list(tensor.shape),
            "offset": 0,
            "path": block_path,
            "block_type": "embed" if "token_embd" in tensor.name else ("output" if "output" in tensor.name else "layer"),
            "size_mb": block.nbytes / (1024 * 1024),
            "dtype": "float16" if tensor.tensor_type in [1, 12] else "float32"
        }
        print(f"Сохранён {block_name}: {tensor.shape}, {block.nbytes / (1024 * 1024):.2f} MB")

with open(f"{output_dir}/metadata.json", "w") as f:
    json.dump(metadata, f, indent=4)

# Тест загрузки
loaded_tensor = torch.load(f"{output_dir}/token_embd.weight.vib", weights_only=True)
print(f"Загружен тензор эмбеддингов: {loaded_tensor.shape}")