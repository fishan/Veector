import os
from gguf import GGUFReader
from pathlib import Path
import json
import logging
from virtual_space import VirtualSpace
from core import Veector

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def split_gguf_by_tensors(gguf_path, block_dir, virtual_space, model_name="default_model"):
    gguf_path = Path(gguf_path)
    block_dir = Path(block_dir)
    block_dir.mkdir(parents=True, exist_ok=True)

    reader = GGUFReader(gguf_path)
    blocks = {}
    
    # Заголовок
    header_size = reader.header_size  # Уточнить
    with open(gguf_path, "rb") as f:
        header_data = f.read(header_size)
    header_path = block_dir / "header.vib"
    with open(header_path, "wb") as f:
        f.write(header_data)
    blocks["header"] = {
        "file": "header.vib",
        "offset": 0,
        "size": header_size,
        "model_name": model_name,
        "coords": (0, 0)
    }

    # Тензоры
    for idx, tensor in enumerate(reader.tensors):
        block_key = f"tensor_{tensor.name}"
        block_path = block_dir / f"{block_key}.vib"
        block_data = tensor.data
        
        with open(block_path, "wb") as f:
            f.write(block_data)
        
        block = torch.from_numpy(np.frombuffer(block_data, dtype=np.float16).reshape(tensor.shape))
        blocks[block_key] = {
            "file": block_path.name,
            "offset": tensor.offset,
            "size": len(block_data),
            "tensor_name": tensor.name,
            "shape": list(tensor.shape),
            "tensor_type": tensor.tensor_type,
            "model_name": model_name,
            "coords": (idx, 0)
        }
        virtual_space.sync_block(block_key, block, gguf_path)

    # Сохраняем метаданные
    metadata_path = block_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(blocks, f, indent=2)
    
    virtual_space.switch_model(gguf_path, blocks)
    logger.info(f"GGUF split into {len(reader.tensors) + 1} blocks (.vib) and synced to VirtualSpace")

if __name__ == "__main__":
    veector = Veector(use_memory=False, ipfs_enabled=True)
    virtual_space = VirtualSpace(veector, use_ipfs=True)
    gguf_path = "/workspaces/Veector/data/models/DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf"
    block_dir = "/workspaces/Veector/data/blocks"
    split_gguf_by_tensors(gguf_path, block_dir, virtual_space, "DeepSeek-R1-Distill-Qwen-1.5B")