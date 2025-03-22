import os
from gguf import GGUFReader
from pathlib import Path
import json
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def split_gguf(gguf_path, block_dir, block_size=4096 * 1024):  # 4 MiB блоки по умолчанию
    gguf_path = Path(gguf_path)
    block_dir = Path(block_dir)
    block_dir.mkdir(parents=True, exist_ok=True)

    reader = GGUFReader(gguf_path)
    total_size = os.path.getsize(gguf_path)
    blocks = {}
    offset = 0

    # Делим файл на блоки
    with open(gguf_path, "rb") as f:
        block_idx = 0
        while offset < total_size:
            block_path = block_dir / f"block_{block_idx}.bin"
            block_data = f.read(block_size)
            block_size_actual = len(block_data)
            
            with open(block_path, "wb") as bf:
                bf.write(block_data)
            
            blocks[f"block_{block_idx}"] = {
                "file": block_path.name,
                "offset": offset,
                "size": block_size_actual
            }
            offset += block_size_actual
            block_idx += 1

    # Сохраняем метаданные
    metadata_path = block_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(blocks, f, indent=2)
    
    logger.info(f"GGUF split into {block_idx} blocks in {block_dir}")

if __name__ == "__main__":
    gguf_path = "/workspaces/Veector/data/models/DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf"
    block_dir = "/workspaces/Veector/data/blocks"
    split_gguf(gguf_path, block_dir)