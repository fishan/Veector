import os
from gguf import GGUFReader
from pathlib import Path
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def create_vib_blocks(gguf_path, block_dir, model_name, split_type="single", block_size=None):
    gguf_path = Path(gguf_path)
    model_block_dir = Path(block_dir) / model_name
    model_block_dir.mkdir(parents=True, exist_ok=True)
    
    reader = GGUFReader(gguf_path)
    total_size = os.path.getsize(gguf_path)
    metadata = {}

    with open(gguf_path, "rb") as f:
        gguf_data = f.read()

    if split_type == "single":
        vib_path = model_block_dir / "model.vib"
        with open(vib_path, "wb") as f:
            f.write(gguf_data)
        metadata["model"] = {
            "file": "model.vib",
            "offset": 0,
            "size": total_size,
            "model_name": model_name,
            "coords": (0, 0)
        }
        logger.info(f"Created single .vib file at {vib_path}")

    elif split_type == "tensors":
        header_size = reader.header_size
        header_path = model_block_dir / "header.vib"
        with open(header_path, "wb") as f:
            f.write(gguf_data[:header_size])
        metadata["header"] = {
            "file": "header.vib",
            "offset": 0,
            "size": header_size,
            "model_name": model_name,
            "coords": (0, 0)
        }
        
        for idx, tensor in enumerate(reader.tensors):
            block_key = f"tensor_{tensor.name}"
            vib_path = model_block_dir / f"{block_key}.vib"
            with open(vib_path, "wb") as f:
                f.write(tensor.data)
            metadata[block_key] = {
                "file": vib_path.name,
                "offset": tensor.offset,
                "size": len(tensor.data),
                "model_name": model_name,
                "coords": (idx, 0)
            }
        logger.info(f"Split into {len(reader.tensors) + 1} .vib files by tensors")

    elif split_type == "fixed" and block_size:
        offset = 0
        idx = 0
        while offset < total_size:
            vib_path = model_block_dir / f"block_{idx}.vib"
            block_data = gguf_data[offset:offset + block_size]
            with open(vib_path, "wb") as f:
                f.write(block_data)
            metadata[f"block_{idx}"] = {
                "file": vib_path.name,
                "offset": offset,
                "size": len(block_data),
                "model_name": model_name,
                "coords": (idx, 0)
            }
            offset += len(block_data)
            idx += 1
        logger.info(f"Split into {idx} .vib files of size {block_size}")

    else:
        raise ValueError("Invalid split_type or missing block_size")

    metadata_path = model_block_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        import json
        json.dump(metadata, f, indent=2)

if __name__ == "__main__":
    gguf_path = "/workspaces/Veector/data/models/DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf"
    block_dir = "/workspaces/Veector/data/blocks"
    model_name = "DeepSeek-R1-Distill-Qwen-1.5B"
    create_vib_blocks(gguf_path, block_dir, model_name, split_type="single")