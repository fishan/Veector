import os
import torch
import numpy as np
from gguf import GGUFReader
from pathlib import Path
from virtual_space import VirtualSpace
from virtual_gguf import VirtualGGUF, VirtualGGUFFile
from llama_cpp import Llama
import logging
import tempfile

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class ModelManager:
    def __init__(self, veector, block_size=4 * 1024 * 1024, ipfs_enabled=False, model_dir="../data/models"):
        self.veector = veector
        self.block_size = block_size  # Теперь по умолчанию 4 MiB
        self.ipfs_enabled = ipfs_enabled
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.virtual_space = VirtualSpace(veector, use_ipfs=ipfs_enabled, model_manager=self)
        self.models = {}
        self.llama_instances = {}

    def load_gguf_model(self, model_name, gguf_path, block_dir, metadata_path, n_threads=4, n_ctx=2048, n_batch=512):
        model_block_dir = Path(block_dir) / model_name
        
        with open(metadata_path, "r") as f:
            import json
            metadata = json.load(f)
        self.virtual_space.switch_model(gguf_path, metadata)
        
        vgguf = VirtualGGUF(self.virtual_space, gguf_path)
        vfile = VirtualGGUFFile(vgguf)
        
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            total_size = vfile.size()
            for offset in range(0, total_size, self.block_size):
                chunk = vfile.read(min(self.block_size, total_size - offset))
                tmp.write(chunk)
                logger.debug(f"Wrote chunk at offset {offset}, size {len(chunk)}")
            tmp_path = tmp.name
        
        logger.debug(f"Loading GGUF model: {model_name} from VirtualSpace")
        reader = GGUFReader(tmp_path)
        
        logger.info(f"Model {model_name} loaded. Tensors: {len(reader.tensors)}")
        
        llama = Llama(model_path=tmp_path, n_threads=n_threads, n_ctx=n_ctx, n_batch=n_batch)
        self.llama_instances[tmp_path] = llama
        
        logger.info(f"Модель {model_name} успешно загружена из VirtualSpace")
        return tmp_path

if __name__ == "__main__":
    from core import Veector  # Оставляем как есть, раз нужен для языка Veector
    veector = Veector(use_memory=False, ipfs_enabled=False)
    manager = ModelManager(veector, ipfs_enabled=False)

    gguf_path = "/workspaces/Veector/data/models/DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf"
    block_dir = "/workspaces/Veector/data/blocks"
    model_name = "DeepSeek-R1-Distill-Qwen-1.5B"
    metadata_path = f"{block_dir}/{model_name}/metadata.json"
    
    from split_onnx import create_vib_blocks
    create_vib_blocks(gguf_path, block_dir, model_name, split_type="single")
    tmp_path = manager.load_gguf_model(model_name, gguf_path, block_dir, metadata_path)