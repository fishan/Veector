import torch
import numpy as np
import logging
from sync import P2PNode
from pathlib import Path

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class VirtualSpace:
    def __init__(self, veector, use_ipfs=False, model_manager=None):
        self.veector = veector
        self.use_ipfs = use_ipfs
        self.model_manager = model_manager
        self.p2p_node = P2PNode("localhost", 5000, use_ipfs=use_ipfs)
        self.models = {}
        self.device = torch.device("cpu")
        self.dispatchers = {}

    def switch_model(self, model_path, metadata):
        self.models[model_path] = {
            "metadata": metadata,
            "blocks": {}
        }
        self.dispatchers[model_path] = VirtualDispatcher(self, model_path)
        logger.debug(f"Switched to model {model_path} with {len(metadata)} blocks")

    def load_block(self, block_key, model_path):
        if model_path not in self.models:
            logger.error(f"Model {model_path} not found in VirtualSpace")
            return None
        
        model_data = self.models[model_path]
        if block_key in model_data["blocks"]:
            return model_data["blocks"][block_key]
        
        if block_key not in model_data["metadata"]:
            logger.warning(f"Block {block_key} not found in metadata")
            return None
        
        meta = model_data["metadata"][block_key]
        model_name = meta.get("model_name", "unknown_model")
        block_path = Path(model_path).parent.parent / "blocks" / model_name / meta["file"]
        
        if block_path.exists():
            with open(block_path, "rb") as f:
                block_data = f.read()
            # Делаем массив записываемым через copy()
            block_array = np.frombuffer(block_data, dtype=np.uint8).copy()
            block = torch.from_numpy(block_array)
            model_data["blocks"][block_key] = block
            logger.debug(f"Loaded block {block_key} from {block_path}")
            return block
        
        model_name = meta.get("model_name", "whole_model")
        coords = meta.get("coords", (0, 0))
        ipfs_hash = self.p2p_node.block_map.get(model_name, {}).get(coords)
        
        if ipfs_hash and self.use_ipfs:
            logger.debug(f"Fetching {block_key} from IPFS: {ipfs_hash}")
            block_data = self.p2p_node.ipfs_client.cat(ipfs_hash)
            block_array = np.frombuffer(block_data, dtype=np.uint8).copy()
            block = torch.from_numpy(block_array)
            model_data["blocks"][block_key] = block
            with open(block_path, "wb") as f:
                f.write(block_data)
            logger.debug(f"Fetched and cached {block_key} from IPFS")
            return block
        
        logger.warning(f"Block {block_key} not available")
        return None

    def sync_block(self, block_key, block, model_path):
        if model_path not in self.models:
            logger.error(f"Model {model_path} not found in VirtualSpace")
            return
        
        model_data = self.models[model_path]
        meta = model_data["metadata"].get(block_key)
        if not meta:
            logger.error(f"Metadata for {block_key} not found")
            return
        
        model_name = meta.get("model_name", "whole_model")
        coords = meta.get("coords", (0, 0))
        self.p2p_node.sync_tensor(block, {
            "tensor_id": f"{model_name}_{block_key}",
            "model_name": model_name,
            "coords": coords
        })
        model_data["blocks"][block_key] = block
        logger.debug(f"Synced block {block_key} to P2P")

class VirtualDispatcher:
    def __init__(self, virtual_space, model_path):
        self.virtual_space = virtual_space
        self.model_path = model_path
        self.gguf_path = model_path

    def load_block(self, block_key):
        return self.virtual_space.load_block(block_key, self.model_path)