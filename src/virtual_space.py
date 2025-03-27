import logging
from pathlib import Path

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class VirtualSpace:
    def __init__(self, veector, use_ipfs=False, model_manager=None):
        self.veector = veector
        self.use_ipfs = use_ipfs
        self.model_manager = model_manager
        self.models = {}
        self.dispatchers = {}

    def switch_model(self, model_path, metadata):
        self.models[model_path] = {
            "metadata": metadata,
            "blocks": {}
        }
        self.dispatchers[model_path] = VirtualDispatcher(self, model_path)
        logger.debug(f"Switched to model {model_path} with {len(metadata)} blocks")

    def load_block(self, block_key, model_path):
        model_data = self.models[model_path]
        if block_key in model_data["blocks"]:
            return model_data["blocks"][block_key]
        
        meta = model_data["metadata"][block_key]
        block_path = Path(model_path).parent / meta["file"]
        
        if block_path.exists():
            with open(block_path, "rb") as f:
                block_data = f.read()
            model_data["blocks"][block_key] = block_data
            logger.debug(f"Loaded block {block_key} from {block_path}")
            return block_data
        logger.warning(f"Block {block_key} not available")
        return None

class VirtualDispatcher:
    def __init__(self, virtual_space, model_path):
        self.virtual_space = virtual_space
        self.model_path = model_path

    def load_block(self, block_key):
        return self.virtual_space.load_block(block_key, self.model_path)