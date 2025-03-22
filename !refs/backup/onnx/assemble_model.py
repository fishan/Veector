# assemble_model.py
import sys
from pathlib import Path
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def load_part(part_id, block_dir, model_name):
    vib_path = Path(block_dir) / model_name / f"part_{part_id:05d}.vib"
    if vib_path.exists():
        with open(vib_path, "rb") as f:
            return f.read()
    return b""

def assemble_model(block_dir, model_name):
    combined_model = b""
    part_id = 0
    while True:
        part_data = load_part(part_id, block_dir, model_name)
        if not part_data:
            break
        combined_model += part_data
        part_id += 1
    return combined_model

if __name__ == "__main__":
    block_dir = "../data/blocks"
    model_name = "DeepSeek-R1-Distill-Qwen-1.5B-ONNX"
    model_data = assemble_model(block_dir, model_name)
    sys.stdout.buffer.write(model_data)  # Выводим бинарные данные в stdout