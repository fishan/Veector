import torch
from pathlib import Path

tensor_dir = "/workspaces/Veector/data/blocks/DeepSeek-R1-Distill-Qwen-1.5B"
block_files = list(Path(tensor_dir).glob("DeepSeek-R1-Distill-Qwen-1.5B_row*_col*.pt"))
if block_files:
    first_block = block_files[0]
    try:
        block = torch.load(first_block, map_location="cpu")
        print(f"Успешно загружен блок {first_block.name}: {block.shape}")
    except Exception as e:
        print(f"Ошибка при загрузке блока {first_block.name}: {e}")
else:
    print("Файлы блоков не найдены.")