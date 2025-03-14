# /workspaces/Veector/device/run_veector.py
import sys
sys.path.append("/workspaces/Veector/device/src")
from core import Veector
import os
import torch

def main():
    blocks_path = "/workspaces/Veector/data/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"  # Путь к блокам
    db_path = "../data/db/veectordb.json"
    model_name = "DeepSeek-R1-Distill-Qwen-1.5B"

    print("Загрузка модели из блоков...")
    veector = Veector(db_path=db_path, use_neural_storage=False, use_memory=False, ipfs_enabled=False)
    veector.model_manager.load_pre_split_model(model_name, blocks_path)

    print(f"Veector chat running with model blocks: {blocks_path}")
    while True:
        message = input("You: ")
        if message.lower() in ["exit", "quit"]:
            break
        # Пока заглушка для инференса
        input_ids = torch.randint(0, 32768, (1, 512))  # Заменить на токенизатор позже
        output = veector.model_manager.perform_inference(model_name, input_ids.numpy())
        print(f"Veector: {output.shape} (ответ пока не реализован)")

if __name__ == "__main__":
    main()