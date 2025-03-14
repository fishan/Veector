# device/src/core.py
import numpy as np
from src.veectordb import VeectorDB  # Correct path import

class Veector:
    def __init__(self, db_path="/workspaces/Veector/device/data/db/user_data.json"):  # Correct Path to
        self.db = VeectorDB(db_path)
        self.space = {} # Simple in-memory space
        self.max_coord = 0 # Simple coordinate tracking

    def _next_coords(self):
        coords = max([key[1][0] for key in self.space.keys()] + [self.max_coord]) + 1
        self.max_coord = coords
        return [coords, coords, coords]

    def add_to_space(self, tensor):
        layer, coords = tensor[0][0], tuple(tensor[0][1])
        self.space[(tuple(layer), coords)] = tensor # Store tensor directly

    def compute(self, tensor):
        # Placeholder compute function
        print(f"Computing tensor: {tensor}")
        return np.random.rand(1, 512)  # Dummy output


from qiskit import QuantumCircuit
from qiskit.primitives import Sampler

# Создание квантовой схемы
circuit = QuantumCircuit(2)
circuit.h(0)  # Гейт Адамара
circuit.cx(0, 1)  # CNOT-гейт

# Использование Sampler
sampler = Sampler()
result = sampler.run(circuit).result()

# Получение результатов
quasi_dists = result.quasi_dists
print("Quasi-distributions:", quasi_dists)



////////////////////////

# device/run_veector.py
import torch
from transformers import AutoTokenizer
import os
import sys
import numpy as np

# Указываем путь к device/src
sys.path.append("/workspaces/Veector/device/src")

from core import Veector
from model_manager import ModelManager
from virtual_space import VirtualSpace

def main():
    # Инициализация Veector
    db_path = "/workspaces/Veector/device/data/db/user_data.json"
    veector = Veector(db_path=db_path, use_neural_storage=False, use_memory=False, ipfs_enabled=False)

    # Путь к блокам модели
    weights_path = "/workspaces/Veector/data/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    if not os.path.exists(weights_path):
        print(f"Папка с моделью {weights_path} не найдена!")
        sys.exit(1)

    # Инициализация ModelManager
    model_manager = ModelManager(veector, block_size=(1024, 1024), ipfs_enabled=False, model_dir=weights_path)
    model_name = "DeepSeek-R1-Distill-Qwen-1.5B"

    # Загрузка модели из локальной папки
    print("Загрузка модели из локальной папки...")
    model_manager.load_pre_split_model(model_name, weights_path)
    print(f"Модель загружена. Количество блоков: {len(model_manager.model_space)}")

    # Токенизатор
    tokenizer_path = weights_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Основной цикл чата
    print("Чат готов! Введите текст (или 'exit' для выхода):")
    while True:
        prompt = input("> ")
        if prompt.lower() == "exit":
            break

        # Токенизация ввода
        input_ids = tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=512)
        print(f"Размер входных данных: {input_ids.shape}")

        # Инференс
        try:
            output_ids = model_manager.perform_inference(model_name, input_ids.numpy())
            if output_ids is not None:
                response = tokenizer.decode(output_ids.flatten(), skip_special_tokens=True)
                print(f"DeepSeek: {response}")
            else:
                print("Ошибка: модель вернула None")
        except Exception as e:
            print(f"Ошибка при инференсе: {e}")

if __name__ == "__main__":
    main()