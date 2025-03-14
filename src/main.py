# /workspaces/Veector/src/main.py
from core import Veector
from model_manager import ModelManager
import numpy as np
import os
import json
from pathlib import Path
import torch
import gc
import psutil

def print_memory_usage():
    if torch.cuda.is_available():
        print(f"GPU память: {torch.cuda.memory_allocated() / 1024**2:.2f} MB выделено, "
              f"{torch.cuda.memory_reserved() / 1024**2:.2f} MB зарезервировано")
    process = psutil.Process(os.getpid())
    ram_usage = process.memory_info().rss / 1024**2  # В MB
    print(f"RAM использование: {ram_usage:.2f} MB")

veector = Veector(use_memory=False, ipfs_enabled=False)
model_manager = ModelManager(veector, ipfs_enabled=False, model_dir="/workspaces/Veector/data")
veector.model_manager = model_manager

model_name = "DeepSeek-R1-Distill-Qwen-1.5B"
tensor_dir = f"/workspaces/Veector/data/blocks/{model_name}"

if not os.path.exists(tensor_dir):
    print(f"Ошибка: Директория {tensor_dir} не существует.")
    exit(1)

config_path = os.path.join(tensor_dir, "config.json")
if not os.path.exists(config_path):
    print(f"Ошибка: Файл config.json не найден в {tensor_dir}.")
    exit(1)
with open(config_path, "r") as f:
    config = json.load(f)

vocab_size = config.get("vocab_size")
hidden_size = config.get("hidden_size")
num_layers = config.get("num_hidden_layers")
print(f"Параметры из config.json: vocab_size={vocab_size}, hidden_size={hidden_size}, num_layers={num_layers}")

if vocab_size is None or hidden_size is None or num_layers is None:
    print(f"Ошибка: Не все параметры найдены в config.json: {config}")
    exit(1)

block_files = list(Path(tensor_dir).glob(f"{model_name}_*_block*.pt"))
if not block_files:
    print(f"Ошибка: Файлы блоков модели {model_name} не найдены в {tensor_dir}.")
    print(f"Ожидаемый формат файлов: {model_name}_*_blockN.pt")
    print(f"Содержимое директории через os.listdir:")
    all_files = os.listdir(tensor_dir)
    for f in all_files:
        print(f" - {f}")
    exit(1)
else:
    print(f"Найдено {len(block_files)} файлов блоков модели {model_name} в {tensor_dir}:")
    for block_file in block_files[:10]:
        print(f" - {block_file.name}")
    if len(block_files) > 10:
        print(f" ... и еще {len(block_files) - 10} файлов")

try:
    veector.model_manager.load_pre_split_model(model_name, tensor_dir, vocab_size, hidden_size, num_layers)
    print(f"Модель {model_name} успешно загружена.")
except Exception as e:
    print(f"Ошибка при загрузке модели: {e}")
    exit(1)

# Очистка памяти перед инференсом
print("Очистка памяти перед инференсом...")
if torch.cuda.is_available():
    torch.cuda.empty_cache()
gc.collect()
print_memory_usage()

max_sequence_length = 32  # Ещё меньше, чтобы точно влезло
batch_size = 1
input_data = np.random.randint(0, vocab_size, (batch_size, max_sequence_length), dtype=np.int32)
print(f"Сгенерированные входные данные: {input_data.shape}")

print("Запуск инференса...")
output = veector.model_manager.perform_inference(model_name, input_data)
print(f"Результат инференса: {output.shape}")

# Очистка после инференса
if torch.cuda.is_available():
    torch.cuda.empty_cache()
gc.collect()
print_memory_usage()