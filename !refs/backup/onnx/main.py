import logging
import os
import json
from pathlib import Path
import torch
import gc
import psutil
from transformers import AutoTokenizer
from model_manager import ModelManager
from virtual_space import VirtualSpace
from observatory import Observer  # Последняя версия без UserAdapter

# Настройка логирования
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def print_memory_usage():
    process = psutil.Process(os.getpid())
    ram_usage = process.memory_info().rss / 1024**2
    logger.info(f"RAM использование: {ram_usage:.2f} MB")

def main():
    # Параметры модели
    model_name = "DeepSeek-R1-Distill-Qwen-1.5B"
    tensor_dir = f"/workspaces/Veector/data/blocks/{model_name}"
    model_config_dir = f"/workspaces/Veector/data/models/{model_name}"

    # Проверка директории с блоками
    if not os.path.exists(tensor_dir):
        print(f"Ошибка: Директория {tensor_dir} не существует.")
        exit(1)

    # Загрузка конфига
    config_path = os.path.join(tensor_dir, "config.json")
    if not os.path.exists(config_path):
        print(f"Ошибка: Файл config.json не найден в {tensor_dir}.")
        exit(1)

    with open(config_path, "r") as f:
        config = json.load(f)
        vocab_size = config["vocab_size"]
        hidden_size = config["hidden_size"]
        num_layers = config["num_hidden_layers"]
        num_attention_heads = config["num_attention_heads"]
        intermediate_size = config["intermediate_size"]
        key_dim = config.get("key_dim", 256)
        num_key_value_heads = config["num_key_value_heads"]

    # Загрузка токенизатора
    tokenizer = AutoTokenizer.from_pretrained(model_config_dir)
    logger.info(f"Переключено на модель: {model_name}")

    # Инициализация VirtualSpace
    virtual_space = VirtualSpace(tokenizer, use_ipfs=False)
    virtual_space.switch_model(model_name, vocab_size, hidden_size, num_layers, num_attention_heads, intermediate_size, key_dim, num_key_value_heads)

    # Проверка блоков
    block_files = list(Path(tensor_dir).glob(f"{model_name}_*_block*.pt"))
    if not block_files:
        print(f"Ошибка: Файлы блоков модели {model_name} не найдены в {tensor_dir}.")
        exit(1)
    else:
        logger.info(f"Найдено {len(block_files)} файлов блоков модели {model_name} в {tensor_dir}:")
        for block_file in block_files[:10]:
            logger.info(f" - {block_file.name}")
        if len(block_files) > 10:
            logger.info(f" ... и еще {len(block_files) - 10} файлов")

    # Инициализация Observer с токенизатором
    observer = Observer(virtual_space.dispatcher, tokenizer, max_layers=28, top_k=10)

    # Очистка памяти перед началом
    print("Очистка памяти перед инференсом...")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print_memory_usage()

    print("\nНачинаем чат с Observer (для выхода нажмите Ctrl+C)")
    try:
        while True:
            prompt = input("Вы: ")
            if not prompt.strip():
                print("Пожалуйста, введите текст.")
                continue

            # Генерация прямо через Observer
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": "<think>\n"}  # Добавляем директиву <think>
            ]
            formatted_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            input_ids = tokenizer.encode(formatted_input, add_special_tokens=False, return_tensors="pt").to(observer.device)
            logger.info(f"Input IDs shape: {input_ids.shape}")

            generated_ids, confidence = observer(input_ids, temperature=0.6, max_length=100)  # Увеличиваем max_length
            response = tokenizer.decode(generated_ids[0].tolist(), skip_special_tokens=True)
            logger.info(f"Generated response: {response}, Confidence: {confidence}")

            print(f"Veector: {response} (Confidence: {confidence:.2f})")
            print_memory_usage()

    except KeyboardInterrupt:
        print("\nЧат завершен.")
        observer.clear_memory()
    except Exception as e:
        logger.error(f"Ошибка в процессе: {str(e)}", exc_info=True)
        print("Проверь логи выше или пришли мне полный traceback!")
        observer.clear_memory()

if __name__ == "__main__":
    main()