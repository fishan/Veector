# Установка библиотек
!pip install onnx onnxruntime transformers psutil huggingface_hub

import onnx
import os
import numpy as np
import shutil
import zipfile
import psutil
import onnxruntime as ort
from google.colab import drive
from transformers import AutoTokenizer
from onnx.external_data_helper import load_external_data_for_model
from huggingface_hub import hf_hub_download

# Настройка логирования
print("🟢 [LOG] Настройка окружения...")

# Загрузка модели
model_repo = "onnx-community/DeepSeek-R1-Distill-Qwen-1.5B-ONNX"
model_filename = "onnx/model_quantized.onnx"  # Используем quantized версию
model_path = hf_hub_download(repo_id=model_repo, filename=model_filename)
print(f"🟢 [LOG] ✅ Модель загружена: {model_path}")

# Создаем папку для модели
model_dir = "model_files"
os.makedirs(model_dir, exist_ok=True)
split_model_path = os.path.join(model_dir, "model_split.onnx")

# Разбиваем модель
chunk_size = 1024 * 1024 * 50  # 50 MB
model = onnx.load(model_path)
onnx.save_model(model, split_model_path, save_as_external_data=True, all_tensors_to_one_file=False, size_threshold=chunk_size)

# Проверяем и логируем файлы
files = [f for f in os.listdir(model_dir) if f.startswith("model_split") or "quantized" in f]
print(f"🟢 [LOG] 📂 Файлы модели в {model_dir}:")
for f in files:
    size_mb = os.path.getsize(os.path.join(model_dir, f)) / (1024 * 1024)
    print(f"🟢 [LOG] 📄 {f}: {size_mb:.2f} MB")

# Архивация модели
zip_name = "DeepSeek-R1-Distill-Qwen-1.5B-splited-onnx.zip"
with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for f in files:
        zipf.write(os.path.join(model_dir, f), arcname=f)  # Сохраняем без пути папки
print(f"🟢 [LOG] 📦 Архив создан: {zip_name}, размер: {os.path.getsize(zip_name) / (1024 * 1024):.2f} MB")

# Выгрузка на Google Drive
drive.mount('/content/drive', force_remount=True)
destination_path = f"/content/drive/My Drive/{zip_name}"
shutil.copy(zip_name, destination_path)
print(f"🟢 [LOG] ✅ Архив загружен на Google Drive: {destination_path}")

# Загрузка модели из папки
os.environ["ONNX_LOAD_EXTERNAL_LOGGING"] = "1"
onnx_model = onnx.load(split_model_path, load_external_data=False)
load_external_data_for_model(onnx_model, model_dir)
onnx.save(onnx_model, split_model_path)
session = ort.InferenceSession(split_model_path)
print("🟢 [LOG] ✅ ONNX Runtime загружен.")

# Логируем память
memory_info = psutil.virtual_memory()
print(f"🟢 [LOG] 📊 Память: {memory_info.used / (1024 * 1024):.2f} MB / {memory_info.total / (1024 * 1024):.2f} MB")

# Токенизатор
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")

# Параметры модели
num_hidden_layers = 28
num_key_value_heads = 2
head_dim = 128
max_new_tokens = 512

# Подготовка входных данных
PROMPT_TEMPLATE = "<｜User｜>{message}<｜Assistant｜>"

def preprocess_text(text):
    formatted_text = PROMPT_TEMPLATE.format(message=text)
    inputs = tokenizer(formatted_text, return_tensors="np")
    input_feed = {
        "input_ids": inputs["input_ids"].astype(np.int64),
        "attention_mask": inputs["attention_mask"].astype(np.int64),
        "position_ids": np.arange(inputs["input_ids"].shape[1], dtype=np.int64)[None, :],
    }
    for i in range(num_hidden_layers):
        input_feed[f"past_key_values.{i}.key"] = np.zeros((1, num_key_value_heads, 0, head_dim), dtype=np.float32)
        input_feed[f"past_key_values.{i}.value"] = np.zeros((1, num_key_value_heads, 0, head_dim), dtype=np.float32)
    return input_feed, inputs["input_ids"], inputs["attention_mask"]

# Генерация текста с стримингом
def generate_text(input_feed, input_ids, attention_mask, max_new_tokens):
    generated_ids = input_ids[0].tolist()
    past_key_values = {k: v for k, v in input_feed.items() if "past_key_values" in k}

    # Принудительно добавляем "<think>\n"
    think_tokens = tokenizer.encode("<think>\n", add_special_tokens=False)
    generated_ids.extend(think_tokens)
    print(f"🟢 [STREAM] {tokenizer.decode(think_tokens, skip_special_tokens=False)}", end='', flush=True)

    # Первый шаг
    print(f"🟢 [LOG] Первый шаг: input_ids shape={input_ids.shape}")
    outputs = session.run(None, input_feed)
    logits = outputs[0][:, -1, :]
    temperature = 0.6
    logits = logits / temperature
    probs = np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)
    next_token = np.random.choice(len(probs[0]), p=probs[0])
    generated_ids.append(next_token)
    print(f"🟢 [STREAM] {tokenizer.decode([next_token], skip_special_tokens=False)}", end='', flush=True)

    # Обновляем past_key_values
    for i in range(num_hidden_layers):
        past_key_values[f"past_key_values.{i}.key"] = outputs[2 * i + 1]
        past_key_values[f"past_key_values.{i}.value"] = outputs[2 * i + 2]
        if i == 0:  # Логируем только для первого слоя
            print(f"🟢 [LOG] past_key_values.{i}.key shape={past_key_values[f'past_key_values.{i}.key'].shape}")

    # Последующие шаги
    for step in range(max_new_tokens - 1):
        seq_length = past_key_values["past_key_values.0.key"].shape[2]
        input_feed = {
            "input_ids": np.array([[next_token]], dtype=np.int64),
            "attention_mask": np.ones((1, seq_length + 1), dtype=np.int64),
            "position_ids": np.arange(seq_length, seq_length + 1, dtype=np.int64)[None, :],
        }
        input_feed.update(past_key_values)

        print(f"🟢 [LOG] Шаг {step + 1}: input_ids shape={input_feed['input_ids'].shape}, attention_mask shape={input_feed['attention_mask'].shape}")
        outputs = session.run(None, input_feed)
        logits = outputs[0][:, -1, :]
        logits = logits / temperature
        probs = np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)

        if step < 10 and 151648 in tokenizer.get_vocab():
            probs[0, 151648] = 0  # Исключаем "<think>"
            probs = probs / np.sum(probs, axis=-1, keepdims=True)

        next_token = np.random.choice(len(probs[0]), p=probs[0])
        generated_ids.append(next_token)
        print(f"🟢 [STREAM] {tokenizer.decode([next_token], skip_special_tokens=False)}", end='', flush=True)

        for i in range(num_hidden_layers):
            past_key_values[f"past_key_values.{i}.key"] = outputs[2 * i + 1]
            past_key_values[f"past_key_values.{i}.value"] = outputs[2 * i + 2]

        if next_token == tokenizer.eos_token_id:
            break

    print()  # Переход на новую строку
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

# Чат
def chat():
    print("\n🤖 ONNX-Чат активен! Напиши что-нибудь ('выход' для выхода).")
    while True:
        user_input = input("Ты: ")
        if user_input.lower() == "выход":
            print("🤖 Чат завершен.")
            break

        print("🟢 [LOG] Начинаем обработку запроса...")
        input_feed, input_ids, attention_mask = preprocess_text(user_input)

        try:
            response_text = generate_text(input_feed, input_ids, attention_mask, max_new_tokens)
            print(f"🟢 [LOG] Генерация завершена. Память: {psutil.virtual_memory().used / (1024 * 1024):.2f} MB")
            print(f"🤖 ONNX: {response_text}")
        except Exception as e:
            print(f"🟢 [LOG] Ошибка генерации: {e}")

# Запуск
chat()