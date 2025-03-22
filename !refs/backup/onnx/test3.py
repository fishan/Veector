import os
import numpy as np
import psutil
import onnxruntime as ort
from transformers import AutoTokenizer

# Логирование памяти
def log_memory():
    memory = psutil.virtual_memory()
    return f"Память: {memory.used / (1024 * 1024):.2f} MB / {memory.total / (1024 * 1024):.2f} MB"

print(f"🟢 [LOG] Настройка окружения... {log_memory()}")

# Путь к модели
model_dir = "../data/models/DeepSeek-R1-Distill-Qwen-1.5B-ONNX"
split_model_path = os.path.join(model_dir, "model_split.onnx")

# Проверка файлов
files = os.listdir(model_dir)
print(f"🟢 [LOG] 📂 Файлы модели в {model_dir}:")
total_size = 0
for f in files:
    size_mb = os.path.getsize(os.path.join(model_dir, f)) / (1024 * 1024)
    total_size += size_mb
    print(f"🟢 [LOG] 📄 {f}: {size_mb:.2f} MB")
print(f"🟢 [LOG] Файлы проверены, общий размер: {total_size:.2f} MB {log_memory()}")

# Создание сессии
session_options = ort.SessionOptions()
session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
print(f"🟢 [LOG] Перед загрузкой сессии {log_memory()}")
session = ort.InferenceSession(split_model_path, sess_options=session_options)
print(f"🟢 [LOG] ✅ ONNX Runtime загружен. Входы: {[inp.name for inp in session.get_inputs()]} {log_memory()}")

# Токенизатор
tokenizer = AutoTokenizer.from_pretrained(model_dir)
print(f"🟢 [LOG] Токенизатор загружен {log_memory()}")

# Минимальный инференс
input_ids = tokenizer("Hello", return_tensors="np")["input_ids"].astype(np.int64)
attention_mask = np.ones_like(input_ids, dtype=np.int64)
position_ids = np.arange(input_ids.shape[1], dtype=np.int64)[None, :]
input_feed = {
    "input_ids": input_ids,
    "attention_mask": attention_mask,
    "position_ids": position_ids,
}
for i in range(28):
    input_feed[f"past_key_values.{i}.key"] = np.zeros((1, 2, 0, 128), dtype=np.float16)
    input_feed[f"past_key_values.{i}.value"] = np.zeros((1, 2, 0, 128), dtype=np.float16)

print(f"🟢 [LOG] Вход подготовлен {log_memory()}")
outputs = session.run(None, input_feed)
print(f"🟢 [LOG] Инференс выполнен {log_memory()}")
print(f"🟢 [LOG] Выходной массив: {outputs[0].shape}")