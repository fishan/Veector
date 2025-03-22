import os
import psutil
import onnxruntime as ort

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

# Создание сессии с минимальными опциями
session_options = ort.SessionOptions()
session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL  # Без оптимизаций
print(f"🟢 [LOG] Перед загрузкой сессии {log_memory()}")
session = ort.InferenceSession(split_model_path, sess_options=session_options)
print(f"🟢 [LOG] ✅ ONNX Runtime загружен. Входы: {[inp.name for inp in session.get_inputs()]} {log_memory()}")