import os
import psutil
import onnxruntime as ort

def log_memory():
    memory = psutil.virtual_memory()
    return f"Память: {memory.used / (1024 * 1024):.2f} MB / {memory.total / (1024 * 1024):.2f} MB"

print(f"🟢 [LOG] Настройка окружения... {log_memory()}")

model_dir = "../data/models/DeepSeek-R1-Distill-Qwen-1.5B-ONNX"
split_model_path = os.path.join(model_dir, "model_split.onnx")

session_options = ort.SessionOptions()
session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
print(f"🟢 [LOG] Перед загрузкой сессии {log_memory()}")
session = ort.InferenceSession(split_model_path, sess_options=session_options)
print(f"🟢 [LOG] ✅ ONNX Runtime загружен {log_memory()}")

print(f"🟢 [LOG] Ждём 5 секунд... {log_memory()}")
import time
time.sleep(5)
print(f"🟢 [LOG] Завершение {log_memory()}")