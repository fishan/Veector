import os
import onnx

# Путь к модели
model_dir = "../data/models/DeepSeek-R1-Distill-Qwen-1.5B-ONNX"
split_model_path = os.path.join(model_dir, "model_split.onnx")

# Загружаем модель без данных
onnx_model = onnx.load(split_model_path, load_external_data=False)
print("Модель загружена без внешних данных")

# Проверяем тензоры с external_data
for tensor in onnx_model.graph.initializer:
    if tensor.data_location == onnx.TensorProto.EXTERNAL:
        print(f"Тензор: {tensor.name}")
        print(f"  data_location: {tensor.data_location}")
        print(f"  external_data: {[(kv.key, kv.value) for kv in tensor.external_data]}")