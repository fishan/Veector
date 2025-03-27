import os
import onnxruntime as ort

model_dir = "../data/models/DeepSeek-R1-Distill-Qwen-1.5B-ONNX"
split_model_path = os.path.join(model_dir, "model_split.onnx")
session = ort.InferenceSession(split_model_path)
print("Входы:")
for inp in session.get_inputs():
    print(f"{inp.name}: {inp.shape}")
print("Выходы:")
for out in session.get_outputs():
    print(f"{out.name}: {out.shape}")