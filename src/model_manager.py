import requests
import os
import logging
from pathlib import Path
import onnxruntime as ort
import numpy as np
from transformers import AutoTokenizer

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class ModelManager:
    def __init__(self, veector, block_size=4 * 1024 * 1024, ipfs_enabled=False, model_dir="../data/models"):
        self.veector = veector
        self.block_size = block_size
        self.ipfs_enabled = ipfs_enabled
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.onnx_sessions = {}
        self.tokenizer = AutoTokenizer.from_pretrained("onnx-community/DeepSeek-R1-Distill-Qwen-1.5B-ONNX")

    def load_onnx_model(self, model_name, onnx_path, block_dir, metadata_path):
        local_dir = block_dir / model_name
        local_dir.mkdir(parents=True, exist_ok=True)
        
        # Скачиваем основной файл модели
        model_url = "http://localhost:8000/model"
        model_response = requests.get(model_url)
        split_model_path = local_dir / f"{model_name}_split.onnx"
        with open(split_model_path, "wb") as f:
            f.write(model_response.content)
        
        # Загружаем модель без внешних данных
        model = onnx.load(str(split_model_path), load_external_data=False)
        
        # Скачиваем внешние файлы по мере необходимости
        for tensor in model.graph.initializer:
            if tensor.data_location == onnx.TensorProto.EXTERNAL:
                for data in tensor.external_data:
                    if data.key == "location":
                        filename = data.value
                        weight_url = f"http://localhost:8000/weights/{filename}"
                        weight_path = local_dir / filename
                        if not weight_path.exists():
                            weight_response = requests.get(weight_url)
                            with open(weight_path, "wb") as f:
                                f.write(weight_response.content)
        
        session = ort.InferenceSession(str(split_model_path))
        self.onnx_sessions[onnx_path] = session
        logger.info(f"Модель {model_name} загружена с внешними данными через API")
        return onnx_path

    def chat(self, onnx_path):
        session = self.onnx_sessions.get(onnx_path)
        if not session:
            print("Модель не загружена!")
            return
        
        print("Чат с моделью DeepSeek-R1-Distill-Qwen-1.5B-ONNX. Введите 'exit' для выхода.")
        while True:
            prompt = input("Вы: ")
            if prompt.lower() == "exit":
                break
            
            inputs = self.tokenizer(prompt, return_tensors="np")
            input_ids = inputs["input_ids"].astype(np.int64)
            
            outputs = session.run(None, {"input_ids": input_ids})
            output_ids = outputs[0].argmax(axis=-1)
            response = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            print(f"Модель: {response}")

if __name__ == "__main__":
    from core import Veector
    veector = Veector(use_memory=False, ipfs_enabled=False)
    manager = ModelManager(veector, ipfs_enabled=False)

    onnx_path = "model.onnx"
    block_dir = Path("../data/blocks")
    model_name = "DeepSeek-R1-Distill-Qwen-1.5B-ONNX"
    metadata_path = block_dir / model_name / "metadata.json"
    
    from split_onnx import split_onnx_to_external_data
    split_onnx_to_external_data(onnx_path, block_dir, model_name)
    model_path = manager.load_onnx_model(model_name, onnx_path, block_dir, metadata_path)
    
    manager.chat(model_path)