# device/run_veector.py
import torch
from transformers import AutoTokenizer
import os
import sys
sys.path.append("device/src")

from src.core import Veector
from src.model_manager import ModelManager
from src.virtual_space import VirtualSpace

def dynamic_inference(input_text):
    # Токенизация
    inputs = tokenizer(input_text, return_tensors='pt').to(device)
    
    # Загрузка необходимых блоков
    embeddings = matrix.load_block('embeddings', block_hashes['embeddings'])
    hidden_states = embeddings(inputs['input_ids'])
    
    for i in range(config.num_hidden_layers):
        layer_name = f"transformer_layer_{i}"
        layer = matrix.load_block(layer_name, block_hashes[layer_name])
        hidden_states = layer(hidden_states)
    
    classifier = matrix.load_block('classifier', block_hashes['classifier'])
    logits = classifier(hidden_states)
    
    return torch.argmax(logits, dim=-1).item()

def main():
    veector = Veector(db_path="/workspaces/Veector/device/data/db/user_data.json")
    weights_path = "/workspaces/Veector/data/deepseek-ai/deepseek-coder-1.3b-base"
    cache_path = "/workspaces/Veector/device/data/local_cache"
    model_manager = ModelManager(veector, model_dir=weights_path, cache_dir=cache_path)

    model_name = "deepseek-coder-1.3b"
    model_manager.load_pre_split_model(model_name, weights_path)
    print("Model metadata loaded. Available tensors:", len(model_manager.model_space))

    tokenizer = AutoTokenizer.from_pretrained(weights_path)

    while True:
        prompt = input("Enter your prompt (or 'exit'): ")
        if prompt.lower() == "exit":
            break

        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        print(f"Input shape: {input_ids.shape}")

        output_ids = model_manager.perform_inference(model_name, input_ids)
        if output_ids is not None:
            response = tokenizer.decode(output_ids, skip_special_tokens=True)
            print(f"DeepSeek: {response}")
        else:
            print("Inference failed.")

if __name__ == "__main__":
    main()


