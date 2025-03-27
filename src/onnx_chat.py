import os
import numpy as np
import psutil
import onnxruntime as ort
from transformers import AutoTokenizer
import gc

def log_memory():
    memory = psutil.virtual_memory()
    return f"Память: {memory.used / (1024 * 1024):.2f} MB / {memory.total / (1024 * 1024):.2f} MB"

print(f"🟢 [LOG] Настройка окружения... {log_memory()}")
gc.collect()

model_dir = "../data/models/DeepSeek-R1-Distill-Qwen-1.5B-ONNX"
split_model_path = os.path.join(model_dir, "model_split.onnx")

session_options = ort.SessionOptions()
session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
session = ort.InferenceSession(split_model_path, sess_options=session_options)
print(f"🟢 [LOG] ✅ ONNX Runtime загружен {log_memory()}")

tokenizer = AutoTokenizer.from_pretrained(model_dir)
print(f"🟢 [LOG] Токенизатор загружен {log_memory()}")

num_hidden_layers = 28
num_key_value_heads = 2
head_dim = 1536 // 12  # 128
max_new_tokens = 3

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
        input_feed[f"past_key_values.{i}.key"] = np.zeros((1, num_key_value_heads, 0, head_dim), dtype=np.float16)
        input_feed[f"past_key_values.{i}.value"] = np.zeros((1, num_key_value_heads, 0, head_dim), dtype=np.float16)
    gc.collect()
    return input_feed, inputs["input_ids"], inputs["attention_mask"]

def generate_text(input_feed, input_ids, attention_mask, max_new_tokens):
    generated_ids = input_ids[0].tolist()
    past_key_values = {k: v for k, v in input_feed.items() if "past_key_values" in k}

    outputs = session.run(None, input_feed)
    logits = outputs[0][:, -1, :].astype(np.float16)
    next_token = np.argmax(logits, axis=-1)[0]
    generated_ids.append(next_token)
    print(f"🟢 [STREAM] {tokenizer.decode([next_token])}", end='', flush=True)

    for i in range(num_hidden_layers):
        past_key_values[f"past_key_values.{i}.key"] = outputs[2 * i + 1].astype(np.float16)
        past_key_values[f"past_key_values.{i}.value"] = outputs[2 * i + 2].astype(np.float16)

    for step in range(max_new_tokens - 1):
        seq_length = past_key_values["past_key_values.0.key"].shape[2]
        input_feed = {
            "input_ids": np.array([[next_token]], dtype=np.int64),
            "attention_mask": np.ones((1, seq_length + 1), dtype=np.int64),
            "position_ids": np.array([[seq_length]], dtype=np.int64),
        }
        input_feed.update(past_key_values)

        outputs = session.run(None, input_feed)
        logits = outputs[0][:, -1, :].astype(np.float16)
        next_token = np.argmax(logits, axis=-1)[0]
        generated_ids.append(next_token)
        print(f"🟢 [STREAM] {tokenizer.decode([next_token])}", end='', flush=True)

        for i in range(num_hidden_layers):
            past_key_values[f"past_key_values.{i}.key"] = outputs[2 * i + 1].astype(np.float16)
            past_key_values[f"past_key_values.{i}.value"] = outputs[2 * i + 2].astype(np.float16)

        if next_token == tokenizer.eos_token_id:
            break
        gc.collect()

    print()
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

def chat():
    print(f"\n🤖 ONNX-Чат активен! Напиши что-нибудь ('выход' для выхода). {log_memory()}")
    while True:
        user_input = input("Ты: ")
        if user_input.lower() == "выход":
            print(f"🟢 [LOG] Чат завершён {log_memory()}")
            print(f"🤖 До встречи!")
            break

        input_feed, input_ids, attention_mask = preprocess_text(user_input)
        print(f"🟢 [LOG] Начинаем обработку запроса: '{user_input}' {log_memory()}")
        try:
            response_text = generate_text(input_feed, input_ids, attention_mask, max_new_tokens)
            print(f"🟢 [LOG] Ответ готов {log_memory()}")
            print(f"🤖 ONNX: {response_text}")
        except Exception as e:
            print(f"🟢 [LOG] Ошибка генерации: {e} {log_memory()}")

chat()