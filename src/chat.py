from llama_cpp import Llama
import sys
import time
import os

MODEL_PATH = "/workspaces/Veector/data/models/DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf"
N_THREADS = 4
N_CTX = 4096
N_BATCH = 512
TEMPERATURE = 0.3
TOP_K = 20
TOP_P = 0.9
MIN_P = 0.1
PROMPT_TEMPLATE = "<｜User｜>{message}<｜Assistant｜>"

def format_size(bytes_size):
    if bytes_size >= 1024**3:
        return f"{bytes_size / (1024**3):.2f} GiB"
    elif bytes_size >= 1024**2:
        return f"{bytes_size / (1024**2):.2f} MiB"
    else:
        return f"{bytes_size / 1024:.2f} KiB"

def print_model_info(llm):
    metadata = llm.metadata if hasattr(llm, 'metadata') else {}
    print("\n=== Информация о модели ===")
    print(f"Архитектура: {metadata.get('general.architecture', 'unknown')}")
    print(f"Название: {metadata.get('general.name', 'unknown')}")
    print(f"Размер модели: {metadata.get('general.size_label', 'unknown')}")
    print(f"Количество слоев: {int(metadata.get('qwen2.block_count', 0))}")
    print(f"Размер эмбеддингов: {int(metadata.get('qwen2.embedding_length', 1536))}")
    print(f"Размер FFN: {int(metadata.get('qwen2.feed_forward_length', 0))}")
    print(f"Количество голов внимания: {int(metadata.get('qwen2.attention.head_count', 0))}")
    print(f"Количество KV-голов: {int(metadata.get('qwen2.attention.head_count_kv', 0))}")
    print(f"Размер словаря: {len(metadata.get('tokenizer.ggml.tokens', []))}")
    print(f"Контекст (n_ctx): {N_CTX}")
    print(f"Тип квантизации: Q4_K (предположительно)")
    model_size = os.path.getsize(MODEL_PATH)
    kv_size = N_CTX * int(metadata.get('qwen2.embedding_length', 1536)) * 2 / 1024**2  # f16
    print(f"\n=== Использование памяти ===")
    print(f"Размер файла модели: {format_size(model_size)}")
    print(f"Оценочный размер KV-буфера (f16): {kv_size:.2f} MiB")
    print(f"Количество потоков: {N_THREADS}")

def main():
    print("Загрузка модели...")
    start_time = time.time()
    llm = Llama(model_path=MODEL_PATH, n_threads=N_THREADS, n_ctx=N_CTX, n_batch=N_BATCH, verbose=True)
    load_time = time.time() - start_time
    print(f"Модель загружена за {load_time:.2f} секунд!")
    print_model_info(llm)
    print("\nВведите сообщение (или 'exit' для выхода):")
    
    while True:
        user_input = input("Вы: ")
        if user_input.lower() == "exit":
            print("До свидания!")
            break
        
        prompt = PROMPT_TEMPLATE.format(message=user_input)
        print("Модель: ", end="", flush=True)
        response = ""
        start_time = time.time()
        for output in llm(prompt, max_tokens=-1, temperature=TEMPERATURE, top_k=TOP_K, top_p=TOP_P, 
                         min_p=MIN_P, stop=["<｜end▁of▁sentence｜>"], echo=False, stream=True):
            token = output["choices"][0]["text"]
            response += token
            print(token, end="", flush=True)
        gen_time = (time.time() - start_time) * 1000
        print("\n")
        print(f"Генерация завершена за {gen_time:.2f} мс")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nПрервано пользователем. До свидания!")
    except Exception as e:
        print(f"Ошибка: {str(e)}")
        sys.exit(1)