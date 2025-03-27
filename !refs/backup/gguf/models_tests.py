from llama_cpp import Llama
import sys
import time
import os
import psutil

MODEL_PATH = "/workspaces/Veector/data/models/DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf"
N_THREADS = 16
N_CTX = 4096
N_BATCH = 4096
TEMPERATURE = 0.5  # Уменьшил для точности
TOP_K = 100
TOP_P = 0.95
MIN_P = 0.05
PROMPT_TEMPLATE = "<｜User｜>{message}<｜Assistant｜>"

def format_size(bytes_size):
    if bytes_size >= 1024**3:
        return f"{bytes_size / (1024**3):.2f} GiB"
    elif bytes_size >= 1024**2:
        return f"{bytes_size / (1024**2):.2f} MiB"
    else:
        return f"{bytes_size / 1024:.2f} KiB"

def print_memory_usage(stage):
    mem = psutil.virtual_memory()
    print(f"{stage} - Использовано RAM: {format_size(mem.used)}, Свободно: {format_size(mem.available)}")

def print_model_info(llm):
    metadata = llm.metadata if hasattr(llm, 'metadata') else {}
    print("\n=== Информация о модели ===")
    print(f"Название: {metadata.get('general.name', 'unknown')}")
    print(f"Размер файла: {format_size(os.path.getsize(MODEL_PATH))}")
    print(f"Тип квантизации: {metadata.get('general.file_type', 'unknown')}")
    print(f"Контекст (n_ctx): {N_CTX}")

def run_test(llm, test_name, prompt, max_tokens=512):
    print(f"\n=== Тест: {test_name} ===")
    print_memory_usage("Перед генерацией")
    response = ""
    thinking = ""
    in_thinking = False
    start_time = time.time()
    
    print("Размышления модели: ", end="", flush=True)
    print("\nОтвет модели: ", end="", flush=True)
    
    for output in llm(prompt, max_tokens=max_tokens, temperature=TEMPERATURE, top_k=TOP_K, top_p=TOP_P, 
                     min_p=MIN_P, stop=["<｜end▁of▁sentence｜>"], echo=False, stream=True):
        token = output["choices"][0]["text"]
        response += token
        
        # Обработка токенов в реальном времени
        if "<think>" in token:
            in_thinking = True
            thinking += token.replace("<think>", "")
            print(token.replace("<think>", ""), end="", flush=True)
        elif "</think>" in token:
            in_thinking = False
            thinking += token.replace("</think>", "")
            print(token.replace("</think>", ""), end="", flush=True)
        elif in_thinking:
            thinking += token
            print(token, end="", flush=True)
        else:
            # Выводим основной ответ по словам
            print(token, end="", flush=True)
    
    # Убираем <think> из финального ответа
    response = response.replace("<think>", "").replace("</think>", "").strip()
    gen_time = (time.time() - start_time) * 1000
    print("\n")  # Переход на новую строку после генерации
    print_memory_usage("После генерации")
    print(f"Финальный ответ: {response}")
    print(f"Время генерации: {gen_time:.2f} мс")

def main():
    print("Загрузка модели...")
    print_memory_usage("До загрузки модели")
    start_time = time.time()
    llm = Llama(model_path=MODEL_PATH, n_threads=N_THREADS, n_ctx=N_CTX, n_batch=N_BATCH, verbose=True)
    load_time = time.time() - start_time
    print_memory_usage("После загрузки модели")
    print(f"Модель загружена за {load_time:.2f} секунд!")
    print_model_info(llm)

    # Тест 1: Короткий ответ
    short_prompt = PROMPT_TEMPLATE.format(message="Привет, как дела? Ответь коротко.")
    run_test(llm, "Короткий ответ", short_prompt, max_tokens=50)

    # Тест 2: Длинный контекст
    long_prompt_text = (
        "Зайцы — это млекопитающие из семейства зайцевых, которые обитают почти на всех континентах, кроме Австралии и Антарктиды. "
        "Они известны своими длинными ушами, сильными задними лапами и пушистым хвостом. Зайцы питаются травой, корой деревьев и иногда овощами. "
        "В отличие от кроликов, зайцы рождаются с шерстью и открытыми глазами, что делает их более самостоятельными с первых дней жизни. "
        "Они могут бегать со скоростью до 70 км/ч, что помогает им убегать от хищников, таких как лисы, волки и ястребы. "
        "Зимой некоторые виды зайцев, например, заяц-беляк, меняют цвет шерсти на белый, чтобы сливаться со снегом. "
        "Зайцы играют важную роль в экосистеме, являясь пищей для многих хищников, а также помогая распространять семена растений. "
        "В культуре зайцы часто встречаются в сказках и легендах, где их изображают как хитрых и быстрых персонажей. "
        "О чём этот текст? Ответь кратко и по-русски."
    )
    long_prompt = PROMPT_TEMPLATE.format(message=long_prompt_text)
    run_test(llm, "Длинный контекст", long_prompt, max_tokens=1000)

    # Тест 3: Творческая задача
    creative_prompt = PROMPT_TEMPLATE.format(message="Придумай сказку про зайца для ребёнка 4 лет. Начинай сразу с рассказа и по-русски.")
    run_test(llm, "Творческая задача", creative_prompt, max_tokens=1000)

    # Тест 4: Команда для помощника
    command_prompt = PROMPT_TEMPLATE.format(message="Запиши встречу на 15:00 с Васей. Подтверди запись. Ответь по-русски.")
    run_test(llm, "Команда для помощника", command_prompt, max_tokens=1000)

    print("\n=== Завершение тестов ===")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nПрервано пользователем.")
    except Exception as e:
        print(f"Ошибка: {str(e)}")
        sys.exit(1)