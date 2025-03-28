from model_handler import ModelHandler
import logging

logging.basicConfig(level=logging.DEBUG)

# Инициализация модели
handler = ModelHandler(
    model_name="DeepSeek-R1-Distill-Qwen-1.5B",
    tensor_dir="/workspaces/Veector/data/blocks/DeepSeek-R1-Distill-Qwen-1.5B",
    vocab_size=151936,
    hidden_size=1536,
    num_layers=28
)

# Чат
def chat():
    print("\n🤖 Чат с блочной моделью активен! Введи 'выход' для завершения.")
    while True:
        user_input = input("Ты: ")
        if user_input.lower() == "выход":
            print("🤖 Чат завершён.")
            handler.clear_memory()
            break

        print("🤖 Генерирую ответ...")
        try:
            input_ids = handler.preprocess_text(user_input)
            generated_ids, confidence = handler.generate(input_ids)  # Никаких параметров из чата
            response = handler.tokenizer.decode(generated_ids, skip_special_tokens=True)
            print(f"🤖: {response} (Confidence: {confidence:.4f})")
        except Exception as e:
            print(f"⚠️ Ошибка генерации: {e}")
            handler.clear_memory()

if __name__ == "__main__":
    chat()