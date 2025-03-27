from llama_cpp import Llama

try:
    llama = Llama(
        model_path="/workspaces/Veector/data/blocks/DeepSeek-R1-Distill-Qwen-1.5B/DeepSeek-R1-Distill-Qwen-1.5B-split-00001-of-00043.gguf",
        n_ctx=2048,
        n_gpu_layers=0,
        verbose=False
    )
    weights = llama.get_embedding_weights()
    print(f"Форма: {weights.shape}, Размер: {weights.nbytes / (1024**2):.2f} MB")
except Exception as e:
    print(f"Ошибка: {e}")

