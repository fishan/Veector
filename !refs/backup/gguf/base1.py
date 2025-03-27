from model_manager import ModelManager
from core import Veector

# Инициализация
veector = Veector(use_memory=False, ipfs_enabled=False)
manager = ModelManager(veector, ipfs_enabled=False)

# Загрузка модели
manager.load_gguf_model(
    "DeepSeek-R1-Distill-Qwen-1.5B",
    "/workspaces/Veector/data/models/DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf"
)

# Выполнение инференса
output = manager.perform_inference("DeepSeek-R1-Distill-Qwen-1.5B", "Привет!")
print(f"Результат: {output}")