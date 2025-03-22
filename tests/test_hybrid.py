from core import Veector
from model_manager import ModelManager
from virtual_space import VirtualSpace
from sync import P2PNode
import numpy as np

# Инициализация P2PNode
p2p_node = P2PNode("localhost", 5000, use_ipfs=True)
p2p_node.start()

# Инициализация Veector, ModelManager и VirtualSpace
veector = Veector(db_path="user_data.json", use_neural_storage=True, cache_size=500, 
                  dropout_rate=0.2, use_memory=True, p2p_node=p2p_node)
model_manager = ModelManager(veector, ipfs_enabled=True, p2p_node=p2p_node)
virtual_space = VirtualSpace(veector, model_manager, use_ipfs=True)

# Добавление моделей (пример с DeepSeek и LLaMA)
weights_deepseek = [np.random.rand(512, 512) for _ in range(32)]  # Пример весов DeepSeek
weights_llama = [np.random.rand(512, 512) for _ in range(32)]     # Пример весов LLaMA

virtual_space.analyze_deepseek("deepseek-7b", weights_deepseek)
virtual_space.analyze_deepseek("llama-7b", weights_llama)

# Выполнение вывода с использованием DeepSeek
input_tensor = np.random.rand(1, 512)
tensor1 = [
    [[0], [0, 0, 0], input_tensor, 512],
    [[0], [0, 0, 0], [70, 0, 0], 1],  # self-attention
    [1, 0, 0],
    [0, 1, 0],
    [[0], [1, 0, 0]]
]
results = virtual_space.execute(tensor1, model_name="deepseek-7b")
print(f"Результаты DeepSeek: {results}")

# Создание гибридного пространства
hybrid_config = {
    "attention_weights": ("deepseek-7b", 0, [0, 0, 0]),
    "feed_forward": ("llama-7b", 1, [1, 0, 0]),
}
virtual_space.create_hybrid_space("hybrid-deepseek-llama", hybrid_config)

# Выполнение вывода с использованием гибридного пространства
results = virtual_space.execute(tensor1, model_name="hybrid-deepseek-llama")
print(f"Результаты гибридной модели: {results}")

# Исследование сгенерированных тензоров
results = virtual_space.explore_generated(model_name="deepseek-7b")
print(f"Исследованные тензоры DeepSeek: {results}")

# Тест синхронизации через P2P
p2p_node.connect_to_peer("localhost", 5001)  # Пример подключения к другому узлу