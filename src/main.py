from core import Veector
from model_manager import ModelManager
from sync import P2PNode
import numpy as np

p2p_node = P2PNode(host="localhost", port=5000, use_ipfs=True)
veector = Veector(db_path="vectordb.json", p2p_node=p2p_node)
veector.model_manager = ModelManager(veector, ipfs_enabled=True, p2p_node=p2p_node)

p2p_node.start()

veector.model_manager.load_pre_split_model("deepseek-coder-1.3b", tensor_dir="../data/deepseek-ai/deepseek-coder-1.3b-base/")

prompt = "Сложи два числа и выведи результат"
program_tensors = veector.generate_program_tensor(prompt)
print("Сгенерирована программа:", [t[1][2] for t in program_tensors])

input_data = np.array([5, 3])
result = veector.execute_program(program_tensors, input_data)
print("Результат:", result)

hashes = veector.share_program(program_tensors)
print("Хэши IPFS:", hashes)

feedback = {"input": np.array([5, 3]), "target": np.array([8])}
improved_program = veector.improve_program(program_tensors, feedback)
print("Улучшенная программа:", [t[1][2] for t in improved_program])