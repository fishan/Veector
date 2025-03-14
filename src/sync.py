import socket
import threading
from ipfshttpclient import connect
import pickle
import numpy as np
import torch
import time
import random
from pathlib import Path
from utils import parse_block_name

class P2PNode:
    def __init__(self, host, port, use_ipfs=True):
        self.host = host
        self.port = port
        self.peers = []
        self.use_ipfs = use_ipfs
        self.ipfs_client = connect() if use_ipfs else None
        self.known_tensors = set()  # Трекер известных тензоров
        self.block_map = {}  # Хранит хэши IPFS для блоков модели: {model_name: {(row, col): ipfs_hash}}

    def start(self):
        server_thread = threading.Thread(target=self._start_server)
        server_thread.daemon = True  # Завершается с главным процессом
        server_thread.start()

    def _start_server(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((self.host, self.port))
            s.listen()
            while True:
                conn, addr = s.accept()
                peer_thread = threading.Thread(target=self._handle_peer, args=(conn, addr))
                peer_thread.daemon = True
                peer_thread.start()

    def _handle_peer(self, conn, addr):
        with conn:
            try:
                data = conn.recv(4096)
                if data:
                    self._process_data(data)
            except Exception as e:
                print(f"Ошибка обработки соединения с {addr}: {e}")

    def connect_to_peer(self, peer_host, peer_port):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect((peer_host, peer_port))
                self.peers.append((peer_host, peer_port))
                print(f"Подключён к узлу: {peer_host}:{peer_port}")
        except Exception as e:
            print(f"Ошибка подключения к {peer_host}:{peer_port}: {e}")

    def send_data(self, data):
        for peer in self.peers:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.connect(peer)
                    serialized_data = pickle.dumps(data)
                    s.sendall(serialized_data)
                print(f"Данные отправлены узлу: {peer}")
            except Exception as e:
                print(f"Ошибка отправки данных узлу {peer}: {e}")

    def _process_data(self, data):
        try:
            received_data = pickle.loads(data)
            if isinstance(received_data, dict) and "tensor_id" in received_data:
                tensor_id = received_data["tensor_id"]
                metadata = received_data.get("metadata", {})

                if tensor_id not in self.known_tensors:
                    self.known_tensors.add(tensor_id)

                    if "ipfs_hash" in metadata:
                        ipfs_hash = metadata["ipfs_hash"]
                        shape = metadata.get("shape")
                        dtype = metadata.get("dtype", "float16")
                        model_name = metadata.get("model_name")
                        coords = metadata.get("coords")  # Координаты блока (row, col)

                        if shape and model_name and coords:
                            tensor_data = self._load_from_ipfs(ipfs_hash, shape, dtype)
                            if tensor_data is not None:
                                print(f"Получен блок модели {model_name} {coords}: {tensor_data.shape}")
                                if model_name not in self.block_map:
                                    self.block_map[model_name] = {}
                                self.block_map[model_name][coords] = ipfs_hash
                            else:
                                print(f"Не удалось загрузить блок {tensor_id} из IPFS")
                        else:
                            print(f"Недостаточно метаданных для {tensor_id}")
                    else:
                        print(f"Получен tensor_id без IPFS-хэша: {tensor_id}")
                else:
                    print(f"Тензор уже известен: {tensor_id}, пропускаем")
            else:
                print(f"Получены данные: {received_data}")
        except Exception as e:
            print(f"Ошибка обработки данных: {e}")

    def store_in_ipfs(self, tensor):
        """Сохранение тензора в IPFS."""
        if not self.use_ipfs:
            return None
        try:
            tensor_bytes = tensor.numpy().tobytes() if isinstance(tensor, torch.Tensor) else tensor.tobytes()
            ipfs_hash = self.ipfs_client.add_bytes(tensor_bytes)
            return ipfs_hash
        except Exception as e:
            print(f"Ошибка сохранения в IPFS: {e}")
            return None

    def _load_from_ipfs(self, ipfs_hash, shape, dtype="float16"):
        """Загрузка тензора из IPFS."""
        if not self.use_ipfs:
            return None
        try:
            tensor_data = self.ipfs_client.cat(ipfs_hash)
            np_dtype = np.dtype(dtype)
            tensor_np = np.frombuffer(tensor_data, dtype=np_dtype).reshape(shape)
            return torch.from_numpy(tensor_np)
        except Exception as e:
            print(f"Ошибка загрузки из IPFS: {e}")
            return None

    def sync_tensor(self, tensor, metadata):
        """Синхронизация тензора с другими узлами."""
        if self.use_ipfs:
            ipfs_hash = self.store_in_ipfs(tensor)
            if ipfs_hash:
                sync_data = {
                    "tensor_id": metadata.get("tensor_id", f"tensor_{random.randint(0, 10000)}"),
                    "metadata": {
                        "ipfs_hash": ipfs_hash,
                        "shape": tensor.shape,
                        "dtype": str(tensor.dtype),
                        **metadata
                    }
                }
                self.send_data(sync_data)
                print(f"Тензор синхронизирован с узлами (IPFS): {sync_data['tensor_id']}")
            else:
                print("Не удалось сохранить тензор в IPFS, синхронизация отменена")
        else:
            print("IPFS отключён, синхронизация пропущена")

    def sync_model_blocks(self, model_name, blocks_dir):
            """Синхронизация блоков модели из директории."""
            if not self.use_ipfs:
                print("IPFS отключён, синхронизация блоков невозможна")
                return
            block_files = list(Path(blocks_dir).glob(f"{model_name}_row*_col*.pt"))
            if not block_files:
                print(f"Блоки для модели {model_name} не найдены в {blocks_dir}")
                return
            for block_file in block_files:
                parsed = parse_block_name(block_file.name)  # Используем функцию разбора имени
                coords = (parsed["row"], parsed["col"])
                block = torch.load(block_file, map_location="cpu")
                ipfs_hash = self.store_in_ipfs(block)
                if ipfs_hash:
                    if model_name not in self.block_map:
                        self.block_map[model_name] = {}
                    self.block_map[model_name][coords] = ipfs_hash
                    sync_data = {
                        "tensor_id": f"{model_name}_block_{coords[0]}_{coords[1]}",
                        "metadata": {
                            "ipfs_hash": ipfs_hash,
                            "shape": block.shape,
                            "dtype": str(block.dtype),
                            "model_name": model_name,
                            "coords": coords
                        }
                    }
                    self.send_data(sync_data)
                    print(f"Блок {block_file.name} синхронизирован: {ipfs_hash}")
                else:
                    print(f"Не удалось синхронизировать блок {block_file.name}")
                del block
                gc.collect()

if __name__ == "__main__":
    node = P2PNode("localhost", 5000, use_ipfs=True)
    node.start()
    node.connect_to_peer("localhost", 5001)
    node.sync_model_blocks("DeepSeek-R1-Distill-Qwen-1.5B", "data/blocks")