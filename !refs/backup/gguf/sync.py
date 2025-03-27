import socket
import threading
from ipfshttpclient import connect
import pickle
import numpy as np
import torch
import time
import random
from pathlib import Path
import gc
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def parse_block_name(block_name):
    # Пример имени: "model_row0_col1.pt" -> {"model": "model", "row": 0, "col": 1}
    parts = block_name.split("_")
    return {
        "model": parts[0],
        "row": int(parts[1][3:]),
        "col": int(parts[2][3:].replace(".pt", ""))
    }

class P2PNode:
    def __init__(self, host, port, use_ipfs=True):
        self.host = host
        self.port = port
        self.peers = []
        self.use_ipfs = use_ipfs
        self.ipfs_client = connect() if use_ipfs else None
        self.known_tensors = set()
        self.block_map = {}

    def start(self):
        server_thread = threading.Thread(target=self._start_server)
        server_thread.daemon = True
        server_thread.start()

    def _start_server(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((self.host, self.port))
            s.listen()
            logger.debug(f"P2P server started on {self.host}:{self.port}")
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
                logger.error(f"Error handling peer {addr}: {e}")

    def connect_to_peer(self, peer_host, peer_port):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect((peer_host, peer_port))
                self.peers.append((peer_host, peer_port))
                logger.info(f"Connected to peer: {peer_host}:{peer_port}")
        except Exception as e:
            logger.error(f"Error connecting to {peer_host}:{peer_port}: {e}")

    def send_data(self, data):
        for peer in self.peers:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.connect(peer)
                    serialized_data = pickle.dumps(data)
                    s.sendall(serialized_data)
                logger.debug(f"Data sent to peer: {peer}")
            except Exception as e:
                logger.error(f"Error sending data to {peer}: {e}")

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
                        coords = metadata.get("coords")

                        if shape and model_name and coords:
                            tensor_data = self._load_from_ipfs(ipfs_hash, shape, dtype)
                            if tensor_data is not None:
                                logger.info(f"Received block {model_name} {coords}: {tensor_data.shape}")
                                if model_name not in self.block_map:
                                    self.block_map[model_name] = {}
                                self.block_map[model_name][coords] = ipfs_hash
                            else:
                                logger.error(f"Failed to load block {tensor_id} from IPFS")
                        else:
                            logger.warning(f"Missing metadata for {tensor_id}")
                    else:
                        logger.debug(f"Received tensor_id without IPFS hash: {tensor_id}")
                else:
                    logger.debug(f"Tensor already known: {tensor_id}, skipping")
            else:
                logger.debug(f"Received data: {received_data}")
        except Exception as e:
            logger.error(f"Error processing data: {e}")

    def store_in_ipfs(self, tensor):
        if not self.use_ipfs:
            return None
        try:
            tensor_bytes = tensor.numpy().tobytes() if isinstance(tensor, torch.Tensor) else tensor.tobytes()
            ipfs_hash = self.ipfs_client.add_bytes(tensor_bytes)
            logger.debug(f"Stored in IPFS: {ipfs_hash}")
            return ipfs_hash
        except Exception as e:
            logger.error(f"Error storing in IPFS: {e}")
            return None

    def _load_from_ipfs(self, ipfs_hash, shape, dtype="float16"):
        if not self.use_ipfs:
            return None
        try:
            tensor_data = self.ipfs_client.cat(ipfs_hash)
            np_dtype = np.dtype(dtype)
            tensor_np = np.frombuffer(tensor_data, dtype=np_dtype).reshape(shape)
            return torch.from_numpy(tensor_np)
        except Exception as e:
            logger.error(f"Error loading from IPFS: {e}")
            return None

    def sync_tensor(self, tensor, metadata):
        if self.use_ipfs:
            ipfs_hash = self.store_in_ipfs(tensor)
            if ipfs_hash:
                sync_data = {
                    "tensor_id": metadata.get("tensor_id", f"tensor_{random.randint(0, 10000)}"),
                    "metadata": {
                        "ipfs_hash": ipfs_hash,
                        "shape": list(tensor.shape),
                        "dtype": str(tensor.dtype),
                        **metadata
                    }
                }
                self.send_data(sync_data)
                logger.info(f"Tensor synced with peers (IPFS): {sync_data['tensor_id']}")
            else:
                logger.error("Failed to store tensor in IPFS, sync aborted")
        else:
            logger.debug("IPFS disabled, sync skipped")