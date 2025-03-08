import socket
import threading
from ipfshttpclient import connect
import pickle

class P2PNode:
    def __init__(self, host, port, use_ipfs=True):
        self.host = host
        self.port = port
        self.peers = []
        self.use_ipfs = use_ipfs
        self.ipfs_client = connect() if use_ipfs else None

    def start(self):
        server_thread = threading.Thread(target=self._start_server)
        server_thread.start()

    def _start_server(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((self.host, self.port))
            s.listen()
            while True:
                conn, addr = s.accept()
                peer_thread = threading.Thread(target=self._handle_peer, args=(conn, addr))
                peer_thread.start()

    def _handle_peer(self, conn, addr):
        with conn:
            data = conn.recv(1024)
            if data:
                self._process_data(data)

    def connect_to_peer(self, peer_host, peer_port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((peer_host, peer_port))
            self.peers.append((peer_host, peer_port))

    def send_data(self, data):
        for peer in self.peers:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect(peer)
                s.sendall(pickle.dumps(data))

    def _process_data(self, data):
        data = pickle.loads(data)
        if isinstance(data, dict) and "ipfs_hash" in data:
            # Получаем тензор из IPFS
            tensor_data = self._load_from_ipfs(data["ipfs_hash"], data["shape"])
            print(f"Received tensor from peer: {tensor_data.shape}")
        else:
            print(f"Received data: {data}")

    def store_in_ipfs(self, tensor_np):
        """
        Сохраняет тензор в IPFS.
        :return: Хэш IPFS.
        """
        if not self.use_ipfs:
            return None
        ipfs_hash = self.ipfs_client.add_bytes(tensor_np.tobytes())
        return ipfs_hash

    def _load_from_ipfs(self, ipfs_hash, shape):
        """
        Загружает тензор из IPFS.
        :return: numpy массив.
        """
        if not self.use_ipfs:
            return None
        tensor_data = self.ipfs_client.cat(ipfs_hash)
        return np.frombuffer(tensor_data, dtype=np.float32).reshape(shape)

    def sync_tensor(self, tensor_np, metadata):
        """
        Синхронизирует тензор с другими узлами через IPFS.
        """
        ipfs_hash = self.store_in_ipfs(tensor_np)
        data = {"ipfs_hash": ipfs_hash, "shape": tensor_np.shape, "metadata": metadata}
        self.send_data(data)