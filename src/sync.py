# src/sync.py
import socket
import threading

class P2PNode:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.peers = []

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
                s.sendall(data.encode())

    def _process_data(self, data):
        # Обработка полученных данных
        print(f"Received data: {data.decode()}")