# src/file_transfer.py
import socket

def send_file(file_path, target_host, target_port):
    with open(file_path, "rb") as file:
        file_data = file.read()
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((target_host, target_port))
            s.sendall(file_data)

def receive_file(file_path, host, port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, port))
        s.listen()
        conn, addr = s.accept()
        with conn:
            with open(file_path, "wb") as file:
                file.write(conn.recv(1024))

