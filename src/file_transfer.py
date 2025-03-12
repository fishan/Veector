# src/file_transfer.py
import socket
import tqdm
import os
import hashlib
import time
import secrets  # Для генерации случайных ключей шифрования
from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
import base64

BUFFER_SIZE = 4096
SEPARATOR = "<SEPARATOR>"  # Используем уникальный разделитель
RESUME_POSITION_REQUEST = "<RESUME_POS>"  # Запрос позиции для возобновления передачи
KEY_LENGTH = 32  # Длина ключа шифрования (32 байта = 256 бит)

def generate_hash(file_path):
    """Генерирует хеш-сумму файла."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as file:
        while chunk := file.read(BUFFER_SIZE):
            hasher.update(chunk)
    return hasher.hexdigest()

def generate_key(password): # Добавлена функция generate_key
    password = password.encode()
    salt = os.urandom(16) # Generate a unique salt for each session
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=390000,
        backend=default_backend()
    )
    key = base64.urlsafe_b64encode(kdf.derive(password))
    return key, salt # Return the derived key and the salt

def encrypt_file(file_path, key):
    """Шифрует файл с использованием Fernet."""
    try:
        fernet = Fernet(key)
        with open(file_path, "rb") as file:
            data = file.read()
        encrypted_data = fernet.encrypt(data)
        return encrypted_data
    except Exception as e:
        print(f"Ошибка при шифровании файла: {e}")
        return None

def decrypt_file(encrypted_data, key):
     """Расшифровывает данные с использованием ключа Fernet."""
     try:
        fernet = Fernet(key)
        decrypted_data = fernet.decrypt(encrypted_data)
        return decrypted_data
     except InvalidToken:
        print("Invalid key - decryption failed")
        return None

def send_file(file_path, target_host, target_port, password=None):
    """
    Отправляет файл на указанный хост и порт с проверкой целостности,
    возможностью возобновления передачи и шифрованием.
    """
    try:
        filesize = os.path.getsize(file_path)
        filename = os.path.basename(file_path)  # Получаем только имя файла
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((target_host, target_port))
        print(f"Подключено к {target_host}:{target_port}, отправка {filename}...")

        # Генерируем хеш-сумму
        file_hash = generate_hash(file_path)

        # Шифруем файл, если указан пароль
        if password:
            key, salt = generate_key(password) # key generation added here
            encrypted_data = encrypt_file(file_path, key) # Encrypt_data now uses Key and Salt together to secure process
            if encrypted_data is None:
                s.close()
                return
            data_to_send = encrypted_data
            filesize = len(encrypted_data)  # Размер шифрованного файла
        else:
            with open(file_path, "rb") as file:
                data_to_send = file.read()

        # Отправляем имя файла, размер, хеш-сумму и наличие шифрования
        header = f"{filename}{SEPARATOR}{filesize}{SEPARATOR}{file_hash}{SEPARATOR}{password is not None}"
        s.send(header.encode())
        
        # Отправляем соль (если шифрование включено)
        if password:
            s.send(salt)

        # Отправляем файл с возможностью возобновления
        sent_bytes = 0
        progress = tqdm.tqdm(range(filesize), f"Отправка {filename}", unit="B", unit_scale=True, unit_divisor=1024)
        while sent_bytes < filesize:
            # Запрашиваем позицию для возобновления на стороне клиента
            s.send(RESUME_POSITION_REQUEST.encode())
            resume_position = int(s.recv(BUFFER_SIZE).decode())
            
            # Проверяем, нужно ли возобновлять
            if resume_position > sent_bytes:
                print(f"Возобновление отправки с позиции {resume_position}")
                sent_bytes = resume_position  # Обновляем sent_bytes
                progress.update(resume_position - progress.n)

            # Отправляем данные
            bytes_to_send = data_to_send[sent_bytes:sent_bytes + BUFFER_SIZE]
            s.sendall(bytes_to_send)
            sent_bytes += len(bytes_to_send)
            progress.update(len(bytes_to_send))

        s.close()
        print(f"{filename} успешно отправлен.")

    except Exception as e:
        print(f"Ошибка при отправке файла: {e}")

def receive_file(host, port, save_dir=".", password=None):
    """
    Принимает файл, сохраняя его в указанную директорию.
    """
    try:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind((host, port))
        s.listen(1)
        print(f"Ожидание входящего соединения на {host}:{port}...")

        conn, addr = s.accept()
        print(f"Подключено к {addr[0]}:{addr[1]}")

        # Получаем имя файла, размер, хеш-сумму и наличие шифрования
        received = conn.recv(BUFFER_SIZE).decode()
        filename, filesize, file_hash, is_encrypted = received.split(SEPARATOR)
        filename = os.path.basename(filename)  # Извлекаем имя файла из пути
        filesize = int(filesize)
        is_encrypted = is_encrypted.lower() == "true"

        # Получаем соль, если шифрование включено
        salt = None
        if is_encrypted:
            salt = conn.recv(16) # Corrected salt size
           
        file_path = os.path.join(save_dir, filename)
        received_bytes = 0
        
        # Проверяем наличие файла и определяем позицию для возобновления
        if os.path.exists(file_path):
            received_bytes = os.path.getsize(file_path)
            print(f"Файл {filename} уже существует. Возобновление с позиции {received_bytes}.")
        
        # Отправляем клиенту позицию для возобновления
        conn.send(str(received_bytes).encode())

        # Получаем файл
        progress = tqdm.tqdm(range(filesize), f"Приём {filename}", unit="B", unit_scale=True, unit_divisor=1024)
        with open(file_path, "ab" if received_bytes > 0 else "wb") as file:
            while received_bytes < filesize:
                bytes_to_read = min(BUFFER_SIZE, filesize - received_bytes)
                bytes_read = conn.recv(bytes_to_read)
                if not bytes_read:
                    break
                file.write(bytes_read)
                received_bytes += len(bytes_read)
                progress.update(len(bytes_read))
        conn.close()
        s.close()

        # Расшифровываем, если необходимо
        if is_encrypted and password:
            # Generate key with the provided password and received salt
            key, _ = generate_key(password)
            
            # Read the encrypted data from the file
            with open(file_path, "rb") as f:
                encrypted_data = f.read()
            
            # Decrypt the data
            decrypted_data = decrypt_file(encrypted_data, key)
            
            if decrypted_data:
                # Save the decrypted data back to the file, overwriting the encrypted content
                with open(file_path, "wb") as f:
                    f.write(decrypted_data)
                print(f"{filename} успешно расшифрован.")
            else:
                print(f"Не удалось расшифровать {filename}.")

        # Проверяем хеш-сумму
        received_hash = generate_hash(file_path)
        if received_hash == file_hash:
            print(f"Хеш-сумма {filename} совпадает. Файл успешно получен.")
        else:
            print(f"Хеш-сумма {filename} не совпадает. Файл повреждён.")

    except Exception as e:
        print(f"Ошибка при получении файла: {e}")

# Дополнительные функции
def list_files(target_host, target_port):
    """
    Запрашивает список файлов с удаленного хоста.
    """
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((target_host, target_port))
        s.send("LIST".encode())  # Отправляем команду LIST

        # Получаем данные (список файлов)
        data = s.recv(4096).decode()
        print("Список файлов на удаленном хосте:")
        print(data)
        s.close()
    except Exception as e:
        print(f"Ошибка при получении списка файлов: {e}")

def respond_to_list_request(host, port, directory="."):
    """
    Отвечает на запрос списка файлов.
    """
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind((host, port))
        s.listen(1)
        print(f"Ожидание запроса списка файлов на {host}:{port}...")

        conn, addr = s.accept()
        with conn:
            data = conn.recv(4096).decode()
            if data == "LIST":
                files = os.listdir(directory)
                conn.send("\n".join(files).encode())
                print("Список файлов отправлен.")
            else:
                print("Неизвестный запрос.")
        s.close()
    except Exception as e:
        print(f"Ошибка при отправке списка файлов: {e}")

# Пример использования
if __name__ == "__main__":
    # Режим отправки
    # send_file("large_file.zip", "127.0.0.1", 5001, password="mysecretpassword")

    # Режим приёма
    # receive_file("127.0.0.1", 5001, "received_files", password="mysecretpassword")

    # Запрос списка файлов
    # list_files("127.0.0.1", 5001)

    # Ответ на запрос списка файлов (запускаем в отдельном терминале)
    # respond_to_list_request("127.0.0.1", 5001)

    print("Закомментируйте или раскомментируйте нужные строки для запуска в нужном режиме.")
