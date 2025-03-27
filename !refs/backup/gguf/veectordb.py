import json
import os
import hashlib
from datetime import datetime
import numpy as np
import uuid # Import the UUID module

class VeectorDB:
    def __init__(self, db_path="../data/db/veectordb.json"):
        self.db_path = db_path
        self.data = {}
        self.load_db()
        self.id_namespace = uuid.uuid4()

    def load_db(self):
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, "r") as f:
                    self.data = json.load(f)
            except (json.JSONDecodeError, ValueError) as e:
                print(f"Ошибка загрузки {self.db_path}: {e}. Создаём новый файл.")
                self.data = {}
                self.save_db()
        else:
            self.data = {}
            self.save_db()

    def save_db(self):
        with open(self.db_path, "w") as f:
            json.dump(self.data, f, default=self._numpy_serializer, indent=4) # Added indent

    def _numpy_serializer(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")
    
    def generate_id(self, data):
        """
        Generates a unique ID using UUID.
        """
        combined_data = str(data) + str(self.id_namespace)
        return hashlib.sha256(combined_data.encode()).hexdigest()

    def insert_model(self, model_name, metadata):
        """
        Добавляет метаданные модели.
        :param model_name: Название модели.
        :param metadata: Метаданные модели (например, vocab_size, hidden_size, num_layers).
        """
        model_id = self.generate_id(model_name)
        self.insert("model", model_name, metadata={"model_id": model_id, **metadata})
        return model_id

    def insert(self, doc_type, data, metadata=None):
        """
        Вставляет документ в базу данных.
        :param doc_type: Тип документа (например, "model", "tensor", "metadata").
        :param data: Данные для сохранения (например, ID тензора или метаданные).
        :param metadata: Дополнительные метаданные.
        """
        doc_id = self.generate_id(data)
        doc = {
            "id": doc_id,
            "type": doc_type,
            "data": data,
            "metadata": metadata or {"timestamp": str(datetime.now())},
            "version": 1,  # Initial version
            "history": [] # Track previous versions
        }
        self.data[doc_id] = doc
        self.save_db()
        return doc_id

    def get(self, doc_id):
        """
        Получает документ по его ID.
        """
        return self.data.get(doc_id)

    def update(self, doc_id, new_data):
        """
        Обновляет данные документа по его ID и создает новую версию.
        """
        if doc_id in self.data:
            current_doc = self.data[doc_id]
            current_version = current_doc["version"]
            new_version = current_version + 1
            
            new_doc_id = self.generate_id(new_data)

            # Store history
            history_entry = {
                "id": current_doc["id"],
                "version": current_doc["version"],
                "timestamp": str(datetime.now()),
                "data": current_doc["data"],
                "metadata": current_doc["metadata"]
            }
            
            current_doc["history"].append(history_entry) # Added new history

            new_doc = {
                "id": new_doc_id,
                "type": current_doc["type"],
                "data": new_data,
                "metadata": {"timestamp": str(datetime.now()), **current_doc["metadata"]},
                "version": new_version,
                "history": [] # no history
            }
            
            self.data[new_doc_id] = new_doc # Use new_doc_id
            self.save_db()
        else:
             print(f"Document with id {doc_id} not found. Can not update.")

    def delete(self, doc_id):
        """
        Удаляет документ по его ID.
        """
        if doc_id in self.data:
            del self.data[doc_id]
            self.save_db()

    def sync(self, peer_db):
        """
        Синхронизирует базу данных с другой базой данных.
        """
        for doc_id, doc in peer_db.data.items():
            if doc_id not in self.data or \
               doc["metadata"]["timestamp"] > self.data[doc_id]["metadata"]["timestamp"]:
                self.data[doc_id] = doc
        self.save_db()

    def sync_shared(self, peer_db):
        """
        Синхронизирует только общие записи с другой базой данных.
        """
        for doc_id, doc in peer_db.data.items():
            if doc["type"] == "tensor_result" and doc["metadata"].get("shared", False):
                if doc_id not in self.data or \
                   doc["metadata"]["timestamp"] > self.data[doc_id]["metadata"]["timestamp"]:
                    self.data[doc_id] = doc
        self.save_db()

    def find_by_type(self, doc_type):
        """
        Возвращает список всех документов заданного типа.
        """
        return [doc for doc in self.data.values() if doc["type"] == doc_type]

    def find_by_metadata(self, key, value):
        """
        Ищет документы по ключу и значению в метаданных.
        """
        return [doc for doc in self.data.values() if doc["metadata"].get(key) == value]

    def insert_model(self, model_name, metadata):
        """
        Добавляет метаданные модели.
        :param model_name: Название модели.
        :param metadata: Метаданные модели (например, путь к файлу, IPFS hash).
        """
        model_id = self.generate_id(model_name)
        self.insert("model", model_name, metadata={"model_id": model_id, **metadata})
        return model_id

    def get_model_metadata(self, model_name):
        """
        Получает метаданные модели по её названию.
        :param model_name: Название модели.
        :return: Метаданные модели или None, если модель не найдена.
        """
        models = self.find_by_type("model")
        for model in models:
            if model["data"] == model_name:
                return model["metadata"]
        return None

    def insert_tensor(self, tensor_id, metadata):
         """
         Добавляет метаданные тензора.
         :param tensor_id: ID тензора (например, IPFS hash или путь к файлу).
         :param metadata: Метаданные тензора (например, shape, dtype).
         """
         self.insert("tensor", tensor_id, metadata)

    def get_tensor_metadata(self, tensor_id):
        """
        Получает метаданные тензора по его ID.
        :param tensor_id: ID тензора.
        :return: Метаданные тензора или None, если тензор не найден.
        """
        tensors = self.find_by_type("tensor")
        for tensor in tensors:
            if tensor["data"] == tensor_id:
                return tensor["metadata"]
        return None
    
    def get_version_history(self, doc_id):
        """
        Retrieves the version history for a given document ID.
        :param doc_id: The ID of the document.
        :return: A list of historical versions, or None if the document is not found.
        """
        doc = self.get(doc_id)
        if doc:
            return doc.get("history", [])
        else:
            return None
