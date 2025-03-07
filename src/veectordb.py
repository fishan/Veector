import json
import os
import hashlib
from datetime import datetime
import numpy as np

class VeectorDB:
    def __init__(self, db_path="vectordb.json"):
        self.db_path = db_path
        self.data = {}
        self.load_db()

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
            json.dump(self.data, f, default=self._numpy_serializer)

    def _numpy_serializer(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

    def generate_id(self, data):
        return hashlib.sha256(str(data).encode()).hexdigest()

    def insert(self, doc_type, data, metadata=None):
        doc_id = self.generate_id(data)
        doc = {
            "id": doc_id,
            "type": doc_type,
            "data": data,
            "metadata": metadata or {"timestamp": str(datetime.now())}
        }
        self.data[doc_id] = doc
        self.save_db()
        return doc_id

    def get(self, doc_id):
        return self.data.get(doc_id)

    def update(self, doc_id, new_data):
        if doc_id in self.data:
            self.data[doc_id]["data"] = new_data
            self.data[doc_id]["metadata"]["timestamp"] = str(datetime.now())
            self.save_db()

    def delete(self, doc_id):
        if doc_id in self.data:
            del self.data[doc_id]
            self.save_db()

    def sync(self, peer_db):
        for doc_id, doc in peer_db.data.items():
            if doc_id not in self.data or \
               doc["metadata"]["timestamp"] > self.data[doc_id]["metadata"]["timestamp"]:
                self.data[doc_id] = doc
        self.save_db()

    def sync_shared(self, peer_db):
        for doc_id, doc in peer_db.data.items():
            if doc["type"] == "tensor_result" and doc["metadata"].get("shared", False):
                if doc_id not in self.data or \
                   doc["metadata"]["timestamp"] > self.data[doc_id]["metadata"]["timestamp"]:
                    self.data[doc_id] = doc
        self.save_db()

    def find_by_type(self, doc_type):
        """Возвращает список всех записей заданного типа"""
        return [doc for doc in self.data.values() if doc["type"] == doc_type]
    
    def find_by_type(self, doc_type):
        return [doc for doc in self.data.values() if doc["type"] == doc_type]

    def find_by_metadata(self, key, value):
        """Поиск записей по ключу и значению в метаданных"""
        return [doc for doc in self.data.values() if doc["metadata"].get(key) == value]