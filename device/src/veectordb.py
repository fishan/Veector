# device/src/veectordb.py
import json
from pathlib import Path

class VeectorDB:
    def __init__(self, db_path="data/db/user_data.json"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.db_path.exists():
            self.data = {"tensors": {}, "settings": {}}
            self._save()
        else:
            self._load()
        # Убеждаемся, что ключ "tensors" всегда существует
        if "tensors" not in self.data:
            self.data["tensors"] = {}
            self._save()

    def _load(self):
        try:
            with open(self.db_path, "r") as f:
                self.data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            # Если файл пуст или поврежден, инициализируем пустую структуру
            self.data = {"tensors": {}, "settings": {}}
            self._save()

    def _save(self):
        with open(self.db_path, "w") as f:
            json.dump(self.data, f, indent=4)

    def get_tensor_metadata(self, tensor_id):
        """Возвращает метаданные тензора (без данных)."""
        return self.data["tensors"].get(tensor_id, None)

    def save_tensor_metadata(self, tensor_id, metadata):
        """Сохраняет только метаданные тензора."""
        self.data["tensors"][tensor_id] = metadata
        self._save()

    def insert_tensor(self, tensor_id, metadata):
        """Добавляет метаданные тензора, созданного Veector."""
        self.data["tensors"][tensor_id] = metadata
        self._save()

    def get_setting(self, key, default=None):
        return self.data["settings"].get(key, default)

    def set_setting(self, key, value):
        self.data["settings"][key] = value
        self._save()