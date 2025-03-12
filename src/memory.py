import hashlib
import pickle
import lru  # pip install lru-cache

class Memory:
    def __init__(self, capacity=1000, use_lru_cache=True, use_hashing=True):
        """
        Инициализация Memory.
        :param capacity: Максимальное количество элементов в хранилище.
        :param use_lru_cache: Использовать LRU-кэш для повышения производительности.
        :param use_hashing: Использовать хеширование ключей для большей стабильности.
        """
        self.use_lru_cache = use_lru_cache
        self.use_hashing = use_hashing
        self.capacity = capacity
        
        if use_lru_cache:
            self.storage = lru.LRU(capacity) # Использовать LRU кэш
        else:
            self.storage = {}

    def _hash_key(self, key):
        """
        Генерирует хеш-сумму для ключа.
        """
        if isinstance(key, (tuple, list)):  # Преобразуем в кортеж
            key = tuple(key)
        if isinstance(key, np.ndarray):  # Преобразуем NumPy array в bytes
            key = key.tobytes()
        return hashlib.sha256(pickle.dumps(key)).hexdigest() # Hash serialized bytes

    def store(self, key, value):
        """
        Сохраняет значение в хранилище.
        :param key: Ключ для сохранения.
        :param value: Значение для сохранения.
        """
        if self.use_hashing:
            key = self._hash_key(key)
        if not self.use_lru_cache and len(self.storage) >= self.capacity:
             self._evict_oldest()

        self.storage[key] = value

    def retrieve(self, key):
        """
        Извлекает значение из хранилища.
        :param key: Ключ для извлечения.
        :return: Значение или None, если ключ не найден.
        """
        if self.use_hashing:
            key = self._hash_key(key)
        return self.storage.get(key)
    
    def _evict_oldest(self):
        """
        Удаляет самый старый элемент из хранилища.
        """
        if self.storage:
            oldest_key = next(iter(self.storage))  # Получаем ключ первого элемента
            del self.storage[oldest_key]

    def clear(self):
        """
        Очищает хранилище.
        """
        self.storage.clear()

    def __len__(self):
        """
        Возвращает количество элементов в хранилище.
        :return: Количество элементов.
        """
        return len(self.storage)

    def __contains__(self, key):
        """
        Проверяет наличие ключа в хранилище.
        :param key: Ключ для проверки.
        :return: True, если ключ есть в хранилище, иначе False.
        """
        if self.use_hashing:
            key = self._hash_key(key)
        return key in self.storage
    
class MemoryManager:
    def __init__(self, max_size=512):
        self.cache = {}
        self.access_times = {}
        self.max_size = max_size  # Максимальный размер кэша в МБ
    
    def add_block(self, block_name, block):
        if self._get_size() + block_size(block) > self.max_size:
            self._evict_lru()
        self.cache[block_name] = block
        self.access_times[block_name] = time.time()
    
    def _evict_lru(self):
        oldest_block = min(self.access_times.items(), key=lambda x: x[1])[0]
        del self.cache[oldest_block]
        del self.access_times[oldest_block]
