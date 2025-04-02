# /workspaces/Veector/device/src/memory.py
import hashlib
import pickle
import numpy as np
import time
from collections import OrderedDict  # Для реализации LRU вручную

# --- Version ---
MEMORY_VERSION = "0.1.0"
# --- End Version ---

class Memory:
    def __init__(self, capacity=1000, use_lru_cache=True, use_hashing=True):
        self.use_lru_cache = use_lru_cache
        self.use_hashing = use_hashing
        self.capacity = capacity
        
        if use_lru_cache:
            self.storage = OrderedDict()  # Используем OrderedDict для LRU
        else:
            self.storage = {}

    def _hash_key(self, key):
        if isinstance(key, (tuple, list)):
            key = tuple(key)
        if isinstance(key, np.ndarray):
            key = key.tobytes()
        return hashlib.sha256(pickle.dumps(key)).hexdigest()

    def store(self, key, value):
        if self.use_hashing:
            key = self._hash_key(key)
        if self.use_lru_cache and len(self.storage) >= self.capacity:
            self.storage.popitem(last=False)  # Удаляем самый старый элемент
        elif not self.use_lru_cache and len(self.storage) >= self.capacity:
            self._evict_oldest()
        self.storage[key] = value
        if self.use_lru_cache:
            self.storage.move_to_end(key)  # Перемещаем в конец для LRU

    def retrieve(self, key):
        if self.use_hashing:
            key = self._hash_key(key)
        if key in self.storage:
            if self.use_lru_cache:
                self.storage.move_to_end(key)  # Обновляем порядок для LRU
            return self.storage[key]
        return None
    
    def _evict_oldest(self):
        if self.storage:
            oldest_key = next(iter(self.storage))
            del self.storage[oldest_key]

    def clear(self):
        self.storage.clear()

    def __len__(self):
        return len(self.storage)

    def __contains__(self, key):
        if self.use_hashing:
            key = self._hash_key(key)
        return key in self.storage
    
class MemoryManager:
    def __init__(self, max_size=512):
        self.cache = {}
        self.access_times = {}
        self.max_size = max_size
    
    def add_block(self, block_name, block):
        if self._get_size() + block_size(block) > self.max_size:
            self._evict_lru()
        self.cache[block_name] = block
        self.access_times[block_name] = time.time()
    
    def _evict_lru(self):
        oldest_block = min(self.access_times.items(), key=lambda x: x[1])[0]
        del self.cache[oldest_block]
        del self.access_times[oldest_block]

    def _get_size(self):
        return sum([block.nbytes / (1024 * 1024) if isinstance(block, np.ndarray) else 0 for block in self.cache.values()])

def block_size(block):
    return block.nbytes / (1024 * 1024) if isinstance(block, np.ndarray) else 0