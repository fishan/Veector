class Memory:
    def __init__(self):
        self.storage = {}

    def store(self, key, value):
        self.storage[tuple(key)] = value

    def retrieve(self, key):
        return self.storage.get(tuple(key))