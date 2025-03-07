class Memory:
    def __init__(self):
        self.storage = {}

    def store(self, key, value):
        self.storage[tuple([key])] = value  # Всегда преобразовываем в кортеж

    def retrieve(self, key):
        return self.storage.get(tuple([key]))  # Всегда преобразовываем в кортеж
