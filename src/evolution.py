class Evolution:
    def __init__(self, veector):
        self.veector = veector

    def evolve(self, tensor, veector):  # Передаем veector в метод
        """
        Выполняет эволюцию тензора, используя операцию Reason.
        """
        # Выполняем операцию Reason
        evolved_tensor = veector.compute([
            tensor[0],
            [[0], tensor[0][1], [9, 0, 0], 1],  # Reason
            tensor[2],
            tensor[3],
            []
        ])
        return evolved_tensor

    def log_evolution(self, tensor, veector):  # Передаем veector в метод
        """
        Логирует процесс эволюции.
        """
        print(f"Начало эволюции для тензора: {tensor}")
        result = self.evolve(tensor, veector)
        print(f"Результат эволюции: {result}")
        return result
