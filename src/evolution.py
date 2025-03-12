class Evolution:
    def __init__(self, veector):
        self.veector = veector
        self.evolution_strategy = "reason"  # Default strategy

    def evolve(self, tensor):
        """
        Выполняет эволюцию тензора в зависимости от выбранной стратегии.
        :param tensor: Тензор для эволюции.
        :return: Эволюционировавший тензор.
        """
        if self.evolution_strategy == "reason":
            return self._evolve_reason(tensor)
        elif self.evolution_strategy == "mutate":
            return self._evolve_mutate(tensor)
        else:
            raise ValueError(f"Неизвестная стратегия эволюции: {self.evolution_strategy}")

    def _evolve_reason(self, tensor):
        """
        Выполняет эволюцию тензора, используя операцию Reason.
        """
        # Выполняем операцию Reason
        evolved_tensor = self.veector.compute([
            tensor[0],
            [[0], tensor[0][1], [9, 0, 0], 1],  # Reason
            tensor[2],
            tensor[3],
            tensor[4] if len(tensor) > 4 else [],
            tensor[5] if len(tensor) > 5 else {}
        ])
        return evolved_tensor

    def _evolve_mutate(self, tensor, mutation_rate=0.1):
        """
        Выполняет эволюцию тензора путем случайной мутации.
        :param tensor: Тензор для мутации.
        :param mutation_rate: Вероятность мутации каждого элемента.
        :return: Мутировавший тензор.
        """
        mutated_data = tensor[0][2]
        if isinstance(mutated_data, np.ndarray):
            mask = np.random.rand(*mutated_data.shape) < mutation_rate
            mutation = np.random.normal(size=mutated_data.shape)
            mutated_data = np.where(mask, mutated_data + mutation, mutated_data)
        elif isinstance(mutated_data, list):
             mutated_data = [x + np.random.normal(0, 0.1) if np.random.rand() < mutation_rate else x for x in mutated_data]
        
        evolved_tensor = [
            tensor[0],
            tensor[1],
            mutated_data,
            tensor[3],
            tensor[4] if len(tensor) > 4 else [],
            tensor[5] if len(tensor) > 5 else {}
        ]
        return evolved_tensor
        
    def log_evolution(self, tensor):
        """
        Логирует процесс эволюции.
        :param tensor: Тензор для логирования.
        :return: Результат эволюции.
        """
        print(f"Начало эволюции для тензора: {tensor}")
        result = self.evolve(tensor)
        print(f"Результат эволюции: {result}")
        return result

    def set_evolution_strategy(self, strategy):
        """
        Устанавливает стратегию эволюции.
        :param strategy: Стратегия эволюции ("reason" или "mutate").
        """
        if strategy not in ["reason", "mutate"]:
            raise ValueError(f"Неподдерживаемая стратегия эволюции: {strategy}")
        self.evolution_strategy = strategy
