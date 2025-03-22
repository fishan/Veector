#!/bin/bash

# Создание структуры директорий
mkdir -p src tests examples data/datasets/arithmetic data/pretrained docs

# Создание базовых файлов
touch src/core.py src/virtual_space.py src/operations.py src/tensors.py src/evolution.py src/memory.py src/interface.py
touch tests/test_basic.py tests/test_logic.py tests/test_loops.py tests/test_system.py tests/test_parallel.py
touch examples/calculator.vtr examples/advanced.py
touch data/pretrained/core.vtr data/pretrained/evolved.vtr
touch docs/README.md docs/veector.md docs/memory.md docs/training.md docs/roadmap.md
touch requirements.txt LICENSE

# Заполнение requirements.txt
echo "numpy" > requirements.txt

# Заполнение README.md
cat <<EOL > docs/README.md
# Veector

Veector — векторный язык и структура для децентрализованных ИИ и нейронных сетей.  
Создан [Твой ник] и Grok 3 от xAI, март 2025.

## Установка
1. Клонируйте репозиторий: \`git clone https://github.com/<твой_ник>/Veector.git\`  
2. Установите зависимости: \`pip install -r requirements.txt\`

## Использование
Смотрите примеры в \`examples/\`.

## Документация
- Спецификация: \`docs/veector.md\`  
- План: \`docs/roadmap.md\`
EOL

# Добавление LICENSE (MIT)
cat <<EOL > LICENSE
MIT License

Copyright (c) 2025 [Твой ник], xAI

Permission is hereby granted, free of charge, to any person obtaining a copy...
[Остальной текст MIT лицензии опущен для краткости, добавлю в репо]
EOL

# Базовый src/core.py
cat <<EOL > src/core.py
import numpy as np

class Veector:
    def __init__(self):
        self.core = {
            (1, 0, 0): np.array([[1, 1]]),  # Сложение
            (1, 1, 1): np.array([[1, -1]]), # Вычитание
            (0, 1, 0): np.array([[0, 1], [1, 0]]),  # Умножение
            (0, 0, 1): lambda x: x[0] / x[1],  # Деление
            (1, 0, 1): lambda x: np.sqrt(x[0])  # Корень
        }

    def compute(self, tensor):
        data = np.array(tensor[0][2])
        op = tuple(tensor[1][2])
        op_func = self.core.get(op, lambda x: x)
        result = op_func(data) if callable(op_func) else np.dot(op_func, data)
        return result

if __name__ == "__main__":
    v = Veector()
    tensor = [[[0], [0, 0, 0], [5, 3], 2], [[0], [0, 0, 0], [1, 0, 0], 1], [1, 0, 0], [0, 1, 0], []]
    print(v.compute(tensor))  # [8]
EOL

# Базовый examples/calculator.vtr
cat <<EOL > examples/calculator.vtr
# Калькулятор Veector
# Ввод: "5 + 3"
[
  [[0], [0, 0, 0], [5, 3], 2],      # Данные
  [[0], [0, 0, 0], [1, 0, 0], 1],   # Сложение
  [1, 0, 0],                         # Контекст
  [0, 1, 0],                         # Версия
  [[0], [1, 0, 0]]                   # Переход к выводу
]
EOL

# Делаем скрипт исполняемым
chmod +x setup_project.sh

echo "Структура проекта создана! Запустите './setup_project.sh' в Codespaces."