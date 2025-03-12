import numpy as np
from datetime import datetime  # Для метаданных

def create_tensor(layer, coords, data, length, op=[1, 0, 0], next_coords=[], metadata=None, version=1):
    """
    Создаёт тензор с заданными параметрами и метаданными, поддерживает комплексные числа.
    :param layer: Слой тензора (список).
    :param coords: Координаты тензора (список).
    :param data: Данные тензора (число, список или массив, включая комплексные числа).
    :param length: Длина данных тензора (число).
    :param op: Операция, применяемая к тензору (список).
    :param next_coords: Координаты следующего тензора (список).
    :param metadata: Дополнительные метаданные (словарь или None).
    :param version: Версия тензора (число).
    :return: Список, представляющий тензор.
    """
    if not isinstance(layer, list) or not isinstance(coords, list) or not isinstance(op, list):
        raise ValueError("Слой, координаты и операция должны быть списками.")

    if not isinstance(length, (int, float)):
        raise TypeError("Длина должна быть числом.")

    if not isinstance(next_coords, list):
        raise TypeError("Координаты следующего тензора должны быть списком.")

    # Преобразуем данные в np.array с поддержкой комплексных чисел
    if isinstance(data, (list, np.ndarray)):
        data = np.array(data, dtype=np.complex128)
    elif isinstance(data, (int, float, complex)):
        data = np.array([data], dtype=np.complex128)
    else:
        raise ValueError("Данные должны быть числом, списком или массивом.")

    # Проверяем, что длина соответствует данным
    if data.size != length:
        raise ValueError(f"Указанная длина {length} не соответствует размеру данных {data.size}")

    return [
        [list(map(int, layer)), list(map(int, coords)), data, length],
        [[0], list(map(int, coords)), list(map(int, op)), 1],
        [1, 0, 0],  # Контекст (по умолчанию)
        [0, 1, 0],  # Версия (по умолчанию)
        next_coords,
        {
            "version": version,
            "created_at": str(datetime.now()),
            "dtype": str(data.dtype),
            "shape": data.shape,
            **(metadata or {})
        }
    ]

def validate_tensor(tensor):
    """
    Проверяет валидность структуры тензора.
    :param tensor: Тензор для проверки.
    :return: True, если тензор валиден, иначе False.
    """
    if not isinstance(tensor, list):
        return False
    if len(tensor) < 4:
        return False
    if not all(isinstance(t, list) for t in tensor[:2]):
        return False
    if not isinstance(tensor[0][2], np.ndarray):  # Проверяем, что данные — это np.ndarray
        return False
    if not isinstance(tensor[0][3], (int, float)):  # Проверяем длину
        return False
    if len(tensor) > 5 and not isinstance(tensor[5], dict):  # Проверяем метаданные
        return False
    return True

def reshape_tensor(tensor, new_shape):
    """
    Изменяет форму данных в тензоре с проверкой объёма.
    :param tensor: Тензор для изменения формы.
    :param new_shape: Новая форма данных (кортеж или список).
    :return: Тензор с изменённой формой данных.
    """
    if not validate_tensor(tensor):
        raise ValueError("Невалидный тензор.")

    data = tensor[0][2]
    if data is None:
        raise ValueError("Данные тензора отсутствуют.")

    try:
        data = np.array(data, dtype=np.complex128)
        if np.prod(new_shape) != data.size:
            raise ValueError(f"Новая форма {new_shape} (объём {np.prod(new_shape)}) не соответствует объёму данных {data.size}")
        reshaped_data = data.reshape(new_shape)
        tensor[0][2] = reshaped_data
        tensor[5]["shape"] = reshaped_data.shape  # Обновляем метаданные
        return tensor
    except Exception as e:
        raise ValueError(f"Не удалось изменить форму тензора: {e}")

def get_tensor_metadata(tensor):
    """
    Получает метаданные тензора.
    :param tensor: Тензор для получения метаданных.
    :return: Метаданные тензора (словарь).
    """
    if not validate_tensor(tensor):
        raise ValueError("Невалидный тензор.")

    return tensor[5] if len(tensor) > 5 else {}

if __name__ == "__main__":
    # Пример использования
    # Создание тензора с комплексными числами
    tensor = create_tensor(
        layer=[0],
        coords=[0, 0, 0],
        data=[1 + 2j, 3 - 4j],
        length=2,
        op=[50, 0, 0],  # Квантовая операция Hadamard
        metadata={"description": "Тестовый тензор"}
    )
    print(f"Созданный тензор: {tensor[0][2]}")
    print(f"Метаданные: {get_tensor_metadata(tensor)}")

    # Проверка валидации
    print(f"Валидность тензора: {validate_tensor(tensor)}")

    # Изменение формы
    reshaped_tensor = reshape_tensor(tensor, (2, 1))
    print(f"Тензор после изменения формы: {reshaped_tensor[0][2]}")
    print(f"Обновлённые метаданные: {get_tensor_metadata(reshaped_tensor)}")