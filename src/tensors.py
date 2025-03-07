def create_tensor(layer, coords, data, length, op=[1, 0, 0], next_coords=[]):
    """
    Создает тензор с заданными параметрами.
    """
    return [
        [list(layer), list(coords), data, length],
        [[0], list(coords), list(op), 1],
        [1, 0, 0],
        [0, 1, 0],
        next_coords
    ]


def validate_tensor(tensor):
    """
    Проверяет валидность структуры тензора.
    """
    return isinstance(tensor, list) and len(tensor) >= 4 and all(isinstance(t, list) for t in tensor[:2])


def reshape_tensor(tensor, new_shape):
    """
    Изменяет форму данных в тензоре.
    """
    if not validate_tensor(tensor):
        raise ValueError("Невалидный тензор.")

    data = np.array(tensor[0][2])
    reshaped_data = data.reshape(new_shape)
    tensor[0][2] = reshaped_data.tolist()
    return tensor
