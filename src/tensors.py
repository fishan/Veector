def create_tensor(layer, coords, data, length, op=[1, 0, 0], next_coords=[]):
    return [
        [list(layer), list(coords), data, length],
        [[0], list(coords), list(op), 1],
        [1, 0, 0],
        [0, 1, 0],
        next_coords
    ]

def validate_tensor(tensor):
    return isinstance(tensor, list) and len(tensor) >= 4 and all(isinstance(t, list) for t in tensor[:2])