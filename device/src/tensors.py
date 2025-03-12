def create_tensor(layer, coords, data, length, op=[1, 0, 0], next_coords=[], metadata=None):
    """Creates a tensor."""
    return [
        [layer, coords, data, length],
        [[0], coords, op, 1],
        [1, 0, 0],
        [0, 1, 0],
        next_coords,
        metadata or {}
    ]
