def human_readable(tensor):
    if not isinstance(tensor, list) or len(tensor) < 4:
        return str(tensor)
    layer, coords, data, length = tensor[0]
    op = tensor[1][2]
    next_coords = tensor[4]
    return f"Layer: {layer}, Coords: {coords}, Data: {data}, Op: {op}, Next: {next_coords}"