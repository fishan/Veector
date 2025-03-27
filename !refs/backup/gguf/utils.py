# /workspaces/Veector/src/utils.py

def parse_block_name(filename):
    """
    Разбирает имя файла блока на составляющие.
    :param filename: Полное имя файла (например, "DeepSeek-R1-Distill-Qwen-1.5B_row1691_col0.pt").
    :return: Словарь с моделью, row и col.
    """
    if not filename.endswith(".pt"):
        raise ValueError("Имя файла должно заканчиваться на .pt")
    base_name = filename[:-3]  # Удаляем ".pt"

    # Извлекаем col
    col_part = base_name.split("_")[-1]
    if not col_part.startswith("col"):
        raise ValueError(f"Некорректный формат col: {col_part}")
    col = int(col_part[3:])  # Удаляем "col"
    base_name = "_".join(base_name.split("_")[:-1])  # Удаляем "_colX"

    # Извлекаем row
    row_part = base_name.split("_")[-1]
    if not row_part.startswith("row"):
        raise ValueError(f"Некорректный формат row: {row_part}")
    row = int(row_part[3:])  # Удаляем "row"
    base_name = "_".join(base_name.split("_")[:-1])  # Удаляем "_rowX"

    # Оставшаяся часть — название модели
    model_name = base_name

    return {
        "model_name": model_name,
        "row": row,
        "col": col
    }