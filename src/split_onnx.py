from pathlib import Path
import logging
import onnx
from onnx.external_data_helper import convert_model_to_external_data

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def split_onnx_to_external_data(onnx_path, block_dir, model_name, split_size=4 * 1024 * 1024):
    block_dir = Path(block_dir) / model_name
    block_dir.mkdir(parents=True, exist_ok=True)

    # Загружаем модель
    model = onnx.load(onnx_path)

    # Конвертируем в формат с внешними данными
    convert_model_to_external_data(
        model,
        all_tensors_to_one_file=False,  # Каждый тензор в отдельном файле
        size_threshold=split_size,      # Минимальный размер тензора для внешнего хранения
        convert_attribute=False         # Не трогаем атрибуты
    )

    # Сохраняем модель с внешними файлами
    output_path = block_dir / f"{model_name}_split.onnx"
    onnx.save_model(model, str(output_path), save_as_external_data=True)
    logger.info(f"Модель сохранена с внешними данными в {output_path}")

    # Создаём метаданные для совместимости с VirtualSpace
    metadata = {}
    external_files = list(block_dir.glob("*.bin"))  # Файлы с весами
    for i, file_path in enumerate(external_files):
        metadata[f"part_{i}"] = {
            "file": file_path.name,
            "offset": 0,  # Offset внутри файла не нужен, ONNX Runtime сам разберётся
            "size": file_path.stat().st_size,
            "model_name": model_name,
            "coords": [i, 0]
        }
        logger.info(f"Внешний файл: {file_path}")

    with open(block_dir / "metadata.json", "w") as f:
        import json
        json.dump(metadata, f)

if __name__ == "__main__":
    onnx_path = "model.onnx"  # Скачай с Hugging Face
    block_dir = "../data/blocks"
    model_name = "DeepSeek-R1-Distill-Qwen-1.5B-ONNX"
    split_onnx_to_external_data(onnx_path, block_dir, model_name)