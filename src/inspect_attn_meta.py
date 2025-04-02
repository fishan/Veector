# inspect_attn_meta.py
import pickle
from pathlib import Path
import json # Для красивого вывода

# Импортируем TensorCoordinate
try:
    from tensors import TensorCoordinate, TENSORS_VERSION
    print(f"Using tensors.py v{TENSORS_VERSION}")
except ImportError:
    print("ERROR: Cannot import TensorCoordinate from tensors.py.")
    class TensorCoordinate:
        @classmethod
        def from_string(cls, s): return s
    print("Warning: Using dummy TensorCoordinate.")
except Exception as e:
    print(f"Error importing from tensors: {e}")
    exit()

# --- КОНФИГУРАЦИЯ ---
# ID процессора Attention Layer 0 из лога
ATTN_PROC_ID = "637fa4ff0945661a2373135c652dc8406ecef73c01fbf3884fb0d10196529c32"
# Путь к файлу метаданных (g500/l0/n0 - из лога процессора)
META_FILE_PATH = Path(f"../data/db/g500/l0/n0/{ATTN_PROC_ID}.meta")
# --- КОНЕЦ КОНФИГУРАЦИИ ---

print(f"\n--- Inspecting Attention Meta File: {META_FILE_PATH.resolve()} ---")

if not META_FILE_PATH.is_file():
    print(f"ERROR: Meta file not found at '{META_FILE_PATH}'")
else:
    try:
        print("Attempting to load structure with pickle...")
        with open(META_FILE_PATH, 'rb') as f:
            structure = pickle.load(f)
        print("Structure loaded successfully.")

        if isinstance(structure, list) and len(structure) >= 5:
            print("Structure is a list with expected length.")
            if isinstance(structure[0], str):
                 print(f"Coordinate is string: '{structure[0]}'. Attempting conversion...")
                 coord_obj = TensorCoordinate.from_string(structure[0])
                 if coord_obj and isinstance(coord_obj, TensorCoordinate): structure[0] = coord_obj; print("Coordinate converted to object.")
                 else: print("Warning: Could not parse coordinate string.")

            if isinstance(structure[4], dict):
                metadata = structure[4]
                print("\nOps Sequences from Metadata ('default' key):")
                ops_sequences = metadata.get('ops_sequences', {})
                default_sequence = ops_sequences.get('default', '!!! DEFAULT SEQUENCE NOT FOUND !!!')

                # Печатаем всю последовательность (если она небольшая) или начало
                print(json.dumps(default_sequence, indent=2, default=str))

                # Печатаем первую команду для анализа
                if isinstance(default_sequence, list) and default_sequence:
                     print("\nFirst command in 'default' sequence:")
                     first_command = default_sequence[0]
                     print(first_command)
                     print(f"Type of first command: {type(first_command)}")
                     if isinstance(first_command, list) and first_command:
                          print(f"Type of first element of first command: {type(first_command[0])}")
                else:
                     print("\n'default' sequence is not a non-empty list.")

            else: print(f"ERROR: Element at index 4 is not a dictionary.")
        else: print(f"ERROR: Loaded structure is not a list or has insufficient length.")

    except Exception as e:
        print(f"\n---!!! ERROR loading or processing the meta file: {e} !!!---")
        import traceback
        traceback.print_exc()

print("\n--- Inspection Finished ---")