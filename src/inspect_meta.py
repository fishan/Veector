# inspect_meta.py
import pickle
from pathlib import Path
import json

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
# ID тензора k_bias из логов
TENSOR_ID = "9e6aece757fdccb6cbea58369bc69c136ff4b48c60a0a60f02749d288c206586"
# Путь к файлу метаданных (g100, l0, n1 - из лога сохранения в Colab / или проверь структуру папок)
META_FILE_PATH = Path(f"../data/db/g100/l0/n1/{TENSOR_ID}.meta")
# --- КОНЕЦ КОНФИГУРАЦИИ ---

print(f"\n--- Inspecting Meta File: {META_FILE_PATH.resolve()} ---")

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
                print("\nFull Metadata Dictionary [4]:")
                try: print(json.dumps(metadata, indent=2, default=str))
                except Exception as json_e: print(f"(Error printing metadata: {json_e}). Raw:\n{metadata}")

                print("\nTags list from Metadata:")
                tags = metadata.get('tags', '!!! TAGS KEY NOT FOUND !!!')
                print(tags)

                # Проверка типов тегов
                if isinstance(tags, list):
                    print("\nChecking types within the tags list:")
                    all_tuples = True
                    for i, tag_item in enumerate(tags):
                        print(f"  Tag {i}: {tag_item} (Type: {type(tag_item)})")
                        if not isinstance(tag_item, tuple): all_tuples = False
                    if all_tuples: print("  All items in the tags list are tuples.")
                    else: print("  WARNING: Not all items in the tags list are tuples!")
                elif tags != '!!! TAGS KEY NOT FOUND !!!': print(f"  WARNING: Tags field is not a list! Type: {type(tags)}")

                # Проверка наличия ожидаемых тегов для k_bias
                print("\nChecking presence of expected tags for k_bias (as tuples):")
                # Теги: Bias(1,2), Layer0(4,0), QWEN2(3,1), FP16(2,16), AttnK(1,22), TypeKnowledge(0,2)
                expected_tags = {(1, 2), (4, 0), (3, 1), (2, 16), (1, 22), (0, 2)}
                if isinstance(tags, list):
                     tags_set = set(tuple(t) if isinstance(t, list) else t for t in tags)
                     all_found = True
                     for etag in expected_tags:
                          is_present = etag in tags_set
                          print(f"  Contains {etag}? {is_present}")
                          if not is_present: all_found = False
                     if all_found: print("  All expected tags seem present.")
                     else: print("  WARNING: Not all expected tags were found in the set!")
                else: print("  Cannot check specific tags because 'tags' is not a list.")

            else: print(f"ERROR: Element at index 4 is not a dictionary.")
        else: print(f"ERROR: Loaded structure is not a list or has insufficient length.")

    except ModuleNotFoundError as mnfe: print(f"\n---!!! ERROR loading pickle file: {mnfe} !!!---"); print("Check class definitions.")
    except Exception as e: print(f"\n---!!! ERROR loading or processing the meta file: {e} !!!---"); import traceback; traceback.print_exc()

print("\n--- Inspection Finished ---")