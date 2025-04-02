import pickle
from pathlib import Path
from typing import Dict, List, Any, Union
import json

try:
    from core import Veector
    from tensors import get_tensor_metadata, get_tensor_coord
    from veectordb import VeectorDB
    print("Core components imported successfully.")
    CORE_IMPORTS_OK = True
except ImportError as e:
    print(f"---!!! FATAL ERROR (ImportError) !!! ---")
    print(f"Could not import core components: {e}")
    CORE_IMPORTS_OK = False


def collect_tensor_metadata(db_path: Union[str, Path]) -> Dict[str, Dict[str, Any]]:
    """
    Collects metadata from all tensors in the VeectorDB and formats it for JSON output.

    Args:
        db_path: Path to the VeectorDB directory.

    Returns:
        A dictionary where keys are tensor IDs and values are dictionaries
        containing relevant metadata (coord, tags, interface, etc.).
    """

    if not CORE_IMPORTS_OK:
        print("Cannot collect metadata due to import errors.")
        return {}

    db = None
    try:
        db = VeectorDB(db_dir=db_path)
        print(f"VeectorDB opened at: {db.db_root_path}")
    except Exception as e:
        print(f"Error initializing VeectorDB: {e}")
        return {}

    all_tensors_metadata: Dict[str, Dict[str, Any]] = {}
    all_tensors = db.find_tensors()

    for tensor_id, tensor_structure in all_tensors.items():
        try:
            metadata = get_tensor_metadata(tensor_structure)
            coord = get_tensor_coord(tensor_structure)
            relevant_data: Dict[str, Any] = {
                "coord": str(coord) if coord else None,
                "tags": metadata.get("tags"),
                "interface": metadata.get("interface"),
                "ops_sequences": metadata.get("ops_sequences"),
                "has_blob_data": metadata.get("has_blob_data"),
                "shape": metadata.get("shape"),
                "dtype": metadata.get("dtype"),
                "original_name": metadata.get("original_name")
                # Add other metadata fields as needed
            }
            all_tensors_metadata[tensor_id] = relevant_data
        except Exception as e:
            print(f"Error processing tensor {tensor_id}: {e}")

    if db:
        db.close()
        print("VeectorDB connection closed.")

    return all_tensors_metadata


if __name__ == "__main__":
    # --- Configuration ---
    target_db_path = Path("../data/db")  # Путь к вашей базе данных

    # --- Collect Metadata ---
    tensor_metadata = collect_tensor_metadata(target_db_path)

    # --- Analyze Metadata (Example) ---
    if tensor_metadata:
        print(f"\nCollected metadata for {len(tensor_metadata)} tensors.")

        # --- Convert to JSON and Print ---
        try:
            json_output = json.dumps(tensor_metadata, indent=4)
            print("\n--- Metadata in JSON Format ---")
            print(json_output)
        except TypeError as e:
            print(f"TypeError during JSON serialization: {e}")
        except Exception as e:
            print(f"Error converting to JSON: {e}")

        # --- Save Metadata to JSON File (Optional) ---
        output_file = Path("tensor_metadata.json")
        try:
            with open(output_file, 'w') as f:
                json.dump(tensor_metadata, f, indent=4)
            print(f"\nMetadata saved to: {output_file}")
        except Exception as e:
            print(f"Error saving metadata to JSON file: {e}")
    else:
        print("No tensor metadata collected.")