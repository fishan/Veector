# FILE: veectordb.py
# English comments added for the community

import json
import os
import hashlib
from datetime import datetime
import numpy as np
import uuid
import pickle # Using pickle for robust serialization/deserialization
import zlib   # For compressing blob data
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union

# Import necessary components from tensors.py
try:
    from tensors import TensorCoordinate, validate_tensor, get_tensor_hash, get_tensor_metadata
except ImportError:
    print("Warning: Could not import from tensors.py. Ensure it's in the Python path.")
    # Define dummy versions if import fails, to allow basic DB functionality
    class TensorCoordinate: pass
    def validate_tensor(t): return isinstance(t, list)
    def get_tensor_hash(t): return hashlib.sha256(pickle.dumps(t)).hexdigest()
    def get_tensor_metadata(t): return t[4] if isinstance(t, list) and len(t)>4 and isinstance(t[4], dict) else {}


class VeectorDB:
    """
    Veector's custom database.
    - Stores metadata (including tensor structures without large data) in a main file (meta.vdb using pickle).
    - Stores large binary data (like NumPy arrays for knowledge tensors) in a 'blobs' subdirectory,
      compressed with zlib and identified by their SHA256 hash (CID).
    - Supports basic CRUD, archiving (status update), and querying.
    """
    def __init__(self, db_dir: Union[str, Path] = "../data/db"):
        self.db_dir = Path(db_dir)
        self.meta_path = self.db_dir / "meta.vdb" # Main metadata file (using pickle)
        self.blobs_dir = self.db_dir / "blobs"    # Directory for binary blobs
        self.db_dir.mkdir(parents=True, exist_ok=True)
        self.blobs_dir.mkdir(exist_ok=True)
        # self.data format: { doc_id (str): metadata_dict (dict) }
        # For tensors, metadata_dict contains the tensor structure (list) under the 'structure' key
        self.data: Dict[str, Dict] = {}
        self._load_db()

    def _load_db(self):
        """Loads the main metadata dictionary from the meta.vdb file."""
        if self.meta_path.exists():
            try:
                with open(self.meta_path, "rb") as f:
                    self.data = pickle.load(f)
                print(f"Database loaded from {self.meta_path}. Contains {len(self.data)} entries.")
            except Exception as e:
                print(f"Error loading {self.meta_path}: {e}. Creating new database.")
                self.data = {}
                self._save_db()
        else:
            print(f"Database file {self.meta_path} not found. Creating new database.")
            self.data = {}
            self._save_db()

    def _save_db(self):
        """Saves the main metadata dictionary to the meta.vdb file."""
        try:
            with open(self.meta_path, "wb") as f:
                pickle.dump(self.data, f)
            # print(f"Database saved to {self.meta_path}") # Optional: noisy
        except Exception as e:
            print(f"Error saving database to {self.meta_path}: {e}")

    def _store_blob(self, data: Any) -> str:
        """
        Serializes, compresses, and stores arbitrary data in the blobs directory.
        Returns the SHA256 hash (CID) of the compressed data.
        """
        try:
            serialized_data = pickle.dumps(data)
            compressed_data = zlib.compress(serialized_data)
            # Use SHA256 of compressed data as the content identifier (CID)
            cid = hashlib.sha256(compressed_data).hexdigest()
            blob_path = self.blobs_dir / cid
            if not blob_path.exists(): # Avoid rewriting identical blobs
                with open(blob_path, "wb") as f:
                    f.write(compressed_data)
            return cid
        except Exception as e:
             print(f"Error storing blob: {e}")
             raise # Re-raise the exception

    def _load_blob(self, cid: str) -> Any | None:
        """Loads, decompresses, and deserializes data from the blobs directory by CID."""
        blob_path = self.blobs_dir / cid
        if blob_path.exists():
            try:
                with open(blob_path, "rb") as f:
                    compressed_data = f.read()
                serialized_data = zlib.decompress(compressed_data)
                return pickle.loads(serialized_data)
            except Exception as e:
                print(f"Error loading/deserializing blob {cid}: {e}")
                return None
        else:
            print(f"Blob {cid} not found.")
            return None

    def _get_doc_metadata(self, doc_id: str) -> Optional[Dict]:
         """Safely gets the metadata dictionary for a document ID."""
         return self.data.get(doc_id)

    # --- Veector Tensor Specific Methods ---

    def insert_veector_tensor(self, tensor_structure: List) -> Optional[str]:
        """
        Stores a Veector tensor structure. Handles processor and knowledge types.
        For knowledge tensors, the data payload is stored as a blob.
        Returns the document ID (tensor hash) if successful, None otherwise.
        """
        if not validate_tensor(tensor_structure):
            print("Error: Attempting to insert invalid tensor structure.")
            return None

        doc_id = get_tensor_hash(tensor_structure)
        metadata = get_tensor_metadata(tensor_structure)
        tensor_type = metadata.get("tensor_type")

        if doc_id in self.data:
             # TODO: Decide on update strategy? Overwrite? Version conflict?
             # For now, just indicate it exists.
             # print(f"Info: Tensor {doc_id} already exists in DB.")
             return doc_id

        # Deep copy the structure to modify it for storage
        structure_to_store = pickle.loads(pickle.dumps(tensor_structure))
        doc_entry = {"type": "veector_tensor"} # Add type for easier querying

        if tensor_type == "knowledge":
            # Knowledge tensor: store data payload in blob
            if len(structure_to_store) == 6: # Data is temporarily at index [5]
                knowledge_data = structure_to_store.pop(5) # Remove data from structure list
                try:
                    blob_cid = self._store_blob(knowledge_data)
                    # Store blob CID within the main metadata section [4]
                    structure_to_store[4]["knowledge_blob_cid"] = blob_cid
                    print(f"Stored knowledge data for {doc_id} in blob {blob_cid}")
                except Exception as e:
                    print(f"Error storing knowledge blob for {doc_id}: {e}")
                    return None # Failed to store blob
            else:
                print(f"Warning: Knowledge tensor {doc_id} structure passed without data payload.")
                structure_to_store[4]["knowledge_blob_cid"] = None # Mark as no blob
        elif tensor_type == "processor":
            # Processor tensor: Ensure no large data payload is accidentally stored
             if len(structure_to_store) == 6:
                print(f"Warning: Processor tensor {doc_id} had unexpected data payload at index [5], ignoring.")
                structure_to_store.pop(5)
                structure_to_store[4]["knowledge_blob_cid"] = None # Processors don't store main data in blob

        # Convert TensorCoordinate to string before saving structure
        if isinstance(structure_to_store[0], TensorCoordinate):
             structure_to_store[0] = structure_to_store[0].to_string()

        doc_entry["structure"] = structure_to_store # Store the potentially modified structure
        self.data[doc_id] = doc_entry
        self._save_db()
        # print(f"Inserted tensor {doc_id} (type: {tensor_type})") # Optional: noisy
        return doc_id

    def get_veector_tensor(self, doc_id: str, load_knowledge_data: bool = False) -> Optional[List]:
        """
        Retrieves a Veector tensor structure by its ID (hash).
        If it's a knowledge tensor and load_knowledge_data is True, attempts to load data from blob storage.
        Returns the full tensor structure list, or None if not found or invalid.
        """
        doc_entry = self._get_doc_metadata(doc_id)
        if not doc_entry or doc_entry.get("type") != "veector_tensor":
            # print(f"Tensor {doc_id} not found or not a veector_tensor.")
            return None

        structure = pickle.loads(pickle.dumps(doc_entry["structure"])) # Deep copy
        metadata = structure[4] # Metadata is at index [4] now
        tensor_type = metadata.get("tensor_type")

        # Restore TensorCoordinate object
        coord_str = structure[0]
        if isinstance(coord_str, str):
             coord_obj = TensorCoordinate.from_string(coord_str)
             if coord_obj:
                 structure[0] = coord_obj
             else:
                 print(f"Warning: Could not restore TensorCoordinate for {doc_id}")

        knowledge_data = None
        if tensor_type == "knowledge" and load_knowledge_data:
            blob_cid = metadata.get("knowledge_blob_cid")
            if blob_cid:
                # print(f"Loading knowledge data blob {blob_cid} for {doc_id}...") # Optional: noisy
                knowledge_data = self._load_blob(blob_cid)
                if knowledge_data is None:
                    print(f"Warning: Failed to load knowledge blob {blob_cid} for {doc_id}")
            else:
                print(f"Warning: Knowledge tensor {doc_id} metadata missing blob CID.")

            # Temporarily add data back for validation (if needed) or return separately?
            # For consistency, let's return the structure WITH the data temporarily at [5]
            # if it was loaded, matching the format before saving.
            # The caller (e.g., compute) should know how to handle this.
            if knowledge_data is not None:
                 structure.append(knowledge_data)

        # Final validation before returning
        # Temporarily remove data payload if added, for validation of the base structure
        temp_data = structure.pop(5) if len(structure) == 6 else None
        is_valid = validate_tensor(structure)
        # Add data back if it was removed
        if temp_data is not None:
             structure.append(temp_data)

        if is_valid:
            return structure
        else:
            print(f"Error: Retrieved structure for {doc_id} is invalid.")
            return None

    def archive_tensor(self, doc_id: str) -> bool:
        """Marks a tensor as 'archived' by updating its status in metadata."""
        doc_entry = self._get_doc_metadata(doc_id)
        if doc_entry and doc_entry.get("type") == "veector_tensor":
            structure = doc_entry.get("structure")
            # Metadata is at index [4]
            if isinstance(structure, list) and len(structure) >= 5 and isinstance(structure[4], dict):
                 structure[4]["status"] = "archived"
                 structure[4]["archived_at"] = datetime.now().isoformat()
                 self._save_db()
                 print(f"Tensor {doc_id} archived.")
                 return True
            else:
                 print(f"Error archiving: Invalid structure for {doc_id}.")
                 return False
        else:
             print(f"Tensor {doc_id} not found for archiving.")
             return False

    def update_tensor_metadata(self, doc_id: str, updates: Dict) -> bool:
         """Updates specific fields in a tensor's metadata."""
         doc_entry = self._get_doc_metadata(doc_id)
         if doc_entry and doc_entry.get("type") == "veector_tensor":
             structure = doc_entry.get("structure")
             if isinstance(structure, list) and len(structure) >= 5 and isinstance(structure[4], dict):
                  structure[4].update(updates)
                  # Add timestamp for the metadata update
                  structure[4]["metadata_updated_at"] = datetime.now().isoformat()
                  self._save_db()
                  print(f"Metadata updated for tensor {doc_id}.")
                  return True
             else:
                  print(f"Error updating metadata: Invalid structure for {doc_id}.")
                  return False
         else:
             print(f"Tensor {doc_id} not found for metadata update.")
             return False

    # --- Querying Methods ---

    def find(self, criteria_func) -> Dict[str, Dict]:
         """
         Finds documents whose metadata entry satisfies the criteria_func.
         criteria_func receives (doc_id, doc_entry).
         Returns a dictionary {doc_id: doc_entry}.
         """
         results = {}
         for doc_id, doc_entry in self.data.items():
             if criteria_func(doc_id, doc_entry):
                 results[doc_id] = doc_entry # Return the raw DB entry
         return results

    def find_tensors(self, criteria_func = None) -> Dict[str, List]:
        """
        Finds Veector tensors, optionally filtering using criteria_func on the tensor structure list.
        Returns a dictionary {doc_id: tensor_structure_list (without knowledge data loaded)}.
        """
        results = {}
        tensor_entries = self.find(lambda id, entry: entry.get("type") == "veector_tensor")
        for doc_id, entry in tensor_entries.items():
            structure = self.get_veector_tensor(doc_id, load_knowledge_data=False) # Get structure only
            if structure:
                 if criteria_func is None or criteria_func(doc_id, structure):
                      results[doc_id] = structure
        return results


    def find_active_tensors(self, tensor_type: Optional[str] = None, tags: Optional[List[str]] = None, coord_filter: Optional[Dict] = None) -> Dict[str, List]:
        """
        Finds active Veector tensors, optionally filtering by type, compatibility/required tags, and coordinate parts.
        Returns {doc_id: tensor_structure_list (metadata only)}.
        """
        def filter_func(doc_id, structure):
            metadata = get_tensor_metadata(structure)
            if metadata.get("status") != "active":
                return False
            if tensor_type and metadata.get("tensor_type") != tensor_type:
                return False

            # Tag filtering (check if tensor has ALL specified tags)
            if tags:
                tensor_tags = set()
                if metadata.get("tensor_type") == "knowledge":
                    tensor_tags.update(metadata.get("compatibility_tags", []))
                elif metadata.get("tensor_type") == "processor":
                     # Processors might also have tags describing their function
                     tensor_tags.update(metadata.get("function_tags", [])) # Example tag key
                if not set(tags).issubset(tensor_tags):
                    return False

            # Coordinate filtering
            if coord_filter:
                coord = get_tensor_coord(structure)
                if not coord: return False
                for key, value in coord_filter.items():
                    if getattr(coord, key, None) != value:
                        return False
            return True

        return self.find_tensors(criteria_func=filter_func)

    def find_knowledge_for_processor(self, processor_structure: List, required_nest: Optional[int] = None) -> Dict[str, List]:
         """Finds active knowledge tensors matching processor requirements and desired nest level."""
         if get_tensor_type(processor_structure) != "processor": return {}

         required_tags = processor_structure[4].get("required_knowledge_tags", [])
         if not required_tags: return {}

         # Filter by type="knowledge", active status, required tags, and optional nest level
         coord_filter = {"nest": required_nest} if required_nest is not None else None
         # We might need knowledge from different groups/layers depending on the processor
         # For now, search globally matching tags and nest level
         matching_knowledge = self.find_active_tensors(
             tensor_type="knowledge",
             tags=required_tags,
             coord_filter=coord_filter
         )
         return matching_knowledge


    def find_children(self, parent_doc_id: str) -> List[str]:
         """Finds IDs of children tensors linking to the given parent ID."""
         children = self.find_tensors(
             lambda id, structure: parent_doc_id in get_tensor_parents(structure)
         )
         return list(children.keys())

    # Basic CRUD for non-tensor documents (if needed)
    def insert_doc(self, doc_type: str, data: Any, doc_id: Optional[str] = None) -> str:
         """Inserts a generic document."""
         if doc_id is None:
              doc_id = hashlib.sha256(pickle.dumps(data)).hexdigest()
         doc_entry = {"type": doc_type, "data": data, "inserted_at": datetime.now().isoformat()}
         self.data[doc_id] = doc_entry
         self._save_db()
         return doc_id

    def get_doc(self, doc_id: str) -> Optional[Dict]:
         """Retrieves a generic document."""
         return self._get_doc_metadata(doc_id)

    def delete_doc(self, doc_id: str) -> bool:
         """Deletes a generic document."""
         if doc_id in self.data:
              del self.data[doc_id]
              self._save_db()
              return True
         return False


# --- Example Usage ---
if __name__ == "__main__":
    print("\n--- VeectorDB Example ---")
    # Use a temporary directory for testing
    script_dir = Path(__file__).parent
    test_db_dir = script_dir / "../data/test_db_v2"

    # Clean up previous test runs
    if test_db_dir.exists():
        import shutil
        shutil.rmtree(test_db_dir)
        print(f"Cleaned up old test DB: {test_db_dir}")

    db = VeectorDB(db_dir=test_db_dir)
    print(f"Initialized DB in: {test_db_dir.resolve()}")

    # 1. Create coordinates
    coord_proc = TensorCoordinate(layer=1, group=1, nest=0, x=1)
    coord_know = TensorCoordinate(layer=0, group=1, nest=0, x=5) # Knowledge at different coord

    # 2. Create & Insert Knowledge Tensor
    weights = np.arange(10, dtype=np.float32).reshape(2, 5)
    knowledge_meta = {"description": "Simple weights"}
    knowledge_tensor_struct = create_tensor(
        coord=coord_know,
        tensor_type="knowledge",
        knowledge_data=weights,
        compatibility_tags=["weight", "float32", "linear"],
        metadata=knowledge_meta
    )
    knowledge_id = db.insert_veector_tensor(knowledge_tensor_struct)
    print(f"\nInserted Knowledge Tensor ID: {knowledge_id}")

    # 3. Create & Insert Processor Tensor referencing the knowledge
    processor_meta = {"description": "Applies weights"}
    processor_tensor_struct = create_tensor(
        coord=coord_proc,
        tensor_type="processor",
        ops_sequence=[OP_MATRIX_MULTIPLY], # Use matrix multiply
        required_knowledge_tags=["weight", "linear"], # Look for knowledge with these tags
        # Mapping is crucial if ops need named args, or by convention
        param_mapping={knowledge_id: "weights"}, # Tell compute how to use the loaded knowledge
        input_channels=["input"],
        output_channels=["output"],
        metadata=processor_meta
    )
    processor_id = db.insert_veector_tensor(processor_tensor_struct)
    print(f"Inserted Processor Tensor ID: {processor_id}")

    # 4. Retrieve Processor (metadata only by default)
    retrieved_proc = db.get_veector_tensor(processor_id)
    if retrieved_proc:
        print(f"\nRetrieved Processor Structure (Metadata Only):")
        print(f"  Type: {get_tensor_type(retrieved_proc)}")
        print(f"  Coord: {get_tensor_coord(retrieved_proc)}")
        print(f"  Meta: {get_tensor_metadata(retrieved_proc)}")
        # print(retrieved_proc) # Print full structure if needed
    else:
        print(f"Failed to retrieve processor {processor_id}")

    # 5. Retrieve Knowledge (with data)
    retrieved_know = db.get_veector_tensor(knowledge_id, load_knowledge_data=True)
    if retrieved_know:
        print(f"\nRetrieved Knowledge Tensor Structure (With Data):")
        print(f"  Type: {get_tensor_type(retrieved_know)}")
        print(f"  Coord: {get_tensor_coord(retrieved_know)}")
        # Data is temporarily at index [5] after loading
        know_data = retrieved_know[5] if len(retrieved_know) == 6 else None
        print(f"  Data Loaded: {know_data is not None}")
        if know_data is not None:
             print(f"  Data Shape: {know_data.shape}")
             print(f"  Data Dtype: {know_data.dtype}")
             # print(know_data)
    else:
        print(f"Failed to retrieve knowledge {knowledge_id}")

    # 6. Find active knowledge tensors with specific tags
    print("\nFinding active 'knowledge' tensors tagged 'weight'...")
    active_weights = db.find_active_tensors(tensor_type="knowledge", tags=["weight"])
    print(f"Found {len(active_weights)} active weight tensors:")
    for found_id, structure in active_weights.items():
        print(f"  - ID: {found_id}, Coord: {get_tensor_coord(structure)}, Meta: {get_tensor_metadata(structure)}")

    # 7. Archive the processor
    print(f"\nArchiving processor {processor_id}...")
    archived = db.archive_tensor(processor_id)
    retrieved_proc_after_archive = db.get_veector_tensor(processor_id)
    if retrieved_proc_after_archive:
        print(f"Processor status after archive: {get_tensor_status(retrieved_proc_after_archive)}")
    else:
        print("Could not retrieve processor after archive attempt.")

    # 8. Try finding active processors now
    print("\nFinding active 'processor' tensors...")
    active_processors = db.find_active_tensors(tensor_type="processor")
    print(f"Found {len(active_processors)} active processors.") # Should be 0 if archiving worked