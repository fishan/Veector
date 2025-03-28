# FILE: tensors.py
# English comments added for the community

import numpy as np
from datetime import datetime
import pickle
import hashlib
from typing import Dict, List, Any, Optional, Tuple, Union # Ensure typing is imported

# --- Tensor Coordinate System ---

class TensorCoordinate:
    """
    Represents the multi-dimensional address of a tensor within the Veector space.
    - layer: Abstraction level (e.g., -1 system, 0 core, 1 optimization, 2 expertise)
    - group: Functional domain (e.g., 0 vision, 1 text, 2 audio, 3 logic)
    - nest: Depth/Expertise level (e.g., 0 base/lite, 1 intermediate/pro, 2 expert/ultimate)
    - x, y, z: Spatial or conceptual position within the layer/group/nest grid.
    """
    def __init__(self, layer: int = 0, group: int = 0, nest: int = 0, x: int = 0, y: int = 0, z: int = 0):
        self.layer = int(layer)
        self.group = int(group)
        self.nest = int(nest)
        self.x = int(x)
        self.y = int(y)
        self.z = int(z)

    def to_tuple(self) -> Tuple[int, int, int, int, int, int]:
        """Return coordinate components as a tuple."""
        return (self.layer, self.group, self.nest, self.x, self.y, self.z)

    def to_string(self) -> str:
        """Return a standardized string representation, suitable for keys or IDs."""
        return f"L{self.layer}_G{self.group}_N{self.nest}_X{self.x}_Y{self.y}_Z{self.z}"

    @classmethod
    def from_string(cls, coord_str: str) -> Optional['TensorCoordinate']:
        """Create a TensorCoordinate object from its string representation."""
        try:
            parts = coord_str.replace('L','').replace('G','').replace('N','').replace('X','').replace('Y','').replace('Z','').split('_')
            coords_int = [int(p) for p in parts]
            if len(coords_int) == 6:
                return cls(*coords_int)
            else:
                # print(f"Warning: Could not parse coordinate string: {coord_str}") # Less verbose
                return None
        except Exception as e:
            # print(f"Error parsing coordinate string {coord_str}: {e}") # Less verbose
            return None

    def __str__(self) -> str:
        return self.to_string()

    def __repr__(self) -> str:
        return f"TensorCoordinate(layer={self.layer}, group={self.group}, nest={self.nest}, x={self.x}, y={self.y}, z={self.z})"

    def __hash__(self) -> int:
        """Allows using as dictionary keys."""
        return hash(self.to_tuple())

    def __eq__(self, other) -> bool:
        """Checks for equality between TensorCoordinate objects."""
        if not isinstance(other, TensorCoordinate):
            return NotImplemented
        return self.to_tuple() == other.to_tuple()

# --- Updated Tensor Structure ---
# [0]: Coordinates (object: TensorCoordinate)
# [1]: Operation & Channels ([default_op_code_list], [input_channels_list], [output_channels_list])
# [2]: Filters (list of filter IDs or filter parameters)
# [3]: Exit Gates (list of gate IDs or predicate functions)
# [4]: Metadata (dict: evolutionary_version, parents, status, tensor_type, created_at, etc...)
# [5]: (Temporary, only for knowledge tensors before DB save) - Actual knowledge data (e.g., np.ndarray)
# ---

def create_tensor(
    coord: TensorCoordinate,
    tensor_type: str, # "processor" or "knowledge"
    metadata: Optional[Dict] = None,
    # --- Fields primarily for "processor" tensors ---
    op: Optional[List[int]] = None, # Default operation
    ops_sequence: Optional[List[List[int]]] = None, # Sequence overrides default op
    filters: Optional[List] = None,
    input_channels: Optional[List[Any]] = None, # Use Any for numeric IDs/tags
    output_channels: Optional[List[Any]] = None, # Use Any for numeric IDs/tags
    exit_gates: Optional[List] = None,
    required_knowledge_tags: Optional[List[Any]] = None, # Use Any for numeric IDs/tags
    param_mapping: Optional[Dict] = None, # { tag_id: param_name_str }
    # --- Fields primarily for "knowledge" tensors ---
    knowledge_data: Any = None,
    compatibility_tags: Optional[List[Any]] = None, # Use Any for numeric IDs/tags
    performance_metrics: Optional[Dict] = None,
    training_context: Optional[Dict] = None,
    # --- Common Metadata ---
    parents: Optional[List[str]] = None, # List of parent tensor IDs (hashes)
    evolutionary_version: int = 1,
    status: str = "active" # "active" or "archived"
    ) -> List:
    """
    Creates the tensor structure based on its type.
    Large data for "knowledge" tensors is passed via 'knowledge_data'.
    Processor tensors store their logic definition directly.
    """
    if not isinstance(coord, TensorCoordinate):
        raise TypeError("Coordinates must be a TensorCoordinate object.")
    if tensor_type not in ["processor", "knowledge"]:
        raise ValueError("tensor_type must be 'processor' or 'knowledge'")

    # Section [1]: Operation & Channels
    default_op_code = op if op is not None else [9, 0, 0] # OP_IDENTITY
    op_channels_section = [
        list(map(int, default_op_code)),
        input_channels or [],
        output_channels or []
    ] if tensor_type == "processor" else [[9,0,0], [], []] # Defaults for knowledge

    # Section [2]: Filters
    filters_section = filters or []

    # Section [3]: Exit Gates
    exit_gates_section = exit_gates or []

    # Section [4]: Metadata
    base_metadata = {
        "evolutionary_version": evolutionary_version,
        "parents": parents or [],
        "status": status,
        "tensor_type": tensor_type,
        "created_at": datetime.now().isoformat(),
        "coordinate_str": coord.to_string(),
        **(metadata or {})
    }

    # Add type-specific metadata
    if tensor_type == "processor":
        base_metadata["required_knowledge_tags"] = required_knowledge_tags or []
        base_metadata["ops_sequence"] = ops_sequence
        base_metadata["param_mapping"] = param_mapping or {}
    elif tensor_type == "knowledge":
        base_metadata["compatibility_tags"] = compatibility_tags or []
        base_metadata["performance_metrics"] = performance_metrics or {}
        base_metadata["training_context"] = training_context or {}
        base_metadata["dtype"] = str(getattr(knowledge_data, 'dtype', type(knowledge_data).__name__))
        base_metadata["shape"] = getattr(knowledge_data, 'shape', None)
        if knowledge_data is not None:
             try: base_metadata["data_hash"] = hashlib.sha256(pickle.dumps(knowledge_data)).hexdigest()
             except Exception as hash_e: print(f"Warning: Could not hash knowledge data: {hash_e}"); base_metadata["data_hash"] = None

    # Section [0]: Coordinates
    coord_section = coord

    # Assemble the base structure (5 elements)
    tensor_structure = [
        coord_section,         # [0]
        op_channels_section,   # [1]
        filters_section,       # [2]
        exit_gates_section,    # [3]
        base_metadata          # [4]
    ]

    # Add temporary data payload for knowledge tensors (index [5])
    if tensor_type == "knowledge":
        tensor_structure.append(knowledge_data)

    return tensor_structure

def validate_tensor(tensor: List) -> bool:
    """Validates the 5 or 6 element tensor structure."""
    base_len = 5
    if not isinstance(tensor, list) or len(tensor) < base_len or len(tensor) > base_len + 1 :
        # print(f"Validation Error: Tensor must be list of {base_len} or {base_len+1} elements (got {len(tensor)})") # Less verbose
        return False
    if not isinstance(tensor[0], TensorCoordinate): return False
    if not isinstance(tensor[1], list) or len(tensor[1]) != 3: return False
    if not isinstance(tensor[1][0], list) or not isinstance(tensor[1][1], list) or not isinstance(tensor[1][2], list): return False
    if not isinstance(tensor[2], list): return False
    if not isinstance(tensor[3], list): return False
    if not isinstance(tensor[4], dict): return False
    required_meta = ["evolutionary_version", "parents", "status", "tensor_type", "created_at", "coordinate_str"]
    if not all(key in tensor[4] for key in required_meta):
         # print(f"Validation Error: Metadata missing required keys: {required_meta}") # Less verbose
         return False
    tensor_type = tensor[4].get("tensor_type")
    if tensor_type not in ["processor", "knowledge"]: return False
    if len(tensor) == base_len + 1 and tensor_type == "processor": print(f"Validation Warning: Processor tensor has unexpected data payload at index [{base_len}].");
    # Removed warning for missing payload on knowledge structure load, as this is expected now
    # if len(tensor) == base_len and tensor_type == "knowledge": print(f"Validation Warning: Knowledge tensor structure missing data payload at index [{base_len}].");
    return True

# --- Getter Functions ---

def get_tensor_coord(tensor: List) -> Optional[TensorCoordinate]:
     """Gets the TensorCoordinate object ([0])."""
     return tensor[0] if validate_tensor(tensor) else None

def get_tensor_op_channels(tensor: List) -> Optional[List]:
     """Gets the Operation & Channels section ([1])."""
     return tensor[1] if validate_tensor(tensor) else None

def get_tensor_default_op(tensor: List) -> Optional[List[int]]:
     """Gets the default operation code list ([1][0])."""
     op_ch = get_tensor_op_channels(tensor)
     return op_ch[0] if op_ch else None

# --- ADDED Missing Getters ---
def get_tensor_input_channels(tensor: List) -> List[Any]:
     """Gets the list of input channels ([1][1])."""
     op_ch = get_tensor_op_channels(tensor)
     return op_ch[1] if op_ch else []

def get_tensor_output_channels(tensor: List) -> List[Any]:
     """Gets the list of output channels ([1][2])."""
     op_ch = get_tensor_op_channels(tensor)
     return op_ch[2] if op_ch else []
# --- End Added Getters ---

def get_tensor_filters(tensor: List) -> List:
     """Gets the list of filters ([2])."""
     return tensor[2] if validate_tensor(tensor) else []

def get_tensor_exit_gates(tensor: List) -> List:
     """Gets the list of exit gates ([3])."""
     return tensor[3] if validate_tensor(tensor) else []

def get_tensor_metadata(tensor: List) -> Dict:
     """Gets a copy of the metadata dictionary ([4])."""
     # Returns a copy to prevent accidental modification
     return tensor[4].copy() if validate_tensor(tensor) else {}

def get_tensor_type(tensor: List) -> Optional[str]:
     """Gets the tensor type from metadata."""
     meta = get_tensor_metadata(tensor)
     return meta.get("tensor_type")

def get_tensor_parents(tensor: List) -> List[str]:
     """Gets the list of parent IDs from metadata."""
     meta = get_tensor_metadata(tensor)
     return meta.get("parents", [])

def get_tensor_status(tensor: List) -> Optional[str]:
     """Gets the status ('active', 'archived') from metadata."""
     meta = get_tensor_metadata(tensor)
     return meta.get("status")

def get_processor_ops_sequence(tensor: List) -> Optional[List[List[int]]]:
    """Gets the specific ops_sequence from processor metadata."""
    if get_tensor_type(tensor) == "processor":
        meta = get_tensor_metadata(tensor)
        return meta.get("ops_sequence") # Returns None if not present
    return None

def get_processor_required_knowledge_tags(tensor: List) -> List[Any]:
    """Gets the required knowledge tags from processor metadata."""
    if get_tensor_type(tensor) == "processor":
        meta = get_tensor_metadata(tensor)
        return meta.get("required_knowledge_tags", [])
    return []

def get_processor_param_mapping(tensor: List) -> Dict:
    """Gets the parameter mapping from processor metadata."""
    if get_tensor_type(tensor) == "processor":
        meta = get_tensor_metadata(tensor)
        return meta.get("param_mapping", {})
    return {}

def get_knowledge_compatibility_tags(tensor: List) -> List[Any]:
    """Gets the compatibility tags from knowledge metadata."""
    if get_tensor_type(tensor) == "knowledge":
        meta = get_tensor_metadata(tensor)
        return meta.get("compatibility_tags", [])
    return []

def get_tensor_hash(tensor_structure: List) -> str:
     """
     Creates a unique hash for the tensor structure based on its coordinates and evolutionary version.
     This identifies the conceptual tensor.
     """
     if not validate_tensor(tensor_structure):
          raise ValueError("Invalid tensor structure for hashing.")
     coord = get_tensor_coord(tensor_structure)
     meta = get_tensor_metadata(tensor_structure)
     # Use coordinate string and evolutionary version for a stable ID
     hash_data = (coord.to_string(), meta.get("evolutionary_version", 1))
     serialized = pickle.dumps(hash_data)
     return hashlib.sha256(serialized).hexdigest()