# FILE: tensors.py
# Version: 0.7.6 (Hybrid Approach: Added metadata_extra to Tuple)

import numpy as np
from datetime import datetime
import pickle
import hashlib
from typing import Dict, List, Any, Optional, Tuple, Union
import traceback # Добавим traceback для отладки

# --- Version ---
TENSORS_VERSION = "0.7.6" # Hybrid list structure, Added metadata_extra to Tuple

# --- Type Hint for Metadata Tuple ---
MetadataTuple = Tuple[
    List[Union[float, int]], # [0] data_description: [version, type_code, dtype_code, name_id, has_blob_flag]
    List[int],              # [1] coord: [layer, group, nest, x, y, z]
    List[int],              # [2] shape: [dim1, dim2, ...]
    List[int],              # [3] tags: [tag1, tag2, ...]
    Optional[Dict],         # [4] ops_sequences: { "default": [...], "prec_key": [...] }
    Optional[Dict],         # [5] interface: { "inputs": [...], "outputs": [...], "knowledge_needed": [...] }
    Optional[List],         # [6] filters: [...]
    Optional[List],         # [7] exit_gates: [...]
    List[int],              # [8] lifecycle: [status_code, evo_version]
    Optional[List[str]],    # [9] parents: [parent_id1, ...]
    Optional[Dict]          # [10] metadata_extra: {"quantization_scale": float, ...} # <<< ИЗМЕНЕНО >>> Добавлен слот
]

# --- Tensor Coordinate System ---
# (Без изменений)
class TensorCoordinate:
    def __init__(self, layer: int = 0, group: int = 0, nest: int = 0, x: int = 0, y: int = 0, z: int = 0):
        self.layer = int(layer)
        self.group = int(group)
        self.nest = int(nest)
        self.x = int(x)
        self.y = int(y)
        self.z = int(z)
    def to_tuple(self) -> Tuple[int, int, int, int, int, int]:
        return (self.layer, self.group, self.nest, self.x, self.y, self.z)
    def to_string(self) -> str:
        return f"L{self.layer}_G{self.group}_N{self.nest}_X{self.x}_Y{self.y}_Z{self.z}"
    @classmethod
    def from_string(cls, coord_str: str) -> Optional['TensorCoordinate']:
        try: parts = coord_str.replace('L','').replace('G','').replace('N','').replace('X','').replace('Y','').replace('Z','').split('_'); coords_int = [int(p) for p in parts]; return cls(*coords_int) if len(coords_int) == 6 else None
        except Exception: return None
    def __str__(self) -> str: return self.to_string()
    def __repr__(self) -> str: return f"TensorCoordinate(layer={self.layer}, group={self.group}, nest={self.nest}, x={self.x}, y={self.y}, z={self.z})"
    def __hash__(self) -> int: return hash(self.to_tuple())
    def __eq__(self, other) -> bool: return self.to_tuple() == other.to_tuple() if isinstance(other, TensorCoordinate) else NotImplemented


# --- Tags, Groups, Mappings ---
# (Без изменений)
TAG_TYPE_PROCESSOR = 1; TAG_TYPE_KNOWLEDGE = 2; TAG_TYPE_CONVERTER = 3; TAG_TYPE_STATE = 4
TAG_MODEL_QWEN2 = 10; TAG_MODEL_LLAMA3 = 11; TAG_MODEL_DEEPSEEK = 12
TAG_PREC_FLOAT32 = 20; TAG_PREC_FLOAT16 = 21; TAG_PREC_BFLOAT16 = 22; TAG_PREC_INT8 = 23; TAG_PREC_INT4 = 24
TAG_COMP_WEIGHTS = 30; TAG_COMP_BIAS = 31; TAG_COMP_EMBEDDING = 32; TAG_COMP_ATTN_Q = 33; TAG_COMP_ATTN_K = 34; TAG_COMP_ATTN_V = 35; TAG_COMP_ATTN_O = 36; TAG_COMP_ATTN_QKV = 37; TAG_COMP_FFN_GATE = 38; TAG_COMP_FFN_UP = 39; TAG_COMP_FFN_DOWN = 40; TAG_COMP_LAYERNORM = 41; TAG_COMP_LM_HEAD = 42
TAG_FUNC_LINEAR = 50; TAG_FUNC_ATTENTION = 51; TAG_FUNC_FFN = 52; TAG_FUNC_EMBED_LOOKUP = 53; TAG_FUNC_CAST_DTYPE = 54; TAG_FUNC_RESHAPE = 55
TAG_SEMANTIC_HIDDEN_STATE = 60; TAG_SEMANTIC_LOGITS = 61; TAG_SEMANTIC_TOKEN_IDS = 62; TAG_SEMANTIC_KV_CACHE = 63
LAYER_IDX_TAG_OFFSET = 100
def tag_layer(idx: int) -> int:
    if not isinstance(idx, int): raise TypeError(f"Layer index must be integer, got {type(idx)}")
    # Разрешаем отрицательные индексы для тегирования (L-1 и т.д.)
    # if idx < 0: raise ValueError(f"Invalid layer index for tagging: {idx}. Must be non-negative.")
    return LAYER_IDX_TAG_OFFSET + idx
USER_TAG_OFFSET = 1000
GROUP_IDX_QWEN_KNOWLEDGE = 100; GROUP_IDX_QWEN_PROCESSOR = 500; GROUP_IDX_LLAMA_KNOWLEDGE = 101; GROUP_IDX_LLAMA_PROCESSOR = 501; GROUP_IDX_DEEPSEEK_KNOWLEDGE = 102; GROUP_IDX_GENERIC_PROCESSOR = 50
DATA_TYPE_MAPPING = {"knowledge": 1, "processor": 2, "converter": 3, "state": 4}
REVERSE_DATA_TYPE_MAPPING = {1: "knowledge", 2: "processor", 3: "converter", 4: "state"}
DTYPE_MAPPING = { 'float32': 1, 'float16': 2, 'bfloat16': 3, 'int8': 4, 'int4': 5, 'int32': 6, 'int64': 7, 'bool': 8, 'complex64': 9, 'complex128': 10, np.float32: 1, np.float16: 2, np.int8: 4, np.int32: 6, np.int64: 7, np.bool_: 8, np.complex64: 9, np.complex128: 10, 'torch.float32': 1, 'torch.float16': 2, 'torch.bfloat16': 3, 'torch.int8': 4, 'torch.int32': 6, 'torch.int64': 7, 'torch.bool': 8, 'torch.complex64': 9, 'torch.complex128': 10, }
REVERSE_DTYPE_MAPPING = { 1: 'float32', 2: 'float16', 3: 'bfloat16', 4: 'int8', 5: 'int4', 6: 'int32', 7: 'int64', 8: 'bool', 9: 'complex64', 10: 'complex128', }
STATUS_MAPPING = {"active": 1, "archived": 0}
REVERSE_STATUS_MAPPING = {1: "active", 0: "archived"}
METADATA_STRUCTURE_VERSION = 1.2 # <<< ИЗМЕНЕНО >>> Увеличиваем версию из-за нового поля

# --- Encoding Helpers ---
# (Функции _encode_coord, _encode_tags, _encode_data_description, _encode_shape, _encode_lifecycle без изменений)
def _encode_coord(coord: TensorCoordinate) -> List[int]:
    if not isinstance(coord, TensorCoordinate): return [-1] * 6
    return [coord.layer, coord.group, coord.nest, coord.x, coord.y, coord.z]
def _encode_tags(tags: Optional[List[int]]) -> List[int]:
    if tags is None: return []
    if not isinstance(tags, list): return []
    processed_tags = []; seen_tags = set()
    for tag in tags:
        if isinstance(tag, int) and tag not in seen_tags: processed_tags.append(tag); seen_tags.add(tag)
    return processed_tags
def _encode_data_description( tensor_type: str, dtype: Union[str, type, None], name_id: int, has_blob_data: bool ) -> List[Union[float, int]]:
    encoded_version=float(METADATA_STRUCTURE_VERSION); encoded_data_type=DATA_TYPE_MAPPING.get(tensor_type,0); encoded_dtype=0
    if dtype is not None:
        dtype_key = dtype;
        if not isinstance(dtype, str):
             try:
                 if hasattr(dtype,'__name__'): dtype_key=dtype.__name__.lower()
                 elif isinstance(dtype, np.dtype): dtype_key=dtype.name
                 else: dtype_key_str=str(dtype).lower();dtype_key=dtype_key_str[6:] if dtype_key_str.startswith('torch.') else dtype_key_str
             except Exception: dtype_key=str(dtype)
        encoded_dtype = DTYPE_MAPPING.get(dtype_key, 0)
        if encoded_dtype == 0 and isinstance(dtype_key, str):
             dtype_str_lower=dtype_key.lower()
             if 'float32' in dtype_str_lower: encoded_dtype = DTYPE_MAPPING.get('float32', 0)
             elif 'float16' in dtype_str_lower: encoded_dtype = DTYPE_MAPPING.get('float16', 0)
             elif 'bfloat16' in dtype_str_lower: encoded_dtype = DTYPE_MAPPING.get('bfloat16', 0)
             elif 'int8' in dtype_str_lower: encoded_dtype = DTYPE_MAPPING.get('int8', 0)
             elif 'int32' in dtype_str_lower: encoded_dtype = DTYPE_MAPPING.get('int32', 0)
             elif 'int64' in dtype_str_lower: encoded_dtype = DTYPE_MAPPING.get('int64', 0)
             elif 'bool' in dtype_str_lower: encoded_dtype = DTYPE_MAPPING.get('bool', 0)
             elif 'complex64' in dtype_str_lower: encoded_dtype = DTYPE_MAPPING.get('complex64', 0)
             elif 'complex128' in dtype_str_lower: encoded_dtype = DTYPE_MAPPING.get('complex128', 0)
             elif 'int4' in dtype_str_lower: encoded_dtype = DTYPE_MAPPING.get('int4', 0)
    encoded_name_id = int(name_id); encoded_has_blob = 1 if has_blob_data else 0
    return [encoded_version, encoded_data_type, encoded_dtype, encoded_name_id, encoded_has_blob]
def _encode_shape(shape: Optional[Union[Tuple[int, ...], List[int], Any]]) -> List[int]:
     if shape is None: return []
     try: return list(int(d) for d in shape)
     except (TypeError, ValueError): return []
def _encode_lifecycle(status: str, evo_version: int) -> List[int]:
    status_code = STATUS_MAPPING.get(status, 0); return [ status_code, int(evo_version) ]


# --- Основная функция кодирования ---
def encode_metadata_tuple(
    coord: TensorCoordinate, tensor_type: str, tags: Optional[List[int]],
    dtype: Union[str, type, None], name_id: int, has_blob_data: bool, shape: Optional[Tuple[int, ...]],
    ops_sequences: Optional[Dict], interface: Optional[Dict], filters: Optional[List],
    exit_gates: Optional[List], status: str, evo_version: int, parents: Optional[List[str]],
    metadata_extra: Optional[Dict] = None # <<< ДОБАВЛЕНО >>> Принимаем metadata_extra
) -> MetadataTuple: # Возвращаем аннотацию типа (уже обновлена)
    """Кодирует все метаданные в 11-элементный кортеж. (v0.7.6)"""
    data_desc_array = _encode_data_description(tensor_type, dtype, name_id, has_blob_data)
    coord_array = _encode_coord(coord); shape_array = _encode_shape(shape); tags_array = _encode_tags(tags)
    lifecycle_array = _encode_lifecycle(status, evo_version); parents_list = parents if parents is not None else None

    # Проверка типов необязательных полей
    ops_sequences_val = ops_sequences
    interface_val = interface
    filters_val = filters
    exit_gates_val = exit_gates
    metadata_extra_val = metadata_extra # <<< ДОБАВЛЕНО >>>
    if ops_sequences_val is not None and not isinstance(ops_sequences_val, dict): raise TypeError("ops_sequences must be dict or None")
    if interface_val is not None and not isinstance(interface_val, dict): raise TypeError("interface must be dict or None")
    if filters_val is not None and not isinstance(filters_val, list): raise TypeError("filters must be list or None")
    if exit_gates_val is not None and not isinstance(exit_gates_val, list): raise TypeError("exit_gates must be list or None")
    if parents_list is not None and not isinstance(parents_list, list): raise TypeError("parents must be list or None")
    if metadata_extra_val is not None and not isinstance(metadata_extra_val, dict): raise TypeError("metadata_extra must be dict or None") # <<< ДОБАВЛЕНО >>>

    # Возвращаем кортеж из 11 элементов
    return (
        data_desc_array,     # [0]
        coord_array,         # [1]
        shape_array,         # [2]
        tags_array,          # [3]
        ops_sequences_val,   # [4]
        interface_val,       # [5]
        filters_val,         # [6]
        exit_gates_val,      # [7]
        lifecycle_array,     # [8]
        parents_list,        # [9]
        metadata_extra_val   # [10] <<< ИЗМЕНЕНО >>> Добавляем metadata_extra в конец
    )

# --- Гибридная Функция Создания Тензора ---
def create_tensor(
    coord: TensorCoordinate, tensor_type: str, knowledge_data: Any = None,
    tags: Optional[List[int]] = None, dtype: Union[str, type, None] = None,
    shape: Optional[Tuple[int, ...]] = None, name_id: int = -1,
    interface: Optional[Dict] = None, ops_sequences: Optional[Dict] = None,
    filters: Optional[List] = None, exit_gates: Optional[List] = None,
    op: Optional[List[int]] = None, input_channels: Optional[List[Any]] = None,
    output_channels: Optional[List[Any]] = None, metadata_extra: Optional[Dict] = None, # <<< ДОБАВЛЕНО >>> Принимаем metadata_extra
    parents: Optional[List[str]] = None, evolutionary_version: int = 1,
    status: str = "active", creator_id: Optional[Any] = None
) -> List: # Сигнатура без изменений (возвращает список)
    """v0.7.6: Создает СТАРУЮ структуру списка, встраивая кортеж (включая metadata_extra)."""
    if not isinstance(coord, TensorCoordinate): raise TypeError("Coordinates must be TensorCoordinate.")
    allowed_types = ["processor", "knowledge", "converter", "state"];
    if tensor_type not in allowed_types: raise ValueError(f"tensor_type must be one of {allowed_types}")
    has_blob = (knowledge_data is not None); final_dtype = dtype; final_shape = shape
    if has_blob:
        if final_dtype is None: final_dtype = getattr(knowledge_data, 'dtype', type(knowledge_data))
        if final_shape is None: final_shape = tuple(getattr(knowledge_data, 'shape', ()))

    # <<< ИЗМЕНЕНО >>> Передаем metadata_extra в encode_metadata_tuple
    encoded_meta_tuple = encode_metadata_tuple(
        coord=coord, tensor_type=tensor_type, tags=tags, dtype=final_dtype,
        name_id=name_id, has_blob_data=has_blob, shape=final_shape,
        ops_sequences=ops_sequences, interface=interface, filters=filters,
        exit_gates=exit_gates, status=status, evo_version=evolutionary_version,
        parents=parents, metadata_extra=metadata_extra # <<< ДОБАВЛЕНО >>>
    )

    default_op_code = op if op is not None else [9, 0, 0]
    op_channels_section = [ list(map(int, default_op_code)), input_channels or [], output_channels or [] ] if tensor_type in ["processor", "converter"] else [[9,0,0], [], []]
    filters_section = filters or []; exit_gates_section = exit_gates or []
    processed_tags = _encode_tags(tags)

    dtype_code_from_tuple = 0; dtype_str_repr = str(final_dtype)
    if isinstance(encoded_meta_tuple, tuple) and len(encoded_meta_tuple)>0 and isinstance(encoded_meta_tuple[0], list) and len(encoded_meta_tuple[0])>2: dtype_code_from_tuple = encoded_meta_tuple[0][2]; dtype_str_repr = REVERSE_DTYPE_MAPPING.get(dtype_code_from_tuple, dtype_str_repr)

    # Собираем базовый MetaDict
    base_metadata = {
        "evolutionary_version": evolutionary_version,
        "parents": parents or [],
        "status": status,
        "tensor_type": tensor_type,
        "created_at": datetime.now().isoformat(),
        "coordinate_str": coord.to_string(),
        "tags": processed_tags,
        "interface": interface or {},
        "ops_sequences": ops_sequences or {},
        "creator_id": creator_id,
        "has_blob_data": has_blob,
        "dtype": dtype_str_repr,
        "shape": final_shape,
        "data_hash": None,
        "_encoded_metadata_v1_": encoded_meta_tuple # Встраиваем кортеж
    }

    # Добавляем metadata_extra в base_metadata (для совместимости со старыми геттерами)
    if metadata_extra:
        base_metadata.update(metadata_extra)

    if has_blob and knowledge_data is not None:
        try: base_metadata["data_hash"] = hashlib.sha256(pickle.dumps(knowledge_data)).hexdigest()
        except Exception: pass

    tensor_structure = [ coord, op_channels_section, filters_section, exit_gates_section, base_metadata ]
    if has_blob: tensor_structure.append(knowledge_data)
    return tensor_structure

# --- Валидация СТАРОЙ структуры ---
# (Без изменений)
def validate_tensor(tensor: List) -> bool: # Версия 0.7.3
    base_len=5;
    if not isinstance(tensor, list): return False;
    if len(tensor) < base_len or len(tensor) > base_len + 1: return False;
    if not isinstance(tensor[0], TensorCoordinate): return False;
    if not isinstance(tensor[1], list) or len(tensor[1]) != 3: return False;
    if not isinstance(tensor[1][0], list): return False;
    if not isinstance(tensor[1][1], list): return False;
    if not isinstance(tensor[1][2], list): return False;
    if not isinstance(tensor[2], list): return False;
    if not isinstance(tensor[3], list): return False;
    if not isinstance(tensor[4], dict): return False;
    return True

# --- Валидация НОВОГО кортежа ---
def validate_tensor_tuple(meta_tuple: Tuple) -> bool:
    if not isinstance(meta_tuple, tuple): return False;
    # <<< ИЗМЕНЕНО >>> Проверяем новую длину кортежа
    if len(meta_tuple) != 11: return False;
    # Можно добавить более строгие проверки типов элементов, если нужно
    if not isinstance(meta_tuple[0], list): return False # data_description
    if not isinstance(meta_tuple[1], list): return False # coord
    # ... и т.д.
    if meta_tuple[10] is not None and not isinstance(meta_tuple[10], dict): return False # metadata_extra
    return True

# --- НОВЫЕ Геттер-Функции для Кортежа (Добавлен геттер для metadata_extra) ---
def get_data_description_from_meta(mt: Tuple) -> List: return mt[0] if isinstance(mt, tuple) and len(mt)>0 else []
def get_version_from_meta(mt: Tuple) -> float: desc=get_data_description_from_meta(mt); return float(desc[0]) if desc else 0.0
def get_type_code_from_meta(mt: Tuple) -> int: desc=get_data_description_from_meta(mt); return int(desc[1]) if len(desc)>1 else 0
def get_dtype_code_from_meta(mt: Tuple) -> int: desc=get_data_description_from_meta(mt); return int(desc[2]) if len(desc)>2 else 0
def get_name_id_from_meta(mt: Tuple) -> int: desc=get_data_description_from_meta(mt); return int(desc[3]) if len(desc)>3 else -1
def get_has_blob_flag_from_meta(mt: Tuple) -> int: desc=get_data_description_from_meta(mt); return int(desc[4]) if len(desc)>4 else 0
def get_coord_list_from_meta(mt: Tuple) -> List[int]: return mt[1] if isinstance(mt, tuple) and len(mt)>1 else []
def get_coord_obj_from_meta(mt: Tuple) -> Optional[TensorCoordinate]: cl = get_coord_list_from_meta(mt); return TensorCoordinate(*cl) if len(cl)==6 else None
def get_shape_list_from_meta(mt: Tuple) -> List[int]: return mt[2] if isinstance(mt, tuple) and len(mt)>2 else []
def get_tags_list_from_meta(mt: Tuple) -> List[int]: return mt[3] if isinstance(mt, tuple) and len(mt)>3 else []
def get_ops_sequences_from_meta(mt: Tuple) -> Optional[Dict]: return mt[4] if isinstance(mt, tuple) and len(mt)>4 else None
def get_interface_from_meta(mt: Tuple) -> Optional[Dict]: return mt[5] if isinstance(mt, tuple) and len(mt)>5 else None
def get_filters_from_meta(mt: Tuple) -> Optional[List]: return mt[6] if isinstance(mt, tuple) and len(mt)>6 else None
def get_exit_gates_from_meta(mt: Tuple) -> Optional[List]: return mt[7] if isinstance(mt, tuple) and len(mt)>7 else None
def get_lifecycle_list_from_meta(mt: Tuple) -> List[int]: return mt[8] if isinstance(mt, tuple) and len(mt)>8 else []
def get_status_code_from_meta(mt: Tuple) -> int: lc=get_lifecycle_list_from_meta(mt); return int(lc[0]) if lc else 0
def get_evo_version_from_meta(mt: Tuple) -> int: lc=get_lifecycle_list_from_meta(mt); return int(lc[1]) if len(lc)>1 else 1
def get_parents_list_from_meta(mt: Tuple) -> Optional[List[str]]: return mt[9] if isinstance(mt, tuple) and len(mt)>9 else None
# <<< ДОБАВЛЕНО >>> Новый геттер для metadata_extra
def get_metadata_extra_from_meta(mt: Tuple) -> Optional[Dict]: return mt[10] if isinstance(mt, tuple) and len(mt)>10 else None

# --- СТАРЫЕ Геттер-Функции (для совместимости) ---
# (Без изменений)
def get_tensor_metadata(tensor: List) -> Dict: return tensor[4].copy() if validate_tensor(tensor) else {}
def get_tensor_coord(tensor: List) -> Optional[TensorCoordinate]: return tensor[0] if validate_tensor(tensor) else None
def get_tensor_op_channels(tensor: List) -> Optional[List]: return tensor[1] if validate_tensor(tensor) else None
def get_tensor_filters(tensor: List) -> List: return tensor[2] if validate_tensor(tensor) else []
def get_tensor_exit_gates(tensor: List) -> List: return tensor[3] if validate_tensor(tensor) else []
def get_tensor_type(tensor: List) -> Optional[str]: return get_tensor_metadata(tensor).get("tensor_type")
def get_tensor_status(tensor: List) -> Optional[str]: return get_tensor_metadata(tensor).get("status")
def get_tensor_tags(tensor: List) -> List[int]: return get_tensor_metadata(tensor).get("tags", [])
def get_tensor_interface(tensor: List) -> Dict: return get_tensor_metadata(tensor).get("interface", {})
def get_processor_ops_sequences(tensor: List) -> Optional[Dict]: return get_tensor_metadata(tensor).get("ops_sequences") if get_tensor_type(tensor) in ["processor", "converter"] else None
def get_tensor_parents(tensor: List) -> List[str]: return get_tensor_metadata(tensor).get("parents", [])
def has_blob_data(tensor: List) -> bool: return get_tensor_metadata(tensor).get("has_blob_data", False)

# --- *** Функция хеширования (ПРИНИМАЕТ КОРТЕЖ) *** ---
# (Без изменений, т.к. хеш зависит только от coord и evo_version, которые не изменили позицию)
def get_tensor_hash(meta_tuple: Tuple) -> str: # Ожидает КОРТЕЖ
     """v0.7.6: Создает хеш на основе координат и версии эволюции из КОРТЕЖА метаданных."""
     coord_list = None
     evo_version = None
     try:
         if not validate_tensor_tuple(meta_tuple): # Валидатор теперь проверяет длину 11
              raise ValueError("Provided invalid metadata tuple (failed validate_tensor_tuple)")

         # Геттеры для coord и evo_version работают как раньше
         coord_list = get_coord_list_from_meta(meta_tuple) # tuple[1]
         evo_version = get_evo_version_from_meta(meta_tuple) # tuple[8][1]

         if not isinstance(coord_list, list) or len(coord_list) != 6: raise ValueError(f"Invalid coordinate list extracted from tuple: {coord_list}")
         if not isinstance(evo_version, int): raise ValueError(f"Invalid evolutionary version extracted from tuple: {evo_version}")

         coord_string = f"L{coord_list[0]}_G{coord_list[1]}_N{coord_list[2]}_X{coord_list[3]}_Y{coord_list[4]}_Z{coord_list[5]}"
         hash_data = (coord_string, evo_version)
         serialized = pickle.dumps(hash_data)
         return hashlib.sha256(serialized).hexdigest()
     except ValueError as ve: raise ValueError(f"Cannot generate hash from metadata tuple: {ve}")
     except IndexError as ie: raise ValueError(f"Cannot generate hash: Index error accessing metadata tuple ({ie}). Tuple: {meta_tuple}")
     except TypeError as te: raise ValueError(f"Cannot generate hash: Type error accessing metadata tuple ({te}). Tuple: {meta_tuple}")
     except Exception as e: raise ValueError(f"Could not serialize/hash data for tensor ID: {e}")