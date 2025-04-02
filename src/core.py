# FILE: core.py
# Version: 0.6.13 (Rewritten parser logic in _execute_op_sequence)


# --- Standard Imports ---
import numpy as np
import queue
import threading
import time
import random
import psutil
import os
import pickle
import hashlib
import traceback # Import traceback for better error logging in specific places
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union

# --- Version ---
CORE_VERSION = "0.6.13"

# --- Optional Imports ---
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import ipfshttpclient
    IPFS_AVAILABLE = True
except ImportError:
    IPFS_AVAILABLE = False

try:
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator
    from qiskit import transpile, assemble
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False

TENSORS_VERSION_REQ = "0.7.6"
VEECTORDB_VERSION_REQ = "0.9.7"
OPERATIONS_VERSION_REQ = "0.7.3" 

VeectorDB_defined = False
# --- Veector Project Imports ---
VeectorDB_defined = False
try:
    # Import VeectorDB and check version
    from veectordb import VeectorDB, VEECTORDB_VERSION
    print(f"  Imported VeectorDB (v{VEECTORDB_VERSION})")
    # Ожидаем версию, которая работает со структурой списка
    if VEECTORDB_VERSION < VEECTORDB_VERSION_REQ: # <<< Обновлено требование версии
         raise ImportError(f"Core v{CORE_VERSION} requires VeectorDB v0.9.1+, found v{VEECTORDB_VERSION}")

    # Import Tensors components and check version
    from tensors import (
        TENSORS_VERSION,
        TensorCoordinate,
        # Импортируем ГИБРИДНЫЙ create_tensor (возвращает список)
        create_tensor,
        # Импортируем ВАЛИДАТОР СТРУКТУРЫ СПИСКА
        validate_tensor,
        # Импортируем ХЕШИРОВАНИЕ для СТРУКТУРЫ СПИСКА
        get_tensor_hash,
        # --- ИМПОРТИРУЕМ СТАРЫЕ ГЕТТЕРЫ (для структуры списка) ---
        get_tensor_metadata, get_tensor_coord, get_tensor_type, get_tensor_status,
        get_tensor_tags, get_tensor_interface, get_processor_ops_sequences,
        get_tensor_filters, get_tensor_exit_gates, has_blob_data,
        get_tensor_parents, get_tensor_op_channels,
        # --- КОНЕЦ СТАРЫХ ГЕТТЕРОВ ---
        # Константы тегов и групп, необходимые здесь
        TAG_PREC_INT8, TAG_PREC_FLOAT16, TAG_PREC_FLOAT32, # Нужны для выбора последовательности
        GROUP_IDX_QWEN_KNOWLEDGE, GROUP_IDX_DEEPSEEK_KNOWLEDGE, # Группы знаний
        TAG_MODEL_QWEN2, TAG_MODEL_DEEPSEEK, # Теги моделей
        TAG_TYPE_PROCESSOR, TAG_TYPE_KNOWLEDGE, TAG_TYPE_CONVERTER, TAG_TYPE_STATE,
        TAG_COMP_EMBEDDING, TAG_COMP_LM_HEAD, # Для определения формата блоба
        # Маппинги, необходимые здесь (DTYPE нужен для проверки int8)
        DTYPE_MAPPING
        # Обратные маппинги не используются напрямую в core v0.6.2
        # REVERSE_DATA_TYPE_MAPPING, REVERSE_STATUS_MAPPING
    )
    print(f"  Imported tensors (v{TENSORS_VERSION})")
    # Ожидаем версию с гибридным create_tensor и исправленным validate/hash
    if TENSORS_VERSION < TENSORS_VERSION_REQ: # <<< Обновлено требование версии
         raise ImportError(f"Core v{CORE_VERSION} requires tensors v0.7.3+, found v{TENSORS_VERSION}")

    # Import operations (Явные импорты)
    from operations import *
    print(f"  Imported operations (v{OPERATIONS_VERSION})")

    if OPERATIONS_VERSION < OPERATIONS_VERSION_REQ:
        raise ImportError(f"Core v{CORE_VERSION} requires operations v{OPERATIONS_VERSION_REQ}+, found v{OPERATIONS_VERSION}")


    # Import Memory module
    from memory import Memory, MEMORY_VERSION
    print(f"  Imported Memory (v{MEMORY_VERSION})")

    print("Core components imported successfully.")
    VeectorDB_defined = True

# --- Handle Import Errors Gracefully ---
except ImportError as e:
    print(f"---!!! FATAL ERROR (ImportError) !!! ---")
    print(f"Specific error: {e}")    
    print(f"Ensure files (tensors v{TENSORS_VERSION_REQ}+, veectordb v{VEECTORDB_VERSION_REQ}+, operations v{OPERATIONS_VERSION_REQ}+, memory) are OK.")
    print(f"-----------------------------------------")

    # Определяем только необходимые заглушки
    class VeectorDB: pass
    VEECTORDB_VERSION = "dummy"
    class TensorCoordinate: pass
    TENSORS_VERSION = "dummy"
    def create_tensor(*a,**kw): return []
    def validate_tensor(t): return False
    def get_tensor_hash(t): return "dummy_hash"
    VeectorDB_defined = False
except Exception as other_e:
    print(f"---!!! FATAL ERROR (Other Exception during Import) !!! ---")
    print(f"Specific error: {other_e}")
    traceback.print_exc()
    print(f"Check imported files for syntax errors.")
    print(f"----------------------------------------------------------")
    VeectorDB_defined = False

# --- Constants ---
DEFAULT_CACHE_SIZE = 1000
DEFAULT_EVICTION_STRATEGY = "LRU"
DEFAULT_IPFS_ADDRESS = '/ip4/127.0.0.1/tcp/5001'
STATE_TENSOR_LAYER = -2

# --- Operation Code Constants ---
OP_SUM = [0, 0, 0]
OP_SUBTRACT = [0, 0, 1]
OP_ADD = [0, 0, 2]
OP_MULTIPLY = [0, 1, 0]
OP_DIVIDE = [0, 1, 1]
OP_SQRT = [0, 2, 0]
OP_POWER = [0, 2, 1]
OP_ABS = [0, 3, 0]
OP_MOD = [0, 5, 0]
OP_FLOOR = [0, 6, 0]
OP_CEIL = [0, 6, 1]
OP_SIN = [1, 0, 0]
OP_COS = [1, 0, 1]
OP_TAN = [1, 1, 0]
OP_COT = [1, 1, 1]
OP_ASIN = [1, 2, 0]
OP_ACOS = [1, 2, 1]
OP_ATAN = [1, 3, 0]
OP_GREATER = [2, 0, 0]
OP_EQUAL = [2, 0, 1]
OP_AND = [2, 1, 0]
OP_OR = [2, 1, 1]
OP_NOT = [2, 2, 0]
OP_XOR = [2, 3, 0]
OP_NAND = [2, 4, 0]
OP_NOR = [2, 4, 1]
OP_IF = [3, 0, 0]
OP_LOOP_MULT = [4, 0, 0]
OP_CHOICE = [7, 0, 0]
OP_RAND_UNIFORM = [5, 1, 0]
OP_RAND_NORMAL = [5, 1, 1]
OP_MEDIAN = [5, 2, 0]
OP_PRINT = [8, 0, 0]
OP_IDENTITY = [9, 0, 0]
OP_TRIGGER_REASON = [10, 0, 0]
OP_DFS = [15, 0, 0]
OP_MEAN = [16, 0, 0]
OP_STDDEV = [16, 1, 0]
OP_RELU = [18, 0, 0]
OP_SIGMOID = [18, 1, 0]
OP_SOFTMAX = [18, 2, 0]
OP_LEAKY_RELU = [18, 3, 0]
OP_SILU = [18, 4, 0]
OP_GELU = [40, 5, 0]
OP_EXP_SMOOTHING = [19, 0, 0]
OP_NORMALIZE_01 = [20, 0, 0]
OP_INTERPOLATE = [20, 1, 0]
OP_LAYER_NORM = [40, 1, 0]
OP_BATCH_NORM = [40, 4, 0]
OP_DROPOUT = [40, 3, 0]
OP_GET_Q_ROT = [40, 7, 1]
OP_GET_K_ROT = [40, 7, 2]
OP_MATRIX_MULTIPLY = [30, 0, 0]
OP_DETERMINANT = [30, 1, 0]
OP_EIGENVALUES = [30, 2, 0]
OP_CONVOLUTION = [30, 3, 0]
OP_TRANSPOSE = [30, 4, 0]
OP_INVERSE = [30, 5, 0]
OP_TRACE = [30, 6, 0]
OP_ATTENTION_MULTIHEAD = [40, 2, 0]
OP_EMBEDDING_LOOKUP = [40, 6, 0]
OP_APPLY_ROPE = [40, 7, 0]
OP_SCALED_DOT_PROD_ATTN = [40, 9, 2]
OP_RESHAPE_HEADS = [40, 9, 0]
OP_REPEAT_KV_HEADS = [40, 9, 1]
OP_MERGE_HEADS = [40, 9, 3]
OP_ADD_BIAS = [0, 0, 3]
OP_RESIDUAL_ADD = OP_ADD
OP_LINEAR = OP_MATRIX_MULTIPLY
OP_FINAL_NORM = OP_LAYER_NORM
OP_LINEAR_HEAD = OP_LINEAR
OP_QUANTUM_HADAMARD = [50, 0, 0]
OP_QUANTUM_PAULI_X = [50, 0, 1]
OP_QUANTUM_CNOT = [50, 1, 0]
OP_QUANTUM_MEASURE = [50, 2, 0]
OP_QUANTUM_SUPERPOS = [50, 3, 0]
OP_QUANTUM_ENTANGLE = [50, 4, 0]
META_OP_CATEGORY = 99
OP_STORE = [99,0,0]
OP_LOAD = [99,0,1]
OP_LOAD_INITIAL_INPUT = [99,0,3]
OP_DEBUG_CONTEXT = [99,1,0]

class Veector:
    """
    Core execution engine v0.6.2: Uses list structure, old getters, dequantization.
    """
    def __init__(self,
                 db_dir: Union[str, Path] = "data/db",
                 cache_size: int = DEFAULT_CACHE_SIZE,
                 eviction_strategy: str = DEFAULT_EVICTION_STRATEGY,
                 use_memory_module: bool = False,
                 p2p_node: Optional[Any] = None,
                 ipfs_enabled: bool = False,
                 ipfs_address: str = DEFAULT_IPFS_ADDRESS):

        print(f"--- Initializing Veector Core v{CORE_VERSION} ---")
        print(f"    Requires: tensors v{TENSORS_VERSION_REQ}+, veectordb v{VEECTORDB_VERSION_REQ}+, operations v{OPERATIONS_VERSION_REQ}+")
        print(f"    IPFS: {ipfs_enabled}, Address: {ipfs_address}")
        self.db_dir = Path(db_dir).resolve()

        if not VeectorDB_defined:
             raise RuntimeError("Cannot initialize Veector: VeectorDB or Tensors failed to import.")

        try:
            self.db = VeectorDB(db_dir=self.db_dir)
            print("VeectorDB initialized.")
        except Exception as e:
             print(f"FATAL ERROR: Failed to initialize VeectorDB: {e}")
             raise

        self.p2p_node = p2p_node
        self.ipfs_client = None
        if ipfs_enabled and IPFS_AVAILABLE and ipfs_address:
             try:
                 self.ipfs_client = ipfshttpclient.connect(addr=ipfs_address, timeout=10)
                 print(f"IPFS client connected to {ipfs_address}.")
             except Exception as e:
                 print(f"Warn: Failed IPFS connect: {e}. IPFS disabled.")
                 self.ipfs_enabled = False
             else:
                 self.ipfs_enabled = True
        else:
             self.ipfs_enabled = False

        self.compute_cache: Dict[Tuple, Any] = {}
        self.knowledge_cache: Dict[str, Any] = {}
        self.cache_size = max(10, cache_size)
        self.eviction_strategy = eviction_strategy.upper() if eviction_strategy.upper() in ["LRU", "LFU"] else "LRU"
        self.cache_access_count: Dict[Union[Tuple, str], int] = {}
        self.cache_timestamps: Dict[Union[Tuple, str], float] = {}
        print(f"Cache initialized: Size={self.cache_size}, Strategy={self.eviction_strategy}")

        self.memory_module = Memory() if use_memory_module and 'Memory' in globals() else None

        self.core_ops: Dict[Tuple[int, ...], callable] = {
            tuple(OP_SUM): lambda d,**kw: np.sum(d), tuple(OP_SQRT): np.sqrt, tuple(OP_ABS): np.abs,
            tuple(OP_FLOOR): floor, tuple(OP_CEIL): ceil, tuple(OP_SIN): np.sin, tuple(OP_COS): np.cos,
            tuple(OP_TAN): np.tan, tuple(OP_COT): lambda d,**kw: 1/np.tan(d) if np.all(np.tan(d)!=0) else np.nan,
            tuple(OP_ASIN): arcsin, tuple(OP_ACOS): arccos, tuple(OP_ATAN): arctan, tuple(OP_NOT): np.logical_not,
            tuple(OP_IDENTITY): lambda d,**kw: d, tuple(OP_MEAN): mean, tuple(OP_STDDEV): std_dev,
            tuple(OP_RELU): relu, tuple(OP_SIGMOID): sigmoid, tuple(OP_SOFTMAX): softmax, tuple(OP_LEAKY_RELU): leaky_relu,
            tuple(OP_GELU): gelu, tuple(OP_SILU): silu, tuple(OP_EXP_SMOOTHING): exponential_smoothing,
            tuple(OP_NORMALIZE_01): normalize, tuple(OP_TRANSPOSE): transpose, tuple(OP_INVERSE): inverse,
            tuple(OP_TRACE): trace, tuple(OP_DETERMINANT): matrix_determinant, tuple(OP_EIGENVALUES): matrix_eigenvalues,
            tuple(OP_MEDIAN): median, tuple(OP_INTERPOLATE): interpolate,
            tuple(OP_SUBTRACT): lambda d,**kw: np.subtract(kw.get('minuend'), kw.get('subtrahend')),
            tuple(OP_MULTIPLY): lambda d,**kw: np.multiply(kw.get('factor1'), kw.get('factor2')),
            tuple(OP_DIVIDE): lambda d,**kw: np.divide(kw.get('dividend'), kw.get('divisor')),
            tuple(OP_ADD): lambda d,**kw: add(kw.get('input_a'), kw.get('input_b')),
            tuple(OP_POWER): lambda d,**kw: np.power(kw.get('base'), kw.get('exponent')),
            tuple(OP_MOD): lambda d,**kw: mod(kw.get('x'), kw.get('y')),
            tuple(OP_GREATER): lambda d,**kw: np.greater(kw.get('a'), kw.get('b')),
            tuple(OP_EQUAL): lambda d,**kw: np.equal(kw.get('a'), kw.get('b')),
            tuple(OP_AND): lambda d,**kw: np.logical_and(kw.get('a'), kw.get('b')),
            tuple(OP_OR): lambda d,**kw: np.logical_or(kw.get('a'), kw.get('b')),
            tuple(OP_XOR): lambda d,**kw: xor(kw.get('a'), kw.get('b')),
            tuple(OP_NAND): lambda d,**kw: nand(kw.get('a'), kw.get('b')),
            tuple(OP_NOR): lambda d,**kw: nor(kw.get('a'), kw.get('b')),
            tuple(OP_MATRIX_MULTIPLY): lambda d,**kw: matrix_multiply(d, weights=kw.get('weights'), bias=kw.get('bias')),
            tuple(OP_CONVOLUTION): lambda d,**kw: convolution(d, kernel=kw.get('kernel'), bias=kw.get('bias')),
            tuple(OP_LAYER_NORM): lambda d,**kw: layer_normalization(d, norm_weight=kw.get('norm_weight'), norm_bias=kw.get('norm_bias'), eps=kw.get('eps', 1e-5)),
            tuple(OP_BATCH_NORM): lambda d,**kw: batch_norm(d, **kw),
            tuple(OP_DROPOUT): lambda d,**kw: dropout(d, rate=kw.get('rate', 0.1), is_training=kw.get('is_training', False)),
            tuple(OP_EMBEDDING_LOOKUP): lambda d,**kw: embedding_lookup(d, embedding_matrix=kw.get('embedding_matrix')),
            tuple(OP_APPLY_ROPE): lambda d,**kw: apply_rope(q=kw.get('q_proj'), k=kw.get('k_proj'), position_ids=kw.get('position_ids')),
            tuple(OP_SCALED_DOT_PROD_ATTN): lambda d,**kw: scaled_dot_product_attention(query=kw.get('q_rot'), key=kw.get('k_rot'), value=kw.get('v_proj'), mask=kw.get('attention_mask')),
            tuple(OP_ATTENTION_MULTIHEAD): lambda d,**kw: multi_head_attention(d, **kw),
            tuple(OP_IF): self._op_conditional_if, tuple(OP_LOOP_MULT): self._op_loop_multiply, tuple(OP_CHOICE): self._op_choice_select,
            tuple(OP_PRINT): self._op_output_print, tuple(OP_TRIGGER_REASON): self._op_trigger_reason, tuple(OP_DFS): self._op_graph_dfs,
            tuple(OP_RAND_UNIFORM): lambda d,**kw: random_uniform(min_val=kw.get('min_val', 0.0), max_val=kw.get('max_val', 1.0)),
            tuple(OP_RAND_NORMAL): lambda d,**kw: random_normal(mu=kw.get('mu', 0.0), scale=kw.get('sigma', 1.0)),
            tuple(OP_QUANTUM_HADAMARD): lambda d,**kw: self._quantum_op_placeholder(d,"hadamard",**kw),
            tuple(OP_QUANTUM_PAULI_X): lambda d,**kw: self._quantum_op_placeholder(d,"pauli_x",**kw),
            tuple(OP_QUANTUM_CNOT): lambda d,**kw: self._quantum_op_placeholder(d,"cnot",**kw),
            tuple(OP_QUANTUM_MEASURE): lambda d,**kw: self._quantum_op_placeholder(d,"measure",**kw),
            tuple(OP_QUANTUM_SUPERPOS): lambda d,**kw: self._quantum_op_placeholder(d,"superposition",**kw),
            tuple(OP_QUANTUM_ENTANGLE): lambda d,**kw: self._quantum_op_placeholder(d,"entanglement",**kw),
            # tuple(OP_GET_Q_ROT): lambda d,**kw: get_q_rot(d, **kw),
            # tuple(OP_GET_K_ROT): lambda d,**kw: get_k_rot(d, **kw),
            tuple(OP_GET_Q_ROT): get_q_rot,
            tuple(OP_GET_K_ROT): get_k_rot, 
        }
        print(f"Initialized {len(self.core_ops)} core operations.")
        self._log_memory("Veector Initialized")

    # --- Logging & Monitoring Methods ---
    def _log_memory(self, stage: str):
        """Logs current RAM and GPU memory usage."""
        try:
            process = psutil.Process(os.getpid())
            ram_usage = process.memory_info().rss / 1024**2
            print(f"Mem({stage}): RAM {ram_usage:.1f}MB", end='')
            if TORCH_AVAILABLE and torch.cuda.is_available():
                 try:
                     allocated = torch.cuda.memory_allocated()/1024**2
                     reserved = torch.cuda.memory_reserved()/1024**2
                     print(f" | GPU Alloc {allocated:.1f}MB, Reserv {reserved:.1f}MB")
                 except Exception:
                      # Ignore GPU errors silently in logging
                      print() # Ensure newline
            else:
                print()
        except Exception as e:
            print(f"Mem log warning: {e}")

    def _get_resource_status(self) -> Dict:
        """Returns current resource status."""
        mem_percent=0
        cpu_percent=0
        gpu_mem_percent=0
        try:
            mem=psutil.virtual_memory()
            mem_percent=mem.percent
            cpu_percent=psutil.cpu_percent()
        except Exception:
            pass # Ignore psutil errors
        if TORCH_AVAILABLE and torch.cuda.is_available():
             try:
                 props=torch.cuda.get_device_properties(0)
                 allocated=torch.cuda.memory_allocated()
                 gpu_mem_percent = (allocated / props.total_memory) * 100 if props.total_memory > 0 else 0
             except Exception:
                 pass # Ignore GPU errors
        return {
            "memory_percent": mem_percent,
            "cpu_percent": cpu_percent,
            "gpu_memory_percent": gpu_mem_percent,
            "battery_percent": 100 # Placeholder
        }

    # --- Tensor Creation and Validation Wrappers ---
    # Uses hybrid create_tensor from tensors v0.7.3+ which returns list structure
    def create_tensor(self, *args, **kwargs) -> List:
        """Wrapper for tensor creation function from tensors.py (returns list)."""
        if 'create_tensor' in globals():
             result = create_tensor(*args, **kwargs)
             return result
        else:
             print("Error: create_tensor function not available in tensors module.")
             return []

    # Uses list validator from tensors v0.7.3+
    def validate_tensor(self, tensor_structure: List) -> bool:
        """Wrapper for tensor list validation function from tensors.py."""
        if 'validate_tensor' in globals():
             result = validate_tensor(tensor_structure)
             return result
        else:
             print("Error: validate_tensor function not available.")
             return False

    # Uses list hash function from tensors v0.7.3+
    def get_tensor_hash(self, tensor_structure: List) -> str:
        """Wrapper for tensor list hashing function."""
        if 'get_tensor_hash' in globals():
             try:
                 result = get_tensor_hash(tensor_structure)
                 return result
             except ValueError as e:
                  print(f"Error hashing tensor structure: {e}")
                  return f"error_hash_{random.random()}"
        else:
             print("Error: get_tensor_hash function not available.")
             return f"error_hash_{random.random()}"

    # --- Database Interaction Wrappers ---
    # Uses list structure input, extracts tuple and data for DB layer
    def save_tensor(self, tensor_structure: List) -> Optional[str]:
         """
         Saves tensor (list structure) to DB.
         Extracts metadata tuple and data for db.insert_veector_tensor.
         """
         tensor_id = None
         try:
             if not self.validate_tensor(tensor_structure):
                  print("Error: Attempted to save invalid tensor list structure.")
                  return None

             meta_dict = get_tensor_metadata(tensor_structure)
             meta_tuple_to_save = meta_dict.get("_encoded_metadata_v1_")
             data_payload = tensor_structure[5] if len(tensor_structure) == 6 else None

             if not meta_tuple_to_save:
                  print("Error: Cannot find embedded metadata tuple for saving.")
                  return None

             # Generate ID based on the list structure before passing tuple to DB
             # tensor_id = self.get_tensor_hash(tensor_structure) # DB generates ID based on tuple now

             # Call DB method which expects tuple and data
             tensor_id = self.db.insert_veector_tensor(meta_tuple_to_save, data_payload)
             return tensor_id
         except Exception as e:
             print(f"Error during save_tensor wrapper call (tensor_id={tensor_id}): {e}")
             return None

    # Returns list structure (as returned by veectordb v0.9.1+)
    def load_tensor(self, doc_id: str, load_knowledge: bool = False, use_mmap: bool = True) -> Optional[List]:
         """Loads tensor as list structure via db.get_veector_tensor."""
         try:
            result_structure = self.db.get_veector_tensor(
                doc_id, load_knowledge_data=load_knowledge, use_mmap=use_mmap
            )
            # Basic validation on the returned list
            if result_structure is not None and not self.validate_tensor(result_structure):
                 print(f"Warning: load_tensor received invalid list structure from DB for {doc_id}.")
                 return None
            return result_structure
         except Exception as e:
             print(f"Error during load_tensor wrapper call for {doc_id}: {e}")
             return None

    # --- load_knowledge_tensor_data (Includes Dequantization) ---
    def load_knowledge_tensor_data(self, knowledge_id: str) -> Optional[Any]:
        """
        Loads knowledge tensor data, using cache. Handles INT8 dequantization.
        """
        # 1. Check cache
        if knowledge_id in self.knowledge_cache:
            self._update_cache_access(knowledge_id)
            return self.knowledge_cache[knowledge_id]

        # 2. Load list structure with data
        loaded_structure = self.load_tensor(knowledge_id, load_knowledge=True)

        # 3. Validate and Extract
        if not loaded_structure or len(loaded_structure) != 6:
             # print(f"Debug: Knowledge structure list not found/invalid for ID {knowledge_id}")
             return None

        try:
            meta_dict = get_tensor_metadata(loaded_structure)
            loaded_data = loaded_structure[5]
        except Exception as e:
             print(f"Error extracting meta/data from loaded structure for {knowledge_id}: {e}")
             return None

        if loaded_data is None:
             # print(f"Warning: Blob data is None for knowledge {knowledge_id}.") # Reduce noise
             return None

        # 4. Check for INT8 and Dequantize
        final_data = loaded_data
        original_dtype_str = meta_dict.get("dtype", "").lower()
        is_int8 = ('int8' in original_dtype_str)

        if is_int8:
            # print(f"  Applying dequantization for INT8 tensor {knowledge_id}...") # Reduce noise
            scale = meta_dict.get("quantization_scale")
            if scale is None or scale <= 0:
                print(f"  ERROR: Missing/invalid 'quantization_scale' for INT8 tensor {knowledge_id}. Cannot dequantize!")
                return None # Cannot proceed without scale
            try:
                scale_float = float(scale)
                # Ensure loaded data is actually int8 before dequantizing
                if loaded_data.dtype != np.int8:
                     print(f"  WARN: Metadata dtype int8 mismatch for {knowledge_id}, loaded: {loaded_data.dtype}. Casting.")
                     loaded_data = loaded_data.astype(np.int8)

                dequantized_data = loaded_data.astype(np.float32) * scale_float
                final_data = dequantized_data
                # print(f"  Dequantized data shape: {final_data.shape}, dtype: {final_data.dtype}") # Reduce noise
            except Exception as dequant_e:
                print(f"  ERROR during dequantization for {knowledge_id}: {dequant_e}")
                return None
        # else: pass # Not INT8, use loaded_data as is

        # 5. Cache and Return
        if final_data is not None:
            self._add_to_cache(knowledge_id, final_data, is_knowledge=True)
        return final_data


    # --- Caching Methods ---
    def _update_cache_access(self, key: Union[str, Tuple]):
        self.cache_timestamps[key] = time.time()
        self.cache_access_count[key] = self.cache_access_count.get(key, 0) + 1

    def _evict_cache(self):
        compute_keys=list(self.compute_cache.keys())
        knowledge_keys=list(self.knowledge_cache.keys())
        total_items = len(compute_keys) + len(knowledge_keys)
        if total_items < self.cache_size:
             return
        num_to_evict = total_items - self.cache_size + 1
        all_keys_metrics=[]
        try:
            sorter=lambda k: self.cache_access_count.get(k,0) if self.eviction_strategy=="LFU" else self.cache_timestamps.get(k,0)
            all_keys_metrics=[(k, sorter(k)) for k in compute_keys+knowledge_keys]
            all_keys_metrics.sort(key=lambda item: item[1])
            keys_to_evict=[item[0] for item in all_keys_metrics[:num_to_evict]]
            for key in keys_to_evict:
                evicted_item=False
                if key in self.compute_cache:
                    del self.compute_cache[key]
                    evicted_item=True
                if key in self.knowledge_cache:
                    del self.knowledge_cache[key]
                    evicted_item=True
                if evicted_item:
                     if key in self.cache_timestamps: del self.cache_timestamps[key]
                     if key in self.cache_access_count: del self.cache_access_count[key]
        except Exception as e:
            print(f"Cache eviction error: {e}")

    def _add_to_cache(self, key: Union[str, Tuple], value: Any, is_knowledge: bool = False):
         cache = self.knowledge_cache if is_knowledge else self.compute_cache
         cache[key]=value
         self._update_cache_access(key)
         self._evict_cache()

    def clear_cache(self, clear_knowledge: bool = True, clear_compute: bool = True):
         if clear_compute: self.compute_cache.clear()
         if clear_knowledge: self.knowledge_cache.clear()
         self.cache_timestamps.clear()
         self.cache_access_count.clear()
         print("Caches cleared.")

    # --- Op Sequence Execution ---
    def _execute_op_sequence(
        self,
        ops_sequence: List[Any],
        initial_data: Any,
        knowledge_params_for_ops: Dict[str, Any], # Загруженные numpy массивы знаний
        **kw_context # Контекст вызова compute (position_ids, attention_mask и т.д.)
    ) -> Tuple[Any, List[Dict]]:
        """
        Executes op sequence. v0.6.12 - Rewritten command parser logic
        combined with argument resolution.
        """
        current_data = initial_data
        step_provenance_list = []
        step_context = {'_initial_input': initial_data}

        for i, op_command in enumerate(ops_sequence):
            step_provenance = {"step": i}; step_start = time.time(); op_tuple = None; op_code_list = None; op_func = None; is_meta_op = False;
            op_call_args_from_processor: Dict = {} # Аргументы из определения процессора
            meta_args = []
            valid_command = False # Флаг для проверки парсинга

            # --- 1. Новый Парсинг команды ---
            if not isinstance(op_command, list) or not op_command:
                 error_msg = f"Command at step {i} is not a non-empty list: {op_command}"
                 step_provenance["error"] = error_msg; step_provenance_list.append(step_provenance); return None, step_provenance_list

            # Попытка 1: "Плоский" формат [OP/META, ...]
            if isinstance(op_command[0], int):
                if len(op_command) >= 3 and all(isinstance(x, int) for x in op_command[:3]):
                    op_code_list = op_command[:3]
                    op_tuple = tuple(op_code_list)
                    is_meta_op = op_tuple in (tuple(OP_STORE), tuple(OP_LOAD), tuple(OP_LOAD_INITIAL_INPUT), tuple(OP_DEBUG_CONTEXT))

                    if is_meta_op: # [META, arg1, ...]
                        meta_args = op_command[3:]
                        op_call_args_from_processor = {}
                        valid_command = True
                    else: # [REGULAR_OP] или [REGULAR_OP, {args}]
                        if len(op_command) == 3: # [REGULAR_OP]
                            op_call_args_from_processor = {}
                            valid_command = True
                        elif len(op_command) == 4 and isinstance(op_command[3], dict): # [REGULAR_OP, {args}]
                            op_call_args_from_processor = op_command[3]
                            valid_command = True
                        # else: Invalid flat regular op format (will fail later)

            # Попытка 2: "Вложенный" формат [[OP/META], ...]
            elif isinstance(op_command[0], list):
                 op_code_container = op_command[0]
                 if len(op_code_container) == 3 and all(isinstance(x, int) for x in op_code_container):
                     op_code_list = op_code_container
                     op_tuple = tuple(op_code_list)
                     is_meta_op = op_tuple in (tuple(OP_STORE), tuple(OP_LOAD), tuple(OP_LOAD_INITIAL_INPUT), tuple(OP_DEBUG_CONTEXT))

                     if is_meta_op: # [[META], arg1, ...]
                         meta_args = op_command[1:]
                         op_call_args_from_processor = {}
                         valid_command = True
                     else: # [[REGULAR_OP]] или [[REGULAR_OP], {args}]
                         if len(op_command) == 1: # [[REGULAR_OP]]
                             op_call_args_from_processor = {}
                             valid_command = True
                         elif len(op_command) == 2 and isinstance(op_command[1], dict): # [[REGULAR_OP], {args}]
                             op_call_args_from_processor = op_command[1]
                             valid_command = True
                         # else: Invalid nested regular op format (will fail later)

            # Если ни один формат не подошел или op_code_list не определен
            if not valid_command or op_code_list is None:
                 error_msg=f"Cannot parse OP CODE / Invalid command format at step {i}: {op_command}";
                 step_provenance["error"]=error_msg; step_provenance_list.append(step_provenance); return None, step_provenance_list

            # Запись в provenance
            if op_tuple: step_provenance["op"] = op_tuple
            if meta_args: step_provenance["meta_args"] = meta_args
            if op_call_args_from_processor: step_provenance["op_args"] = op_call_args_from_processor
            # --- Конец Парсинга ---


            # --- 2. Выполнение ---
            if is_meta_op: # --- Выполнение Meta Op ---
                # (Логика Meta Op без изменений)
                if op_tuple == tuple(OP_STORE):
                    if not meta_args or not isinstance(meta_args[0], str): error_msg = f"OP_STORE needs str arg"; step_provenance["error"] = error_msg; step_provenance_list.append(step_provenance); return None, step_provenance_list
                    step_context[meta_args[0]] = current_data
                elif op_tuple == tuple(OP_LOAD):
                    if not meta_args or not isinstance(meta_args[0], str): error_msg = f"OP_LOAD needs str arg"; step_provenance["error"] = error_msg; step_provenance_list.append(step_provenance); return None, step_provenance_list
                    var_name = meta_args[0]
                    if var_name in step_context: current_data = step_context[var_name]
                    else: error_msg = f"LOAD failed: Var '{var_name}' not in context"; step_provenance["error"] = error_msg; step_provenance_list.append(step_provenance); return None, step_provenance_list
                elif op_tuple == tuple(OP_LOAD_INITIAL_INPUT): current_data = step_context.get('_initial_input')
                elif op_tuple == tuple(OP_DEBUG_CONTEXT): print(f"  DEBUG META [{i}] CONTEXT:"); [print(f"    '{k}': type={type(v)}, shape={getattr(v, 'shape', 'N/A')}") for k, v in step_context.items()]

            else: # --- Выполнение Regular Op ---
                op_func = self.core_ops.get(op_tuple)
                if not op_func: error_msg = f"Op {op_tuple} not found"; step_provenance["error"] = error_msg; step_provenance_list.append(step_provenance); return None, step_provenance_list

                # --- Разрешение аргументов (Логика из v0.6.9) ---
                resolved_args = {}
                if isinstance(op_call_args_from_processor, dict):
                    for arg_name, value_source_name in op_call_args_from_processor.items():
                        resolved_value = None; found = False
                        if isinstance(value_source_name, str):
                            if value_source_name in knowledge_params_for_ops: resolved_value = knowledge_params_for_ops[value_source_name]; found = True
                            elif value_source_name in step_context: resolved_value = step_context[value_source_name]; found = True
                            elif value_source_name in kw_context: resolved_value = kw_context[value_source_name]; found = True
                        else: resolved_value = value_source_name; found = True
                        resolved_args[arg_name] = resolved_value
                        if not found and isinstance(value_source_name, str): print(f"WARN: Arg '{arg_name}' source '{value_source_name}' not found for Op {op_tuple}. Passing None.")
                # --- Конец Разрешения аргументов ---

                # Собираем финальный словарь kw для функции операции
                op_kw = {**resolved_args, **kw_context}

                try:
                    # Вызываем функцию операции
                    current_data = op_func(current_data, **op_kw)
                except Exception as e:
                    error_msg = f"Op {op_tuple} at step {i} failed: {e}"; step_provenance["error"] = str(e);
                    print(f"--- Exception during Op {op_tuple} execution (Step {i}) ---"); traceback.print_exc(); print(f"--- End Exception ---")
                    step_provenance_list.append(step_provenance); return None, step_provenance_list

            # --- 3. Завершение шага ---
            step_provenance["duration_ms"] = (time.time() - step_start) * 1000
            step_provenance_list.append(step_provenance)
            if isinstance(current_data, dict) and current_data.get('status') == 'error':
                 error_msg = f"Op {op_tuple} returned error status: {current_data.get('error')}"; step_provenance["error"] = error_msg; return None, step_provenance_list

        return current_data, step_provenance_list



    # --- Knowledge Selection (Адаптировано для структуры списка) ---
    def _select_knowledge_tensors(self,
                                  processor_structure: List, # Принимает список
                                  context: Dict
                                 ) -> Dict[str, str]:
        """
        Находит ID тензоров знаний. v0.6.3 - Исправлен приоритет target_knowledge_group.
        Работает со структурой списка.
        """
        # Блок try для перехвата ошибок при доступе к структуре
        try:
            # Используем СТАРЫЕ геттеры для структуры списка
            interface = get_tensor_interface(processor_structure) or {}
            processor_tags_list = get_tensor_tags(processor_structure)
            processor_coord = get_tensor_coord(processor_structure)
            # Проверяем, что координаты извлечены
            if not processor_coord:
                raise ValueError("Processor coordinates not found in structure")
        except Exception as e:
            # Логируем ошибку и возвращаем пустой словарь
            print(f"Error selecting knowledge: Cannot access metadata from processor structure: {e}")
            return {}

        # Получаем список необходимых знаний из интерфейса
        knowledge_needs = interface.get("knowledge_needed", [])
        # Если знания не нужны, возвращаем пустой словарь
        if not knowledge_needs:
            return {}

        # Получаем требуемый nest из контекста
        required_nest = context.get("required_nest")
        processor_tags_set = set(processor_tags_list) # Множество для быстрой проверки

        # --- Определение Целевой Группы Знаний ---
        # 1. ПРИОРИТЕТ: Используем группу из контекста, если она явно передана
        target_knowledge_group = context.get("target_knowledge_group")
        model_tag = None # Тег модели процессора

        # Если группа не передана в контексте, пытаемся определить по тегу модели процессора
        if target_knowledge_group is None:
            print(f"  WARN: target_knowledge_group not in context. Determining from processor tags...")
            for tag in processor_tags_list:
                if 10 <= tag <= 19: # Диапазон тегов моделей
                     model_tag = tag;
                     # Старая логика определения группы по тегу модели (Fallback)
                     if model_tag == TAG_MODEL_QWEN2: target_knowledge_group = GROUP_IDX_QWEN_KNOWLEDGE # 100
                     elif model_tag == TAG_MODEL_DEEPSEEK: target_knowledge_group = GROUP_IDX_DEEPSEEK_KNOWLEDGE # 102
                     # Добавить другие маппинги если нужно
                     break # Нашли тег модели
            if target_knowledge_group is None:
                  print(f"  WARN: Could not determine target knowledge group from processor tags. Searching without group filter.")
        else:
             print(f"  Using target_knowledge_group from context: {target_knowledge_group}")
             # Можно на всякий случай найти тег модели
             for tag in processor_tags_list:
                 if 10 <= tag <= 19: model_tag = tag; break
        # --- Конец Определения Группы ---

        print(f"  Targeting Knowledge Group: {target_knowledge_group}, Nest: {required_nest} (Processor Model Tag: {model_tag})")

        # Фильтр для БД
        db_coord_filter = {}
        if target_knowledge_group is not None:
            db_coord_filter["group"] = target_knowledge_group
        # Определяем nest для фильтрации
        target_nest = required_nest if isinstance(required_nest, int) else processor_coord.nest
        db_coord_filter["nest"] = target_nest

        # Ищем кандидатов (veectordb v0.9.x возвращает Dict[str, List])
        all_candidate_structures: Dict[str, List] = self.db.find_active_tensors(
            tensor_type="knowledge",
            coord_filter=db_coord_filter # Передаем фильтр
        )
        print(f"  Found {len(all_candidate_structures)} candidates via index filter G={db_coord_filter.get('group')}, N={db_coord_filter.get('nest')}.")

        # Фильтруем кандидатов по тегам
        selected_knowledge_map = {}
        for need in knowledge_needs:
            param_name = need.get("param_name"); base_tags_needed = need.get("tags", [])
            is_optional = need.get("optional", False)
            if not param_name or not base_tags_needed: continue

            best_candidate_id = None
            # Используем теги из need как есть (они должны быть полными)
            query_tags_set = set(base_tags_needed)
            # print(f"  Searching for '{param_name}' with required tags: {query_tags_set}") # Debug

            found_match_for_need = False
            for cand_id, cand_structure in all_candidate_structures.items():
                 try:
                     # Используем СТАРЫЙ геттер для структуры списка
                     tensor_tags_list = get_tensor_tags(cand_structure)
                     # Проверяем, содержит ли тензор ВСЕ необходимые теги
                     if query_tags_set.issubset(set(tensor_tags_list)):
                          best_candidate_id = cand_id
                          found_match_for_need = True
                          break # Нашли лучший (первый) подходящий в отфильтрованном списке
                 except Exception as e:
                      # Логируем ошибку обработки кандидата, но продолжаем поиск
                      print(f"    Warn: Error processing candidate {cand_id} for need '{param_name}': {e}")
                      continue # Пропускаем невалидного кандидата

            # Обрабатываем результат поиска для текущей потребности
            if found_match_for_need:
                 selected_knowledge_map[param_name] = best_candidate_id
                 # print(f"    Selected '{param_name}': {best_candidate_id}") # Debug log
            elif not is_optional:
                 # Если обязательное знание не найдено, выбрасываем ошибку
                 err_msg = f"Missing REQUIRED knowledge for '{param_name}' with tags {query_tags_set} in G={target_knowledge_group}, N={target_nest}."
                 print(f"ERROR: {err_msg}")
                 raise ValueError(err_msg) # Прерываем выполнение compute
            # else: print(f"    Warn: Optional knowledge '{param_name}' not found.") # Необязательное не найдено

        # Возвращаем карту найденных ID знаний
        return selected_knowledge_map

    # --- Early Exit Check (Использует СТАРЫЙ геттер) ---
    def _check_early_exit(self, tensor_structure: List, result_data: Any, context: Dict) -> bool:
        try: exit_gates = get_tensor_exit_gates(tensor_structure) or [] # Старый геттер
        except Exception: return False
        if not exit_gates: return False
        check_context = {**context, "current_result": result_data}
        for gate_ref in exit_gates:
            try:
                gate_triggered = False
                if isinstance(gate_ref, str):
                    gate_result = self.compute(gate_ref, context={"input_data": check_context})
                    gate_triggered = (gate_result.get("status") == "completed" and isinstance(gate_result.get("data"), bool) and gate_result["data"])
                elif callable(gate_ref): gate_triggered = gate_ref(check_context)
                if gate_triggered: return True
            except Exception as e: print(f"Error checking exit gate {gate_ref}: {e}"); continue
        return False

    # --- compute Method (Работает со СТРУКТУРОЙ СПИСКА) ---
    def compute(self, processor_id: str, context: Optional[Dict] = None) -> Dict:
        """Основной метод вычислений v0.6.2 - работает со структурой списка."""
        start_time = time.time(); context = context or {}; provenance = { "processor_id": processor_id, "steps": [], "timestamp_start": datetime.now().isoformat(), "context_received": {k: v for k, v in context.items() if k != 'input_data'} }

        # Кэширование
        input_data = context.get("input_data"); input_hash_str = "no_input"
        if input_data is not None:
             try: input_hash_str = hashlib.sha256(pickle.dumps(input_data)).hexdigest()[:8]
             except Exception: input_hash_str = f"unhashable_{type(input_data).__name__}"
        state_id = context.get("state_id"); required_nest = context.get("required_nest", "default")
        cache_key = (processor_id, required_nest, state_id if state_id else input_hash_str)
        if cache_key in self.compute_cache:
            cached_data = self.compute_cache[cache_key]; self._update_cache_access(cache_key); provenance["status"] = "cached"; provenance["timestamp_end"] = datetime.now().isoformat(); provenance["duration_ms"] = (time.time() - start_time) * 1000;
            return {"data": cached_data, "provenance": provenance, "status": "completed"}

        # Загрузка СТРУКТУРЫ СПИСКА процессора
        processor_structure = self.load_tensor(processor_id, load_knowledge=False)
        if not processor_structure:
            error_msg = f"Processor {processor_id} structure list not found/invalid."; provenance["error"] = error_msg; provenance["status"] = "error"
            return {"data": None, "provenance": provenance, "status": "error"}

        # Валидация и извлечение данных с помощью СТАРЫХ геттеров
        try:
            if not self.validate_tensor(processor_structure): raise ValueError("Invalid processor structure list.")
            tensor_type_str = get_tensor_type(processor_structure)
            status_str = get_tensor_status(processor_structure)
            coord_obj = get_tensor_coord(processor_structure)
            meta_dict = get_tensor_metadata(processor_structure)
            evo_version = meta_dict.get("evolutionary_version", 1)
            if tensor_type_str not in ["processor", "converter"]: raise ValueError(f"Tensor {processor_id} not processor/converter.")
            if status_str == "archived": raise ValueError(f"Processor {processor_id} is archived.")
            if not coord_obj: raise ValueError(f"Cannot get coordinates for {processor_id}")
            provenance.update({"coord": str(coord_obj), "evo_version": evo_version})
        except Exception as e:
             error_msg = f"Error processing loaded structure for {processor_id}: {e}"; provenance["error"] = error_msg; provenance["status"] = "error"
             return {"data": None, "provenance": provenance, "status": "error"}

        # Определение Ops Sequence (используем СТАРЫЙ геттер)
        ops_sequences_dict = get_processor_ops_sequences(processor_structure) or {}
        precision_key = 'default'; # ... (логика определения precision_key) ...
        if isinstance(required_nest, int):
            if required_nest == 0: precision_key = TAG_PREC_INT8
            elif required_nest == 1: precision_key = TAG_PREC_FLOAT16
            elif required_nest == 2: precision_key = TAG_PREC_FLOAT32
        elif isinstance(required_nest, int) and 20 <= required_nest <= 29: precision_key = required_nest
        elif required_nest != "default": precision_key = required_nest
        ops_sequence = ops_sequences_dict.get(precision_key)
        if ops_sequence is None and precision_key != 'default': ops_sequence = ops_sequences_dict.get('default')

        if not ops_sequence:
             result_data = context.get("input_data"); provenance["status"] = "completed (no ops)";
             provenance["timestamp_end"] = datetime.now().isoformat(); provenance["duration_ms"] = (time.time() - start_time) * 1000
             self._add_to_cache(cache_key, result_data); return {"data": result_data, "provenance": provenance, "status": "completed"}

        provenance["ops_sequence_length"] = len(ops_sequence); provenance["precision_key_used"] = str(precision_key)

        # Выбор и Загрузка Знаний (передаем СТРУКТУРУ СПИСКА)
        load_start_time = time.time(); loaded_knowledge_by_param_name = {}; knowledge_ids_loaded = []
        try:
            selected_knowledge_map = self._select_knowledge_tensors(processor_structure, context)
            for param_name, knowledge_id in selected_knowledge_map.items():
                 data = self.load_knowledge_tensor_data(knowledge_id) # Включает деквантование
                 if data is None: raise ValueError(f"Failed load knowledge blob {knowledge_id} ({param_name})")
                 loaded_knowledge_by_param_name[param_name] = data; knowledge_ids_loaded.append(knowledge_id)
        except Exception as load_err:
            error_msg = f"Knowledge select/load fail: {load_err}"; provenance["error"] = error_msg; provenance["status"] = "error"
            return {"data": None, "provenance": provenance, "status": "error"}
        provenance["knowledge_load_ms"] = (time.time() - load_start_time) * 1000
        provenance["loaded_knowledge_ids"] = knowledge_ids_loaded

        # Начальные данные и Состояние
        current_step = 0; current_data = None; state_id_used = context.get("state_id")
        if state_id_used:
             intermediate_state_data = self.load_knowledge_tensor_data(state_id_used)
             if not isinstance(intermediate_state_data, dict):
                 error_msg = f"Invalid state data for {state_id_used}"; provenance["error"] = error_msg; provenance["status"] = "error";
                 return {"data": None, "provenance": provenance, "status": "error"}
             current_data = intermediate_state_data.get("data"); current_step = intermediate_state_data.get("next_step", 0);
             provenance["resumed_from_state"] = state_id_used; provenance["resumed_from_step"] = current_step;
        else: current_data = context.get("input_data")

        # Фильтры
        filters = get_tensor_filters(processor_structure) or []
        if filters and current_data is not None: pass # Placeholder

        # Выполнение Последовательности Операций
        max_steps_per_call = context.get("max_steps", len(ops_sequence))
        steps_executed_this_call = 0; exec_start_time = time.time(); ops_to_execute = []; result_data = None; step_provenance_list = []
        if 0 <= current_step < len(ops_sequence):
             end_step = min(current_step + max_steps_per_call, len(ops_sequence)); ops_to_execute = ops_sequence[current_step : end_step]
        if ops_to_execute:
            try:
                exec_context = {"_current_processor_id": processor_id, **context}
                result_data, step_provenance_list = self._execute_op_sequence( ops_to_execute, current_data, loaded_knowledge_by_param_name, **exec_context )
                provenance["steps"].extend(step_provenance_list); steps_executed_this_call = len(step_provenance_list)
                if step_provenance_list and "error" in step_provenance_list[-1]:
                     error_msg = f"Op sequence failed: {step_provenance_list[-1].get('error','Unknown')}"; provenance["error"] = error_msg; provenance["status"] = "error"
                     return {"data": None, "provenance": provenance, "status": "error"}
                else: current_step += steps_executed_this_call
            except Exception as exec_e: error_msg = f"Error during op sequence exec: {exec_e}"; provenance["error"] = error_msg; provenance["status"] = "error"; return {"data": None, "provenance": provenance, "status": "error"}
        else: result_data = current_data
        provenance["execution_ms"] = (time.time() - exec_start_time) * 1000

        # Обработка Состояния (вызов гибридного create_tensor)
        if current_step < len(ops_sequence):
            state_to_save = {"data": result_data, "next_step": current_step}
            state_coord = TensorCoordinate( layer=STATE_TENSOR_LAYER, group=coord_obj.group, nest=coord_obj.nest, x=random.randint(10000,99999), y=coord_obj.y, z=coord_obj.z )
            try:
                # Вызываем ГИБРИДНЫЙ create_tensor, он вернет СТРУКТУРУ СПИСКА
                state_tensor_structure = self.create_tensor(
                     coord=state_coord, tensor_type="state",
                     knowledge_data=state_to_save, status="temporary", parents=[processor_id]
                )
                # Вызываем save_tensor, который ожидает СТРУКТУРУ СПИСКА
                state_id_new = self.save_tensor(state_tensor_structure)
                if not state_id_new: raise RuntimeError("Failed to save state tensor.")
                provenance["status"] = "pending"; provenance["next_state_id"] = state_id_new
                provenance["timestamp_end"] = datetime.now().isoformat(); provenance["duration_ms"] = (time.time() - start_time) * 1000
                return {"data": None, "provenance": provenance, "status": "pending", "state_id": state_id_new}
            except Exception as state_e:
                 error_msg = f"Failed create/save state: {state_e}"; provenance["error"] = error_msg; provenance["status"] = "error";
                 return {"data": None, "provenance": provenance, "status": "error"}

        # Ранний Выход (используем СТАРЫЙ геттер)
        if self._check_early_exit(processor_structure, result_data, context): # Передаем структуру списка
            provenance["status"] = "early_exit"; provenance["timestamp_end"] = datetime.now().isoformat(); provenance["duration_ms"] = (time.time() - start_time) * 1000
            self._add_to_cache(cache_key, result_data);
            return {"data": result_data, "provenance": provenance, "status": "completed"}

        # Завершение
        provenance["status"] = "completed"; provenance["timestamp_end"] = datetime.now().isoformat(); provenance["duration_ms"] = (time.time() - start_time) * 1000
        self._add_to_cache(cache_key, result_data);

        return {"data": result_data, "provenance": provenance, "status": "completed"}


    # --- Spawning / Evolution (Адаптировано для структуры списка) ---
    def spawn_tensor(self, parent_id: str, strategy: str = "inherit", context: Optional[Dict] = None) -> Optional[str]:
        """Создает дочерний тензор. Работает со структурой списка."""
        parent_structure = self.load_tensor(parent_id, load_knowledge=False)
        if not parent_structure or not self.validate_tensor(parent_structure):
            print(f"Error spawn: Parent {parent_id} not found or invalid list structure.")
            return None

        try:
            # Извлекаем инфо с помощью СТАРЫХ геттеров
            parent_type_str = get_tensor_type(parent_structure)
            parent_coord_obj = get_tensor_coord(parent_structure)
            parent_tags = get_tensor_tags(parent_structure)
            parent_interface = get_tensor_interface(parent_structure)
            parent_ops_sequences = get_processor_ops_sequences(parent_structure)
            parent_filters = get_tensor_filters(parent_structure)
            parent_exit_gates = get_tensor_exit_gates(parent_structure)
            parent_meta_dict = get_tensor_metadata(parent_structure)
            parent_dtype_str = parent_meta_dict.get("dtype")
            parent_shape = parent_meta_dict.get("shape")
            if not parent_coord_obj: raise ValueError("Cannot get parent coordinates")
        except Exception as e: print(f"Error spawn: Failed extraction from parent {parent_id}: {e}"); return None

        # Генерация координат и наследование
        child_coord = self._generate_next_coords(parent_coord_obj)
        child_metadata_extra = {"spawn_strategy": strategy}
        child_type = parent_type_str; child_knowledge_data = None
        child_tags = list(parent_tags); child_interface = parent_interface.copy() if parent_interface else None
        child_ops_sequences = parent_ops_sequences.copy() if parent_ops_sequences else None
        child_filters = list(parent_filters) if parent_filters else None
        child_exit_gates = list(parent_exit_gates) if parent_exit_gates else None
        child_dtype = parent_dtype_str; child_shape = parent_shape

        # Применение стратегии
        try:
            parent_has_blob = has_blob_data(parent_structure)
            if parent_type_str == "knowledge":
                if strategy == "inherit" or not parent_has_blob: child_knowledge_data = None
                elif strategy == "mutate_knowledge":
                    parent_data = self.load_knowledge_tensor_data(parent_id) # Загрузит и деквантует
                    if parent_data is not None and isinstance(parent_data, np.ndarray):
                        scale = context.get("mutation_scale", 0.05) if context else 0.05
                        noise = np.random.normal(0, scale, parent_data.shape).astype(parent_data.dtype)
                        child_knowledge_data = parent_data + noise; child_dtype = child_knowledge_data.dtype; child_shape = child_knowledge_data.shape
                        child_tags = [t for t in child_tags if not (20 <= t <= 29)]; float_prec_tag = DTYPE_MAPPING.get(child_dtype, TAG_PREC_FLOAT16); child_tags.append(float_prec_tag)
                    else: raise ValueError("Cannot mutate missing/invalid knowledge.")
                elif strategy == "distill_knowledge": # Квантуем в int8
                    parent_data = self.load_knowledge_tensor_data(parent_id)
                    if parent_data is not None:
                        target_format_str = "int8"; scale = None
                        if np.issubdtype(parent_data.dtype, np.floating):
                             abs_max = np.max(np.abs(parent_data)); scale = abs_max / 127.0 if abs_max > 1e-9 else 1.0
                             distilled_data = np.round(parent_data / max(scale, 1e-9)).astype(np.int8)
                             child_knowledge_data = distilled_data; child_dtype = np.int8; child_shape = distilled_data.shape
                             new_prec_tag = TAG_PREC_INT8; child_tags = [t for t in child_tags if not (20 <= t <= 29)]; child_tags.append(new_prec_tag)
                             child_metadata_extra["quantization_scale"] = float(scale)
                        else: raise ValueError(f"Cannot distill non-float data to {target_format_str}")
                    else: raise ValueError("Cannot distill missing knowledge.")
            # ... (другие стратегии и типы) ...
        except Exception as e: print(f"Error applying spawn strategy '{strategy}': {e}"); return None

        # --- Создание Дочернего Тензора (вызов гибридного create_tensor) ---
        try:
            child_tensor_structure = self.create_tensor(
                coord=child_coord, tensor_type=child_type, tags=child_tags,
                interface=child_interface, ops_sequences=child_ops_sequences,
                filters=child_filters, exit_gates=child_exit_gates,
                knowledge_data=child_knowledge_data, dtype=child_dtype, shape=child_shape,
                name_id=-1, parents=[parent_id], evolutionary_version=1, status="active",
                metadata_extra=child_metadata_extra
            )
            # --- Сохранение Дочернего Тензора (передаем структуру списка) ---
            child_id = self.save_tensor(child_tensor_structure)

            if child_id: print(f"Spawn successful: Child {child_id} from {parent_id}")
            else: print(f"Error spawn: Failed to save child tensor.")
            return child_id
        except Exception as e: print(f"Error during child tensor creation/saving: {e}"); return None

    # --- Вспомогательные и специальные методы ---
    def _distill_knowledge_data(self, data: Any, target_format: str) -> Any:
        # (Логика без изменений)
        print(f"Distilling data to {target_format} (Placeholder)..."); # ...
        if isinstance(data, np.ndarray):
            if target_format == "int8" and np.issubdtype(data.dtype, np.floating): scale = np.max(np.abs(data))/127.0 if np.max(np.abs(data))>1e-9 else 1.0; return np.round(data/max(scale, 1e-9)).astype(np.int8)
            elif target_format == "float16": return data.astype(np.float16)
            elif target_format == "bfloat16": return data.astype(np.float16)
        return data
    def _generate_next_coords(self, c: TensorCoordinate) -> TensorCoordinate:
        return TensorCoordinate( layer=c.layer, group=c.group, nest=c.nest, x=c.x, y=c.y, z=c.z + 1 )
    def _op_conditional_if(self, data: Any, **kw) -> Any: # Placeholder
        c=bool(data[0]) if isinstance(data,(list,np.ndarray)) and len(data)>0 else bool(data);t=data[1] if isinstance(data,(list,np.ndarray)) and len(data)>1 else None;f=data[2] if isinstance(data,(list,np.ndarray)) and len(data)>2 else None;return t if c else f
    def _op_loop_multiply(self, data: Any, **kw) -> Any: # Placeholder
        v=data;n=1; #... (logic) ...
        if isinstance(v,(int,float,complex,np.number)): return v*n
        elif isinstance(v, np.ndarray): return v*n
        else: return [v]*n
    def _op_choice_select(self, data: Any, **kw) -> Any: # Placeholder
        i=0;o=[]; #... (logic) ...
        return o[i] if 0<=i<len(o) else None
    def _op_trigger_reason(self, data: Any, **kw) -> Any: return data # Placeholder
    def _op_graph_dfs(self, data: Any, **kw) -> Any: return [] # Placeholder
    def _op_output_print(self, data: Any, **kw): return data # Placeholder
    def _quantum_op_placeholder(self, data: Any, op_type: str, **kw) -> Any: return data # Placeholder
    def _get_tensor_manager(self) -> Optional[Any]: return None
    def handle_task(
        self,
        task_description: Any,
        input_data: Any,
        resources: Dict,
        priority: float
    ) -> Optional[Dict]:
        """Placeholder for handling tasks via TensorManager."""
        # Эта функция требует интеграции с TensorManager, который еще не реализован.
        print("Warning: handle_task called, but TensorManager is not implemented.")
        # Возвращаем None как заглушку
        return None
    def get_available_processors(self, filter_dict=None) -> List[str]: # Uses find_active_tensors
        try: p=self.db.find_active_tensors(tensor_type="processor", coord_filter=filter_dict); return list(p.keys())
        except Exception: return []
    def get_tensor_structure(self, tensor_id: str) -> Optional[List]: # <<< Возвращает список
        return self.load_tensor(tensor_id, load_knowledge=False)

# --- Пример использования (Требует обновления) ---
if __name__ == "__main__":
    print("\n--- Veector Core Example (v0.6.11) ---")
    print("--- !!! EXAMPLE NEEDS UPDATE FOR HYBRID LIST STRUCTURE !!! ---")
    print("\n--- Example Finished (Needs Update) ---")