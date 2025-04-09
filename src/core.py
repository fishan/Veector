# FILE: core.py
# Version: 0.7.16 (Fixed Arg Resolution for Placeholder)

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
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union
import importlib # Dlja dinamicheskogo importa

# --- Version ---
# ИЗМЕНЕНО: Обновлена версия и описание
CORE_VERSION = "0.7.17" # Core operations update

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
    # from qiskit import QuantumCircuit # Primer, esli by ispol'zovalsja
    QISKIT_AVAILABLE = False
except ImportError:
    QISKIT_AVAILABLE = False

# --- Ожидаемые версии зависимостей ---
TENSORS_VERSION_REQ = "0.7.6"
VEECTORDB_VERSION_REQ = "0.9.8"
OPERATIONS_VERSION_REQ = "0.8.9"

VeectorDB_defined = False
# --- Veector Project Imports ---
try:
    from veectordb import VeectorDB, VEECTORDB_VERSION
    print(f"  Imported VeectorDB (v{VEECTORDB_VERSION})")
    if VEECTORDB_VERSION < VEECTORDB_VERSION_REQ:
         raise ImportError(f"Core v{CORE_VERSION} requires VeectorDB v{VEECTORDB_VERSION_REQ}+, found v{VEECTORDB_VERSION}")

    # ИЗМЕНЕНО: Добавлен импорт validate_tensor_tuple
    from tensors import (
        TENSORS_VERSION, TensorCoordinate, create_tensor, validate_tensor,
        validate_tensor_tuple, # <--- ДОБАВЛЕНО
        get_tensor_hash, get_tensor_metadata, get_tensor_coord, get_tensor_type,
        get_tensor_status, get_tensor_tags, get_tensor_interface,
        get_processor_ops_sequences, get_tensor_filters, get_tensor_exit_gates,
        has_blob_data, get_tensor_parents, get_tensor_op_channels,
        TAG_PREC_INT8, TAG_PREC_FLOAT16, TAG_PREC_FLOAT32,
        GROUP_IDX_QWEN_KNOWLEDGE, GROUP_IDX_DEEPSEEK_KNOWLEDGE,
        TAG_MODEL_QWEN2, TAG_MODEL_DEEPSEEK,
        TAG_TYPE_PROCESSOR, TAG_TYPE_KNOWLEDGE, TAG_TYPE_CONVERTER, TAG_TYPE_STATE,
        TAG_COMP_EMBEDDING, TAG_COMP_LM_HEAD,
        DTYPE_MAPPING
    )
    print(f"  Imported tensors (v{TENSORS_VERSION})")
    if TENSORS_VERSION < TENSORS_VERSION_REQ:
         raise ImportError(f"Core v{CORE_VERSION} requires tensors v{TENSORS_VERSION_REQ}+, found v{TENSORS_VERSION}")

    # Importiruem bazovye operacii
    from operations import * # Import all operations
    print(f"  Imported operations (v{OPERATIONS_VERSION})")
    if OPERATIONS_VERSION < OPERATIONS_VERSION_REQ:
        raise ImportError(f"Core v{CORE_VERSION} requires operations v{OPERATIONS_VERSION_REQ}+, found v{OPERATIONS_VERSION}")

    from memory import Memory, MEMORY_VERSION
    print(f"  Imported Memory (v{MEMORY_VERSION})")

    # Staticheskij (opcional'nyj) import dlja Qwen2 Ops
    try:
        import veector_models.qwen2.ops as qwen2_ops_module_static
        print("  Found optional module: veector_models.qwen2.ops")
        QWEN2_OPS_AVAILABLE = True
    except ImportError:
        print("  Optional module not found: veector_models.qwen2.ops")
        QWEN2_OPS_AVAILABLE = False

    print("Core components imported successfully.")
    VeectorDB_defined = True

# --- Handle Import Errors Gracefully ---
except ImportError as e:
    print(f"---!!! FATAL ERROR (ImportError) !!! ---")
    print(f"Specific error: {e}")
    print(f"Ensure files (tensors v{TENSORS_VERSION_REQ}+, veectordb v{VEECTORDB_VERSION_REQ}+, operations v{OPERATIONS_VERSION_REQ}+, memory) are OK.")
    print(f"-----------------------------------------")
    # Define dummies to avoid further errors if possible
    class VeectorDB: pass
    VEECTORDB_VERSION = "dummy"
    class TensorCoordinate: pass
    TENSORS_VERSION = "dummy"
    def create_tensor(*a,**kw): return []
    def validate_tensor(t): return False
    def validate_tensor_tuple(t): return False # Dummy
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

# --- Operation Code Constants (Без изменений) ---
# Nizkourovnevye
OP_SUM=[0,0,0]; OP_SUBTRACT=[0,0,1]; OP_ADD=[0,0,2]; OP_MULTIPLY=[0,1,0]
OP_DIVIDE=[0,1,1]; OP_SQRT=[0,2,0]; OP_POWER=[0,2,1]; OP_ABS=[0,3,0]
OP_MOD=[0,5,0]; OP_FLOOR=[0,6,0]; OP_CEIL=[0,6,1]; OP_SIN=[1,0,0]
OP_COS=[1,0,1]; OP_TAN=[1,1,0]; OP_COT=[1,1,1]; OP_ASIN=[1,2,0]
OP_ACOS=[1,2,1]; OP_ATAN=[1,3,0]; OP_GREATER=[2,0,0]; OP_EQUAL=[2,0,1]
OP_AND=[2,1,0]; OP_OR=[2,1,1]; OP_NOT=[2,2,0]; OP_XOR=[2,3,0]
OP_NAND=[2,4,0]; OP_NOR=[2,4,1]; OP_IF=[3,0,0]; OP_LOOP_MULT=[4,0,0]
OP_CHOICE=[7,0,0]; OP_RAND_UNIFORM=[5,1,0]; OP_RAND_NORMAL=[5,1,1]
OP_MEDIAN=[5,2,0]; OP_PRINT=[8,0,0]; OP_IDENTITY=[9,0,0]
OP_TRIGGER_REASON=[10,0,0]; OP_DFS=[15,0,0]; OP_MEAN=[16,0,0]
OP_STDDEV=[16,1,0]; OP_RELU=[18,0,0]; OP_SIGMOID=[18,1,0]
OP_SOFTMAX=[18,2,0]; OP_LEAKY_RELU=[18,3,0]; OP_SILU=[18,4,0]
OP_GELU=[40,5,0]; OP_EXP_SMOOTHING=[19,0,0]; OP_NORMALIZE_01=[20,0,0]
OP_INTERPOLATE=[20,1,0]; OP_LAYER_NORM=[40,1,0]; OP_BATCH_NORM=[40,4,0]
OP_DROPOUT=[40,3,0]; OP_GET_Q_ROT=[40,7,1]; OP_GET_K_ROT=[40,7,2]
OP_MATRIX_MULTIPLY=[30,0,0]; OP_DETERMINANT=[30,1,0]; OP_EIGENVALUES=[30,2,0]
OP_CONVOLUTION=[30,3,0]; OP_TRANSPOSE=[30,4,0]; OP_INVERSE=[30,5,0]
OP_TRACE=[30,6,0]; OP_ATTENTION_MULTIHEAD=[40,2,0]; OP_EMBEDDING_LOOKUP=[40,6,0]
OP_APPLY_ROPE=[40,7,0]; OP_RESHAPE_HEADS=[40,9,0]; OP_REPEAT_KV_HEADS=[40,9,1]
OP_SCALED_DOT_PROD_ATTN=[40,9,2]; OP_MERGE_HEADS=[40,9,3]; OP_ADD_BIAS=[0,0,3]
OP_UPDATE_KV_CACHE = [40, 10, 0]; OP_CREATE_CAUSAL_MASK = [40, 10, 1]
OP_RESIDUAL_ADD=OP_ADD; OP_LINEAR=OP_MATRIX_MULTIPLY; OP_FINAL_NORM=OP_LAYER_NORM
OP_LINEAR_HEAD=OP_LINEAR; OP_QUANTUM_HADAMARD=[50,0,0]; OP_QUANTUM_PAULI_X=[50,0,1]
OP_QUANTUM_CNOT=[50,1,0]; OP_QUANTUM_MEASURE=[50,2,0]; OP_QUANTUM_SUPERPOS=[50,3,0]
OP_QUANTUM_ENTANGLE=[50,4,0];
# Meta
META_OP_CATEGORY=99; OP_STORE=[99,0,0]; OP_LOAD=[99,0,1]; OP_LOAD_INITIAL_INPUT=[99,0,3];
OP_DEBUG_CONTEXT=[99,1,0]; OP_MAKE_TUPLE = [99, 2, 0]
OP_GET_TUPLE_ELEM_0 = [99, 3, 0]
OP_GET_TUPLE_ELEM_1 = [99, 3, 1]
OP_GET_TUPLE_ELEM_2 = [99, 3, 2]
OP_GET_TUPLE_ELEM_3 = [99, 3, 3]
# Vysokourovnevye Qwen2
OP_QWEN2_RMSNORM = [300, 0, 0]
OP_QWEN2_ATTENTION = [300, 1, 0]
OP_QWEN2_MLP = [300, 2, 0]

# --- Helper dlja logirovanija tenzorov vnutri core (Без изменений) ---
def _log_core_tensor_stats(label: str, data: Any, log_sample: bool = False):
    """Vspomogatel'naja funkcija dlja logirovanija statistiki tenzora vnutri core."""
    if data is None:
        print(f"  [CORE DBG] {label}: None")
        return
    prefix = f"  [CORE DBG] {label}:"
    if isinstance(data, np.ndarray):
        has_nan = np.any(np.isnan(data))
        print(f"{prefix} shape={data.shape}, dtype={data.dtype}, NaN={has_nan}")
        if (has_nan or log_sample) and data.size > 0:
            try:
                sample_slice = data.flatten()[:5]
                print(f"           Sample: {sample_slice}")
            except Exception as e:
                print(f"           Error getting sample: {e}")
    elif isinstance(data, tuple):
         print(f"{prefix} type=tuple, len={len(data)}")
    elif isinstance(data, list):
         print(f"{prefix} type=list, len={len(data)}")
    else:
         print(f"{prefix} type={type(data).__name__}")


class Veector:
    """ Core execution engine. v0.7.13 """
    def __init__(self,
                 db_dir: Union[str, Path] = "data/db",
                 cache_size: int = DEFAULT_CACHE_SIZE,
                 eviction_strategy: str = DEFAULT_EVICTION_STRATEGY,
                 use_memory_module: bool = False,
                 p2p_node: Optional[Any] = None,
                 ipfs_enabled: bool = False,
                 ipfs_address: str = DEFAULT_IPFS_ADDRESS,
                 initial_index_path: Optional[Union[str, Path]] = None
                ):
        """
        Инициализирует Veector Core.
        Args:
            db_dir: Путь к директории базы данных VeectorDB.
            initial_index_path: Опциональный путь к файлу индекса для VeectorDB.
            ... (остальные аргументы) ...
        """
        print(f"--- Initializing Veector Core v{CORE_VERSION} ---")
        print(f"    Requires: tensors v{TENSORS_VERSION_REQ}+, veectordb v{VEECTORDB_VERSION_REQ}+, operations v{OPERATIONS_VERSION_REQ}+")
        print(f"    IPFS: {ipfs_enabled}, Address: {ipfs_address}")
        self.db_dir = Path(db_dir).resolve()

        if not VeectorDB_defined:
             raise RuntimeError("Cannot initialize Veector: VeectorDB or Tensors failed to import.")

        try:
            # Передаем initial_index_path в VeectorDB
            self.db = VeectorDB(db_dir=self.db_dir, initial_index_path=initial_index_path)
            print("VeectorDB initialized by Veector Core.")
        except Exception as e:
             print(f"FATAL ERROR: Failed to initialize VeectorDB: {e}")
             traceback.print_exc()
             raise

        # IPFS Initialization (Без изменений)
        self.p2p_node = p2p_node
        self.ipfs_client = None
        if ipfs_enabled and IPFS_AVAILABLE and ipfs_address:
             try:
                 self.ipfs_client = ipfshttpclient.connect(addr=ipfs_address, timeout=10)
                 print(f"IPFS client connected to {ipfs_address}.")
                 self.ipfs_enabled = True
             except Exception as e:
                 print(f"Warn: Failed IPFS connect: {e}. IPFS disabled.")
                 self.ipfs_enabled = False
        else:
             self.ipfs_enabled = False

        # Cache Initialization (Без изменений)
        self.compute_cache: Dict[Tuple, Any] = {}
        self.knowledge_cache: Dict[str, Any] = {}
        self.cache_size = max(10, cache_size)
        self.eviction_strategy = eviction_strategy.upper() if eviction_strategy.upper() in ["LRU", "LFU"] else "LRU"
        self.cache_access_count: Dict[Union[Tuple, str], int] = {}
        self.cache_timestamps: Dict[Union[Tuple, str], float] = {}
        print(f"Cache initialized: Size={self.cache_size}, Strategy={self.eviction_strategy}")

        self.memory_module = Memory() if use_memory_module and 'Memory' in globals() else None

        # --- Define Operation Handlers ---
        self.core_ops: Dict[Tuple[int, ...], callable] = {}
        self._register_standard_ops()
        self._register_model_specific_ops()

        print(f"Initialized {len(self.core_ops)} total core operations.")
        self._log_memory("Veector Initialized")

    # --- Остальные методы класса Veector ---

    # --- Регистрация стандартных операций (без изменений) ---
    def _register_standard_ops(self):
        """Registers standard low-level operations."""


        standard_ops = {
            # Basic Math/Logic (existing definitions omitted for brevity)
            tuple(OP_SUM): np.sum, tuple(OP_SQRT): np.sqrt, tuple(OP_ABS): np.abs,
            tuple(OP_FLOOR): floor, tuple(OP_CEIL): ceil, tuple(OP_SIN): np.sin,
            tuple(OP_COS): np.cos, tuple(OP_TAN): np.tan,
            tuple(OP_COT): lambda d,**kw: 1/np.tan(d) if np.all(np.tan(d)!=0) else np.nan,
            tuple(OP_ASIN): arcsin, tuple(OP_ACOS): arccos, tuple(OP_ATAN): arctan,
            tuple(OP_NOT): np.logical_not, tuple(OP_IDENTITY): lambda d,**kw: d,
            tuple(OP_MEAN): mean, tuple(OP_STDDEV): std_dev, tuple(OP_MEDIAN): median,
            tuple(OP_INTERPOLATE): interpolate,
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
            tuple(OP_RAND_UNIFORM): lambda d,**kw: random_uniform(min_val=kw.get('min_val', 0.0), max_val=kw.get('max_val', 1.0)),
            tuple(OP_RAND_NORMAL): lambda d,**kw: random_normal(mu=kw.get('mu', 0.0), scale=kw.get('sigma', 1.0)),

            # Activations (existing definitions omitted for brevity)
            tuple(OP_RELU): relu, tuple(OP_SIGMOID): sigmoid, tuple(OP_SOFTMAX): softmax,
            tuple(OP_LEAKY_RELU): leaky_relu, tuple(OP_GELU): gelu, tuple(OP_SILU): silu,

            # Linear Algebra / NN Ops (existing definitions omitted for brevity)
            tuple(OP_MATRIX_MULTIPLY): lambda d,**kw: matrix_multiply(d, weights=kw.get('weights'), bias=kw.get('bias')),
            tuple(OP_CONVOLUTION): lambda d,**kw: convolution(d, kernel=kw.get('kernel'), bias=kw.get('bias')),
            tuple(OP_LAYER_NORM): lambda d,**kw: layer_normalization(d, norm_weight=kw.get('norm_weight'), norm_bias=kw.get('norm_bias'), eps=kw.get('eps', 1e-5)),
            tuple(OP_BATCH_NORM): lambda d,**kw: batch_norm(d, **kw),
            tuple(OP_DROPOUT): lambda d,**kw: dropout(d, rate=kw.get('rate', 0.1), is_training=kw.get('is_training', False)),
            tuple(OP_EMBEDDING_LOOKUP): lambda d,**kw: embedding_lookup(d, embedding_matrix=kw.get('embedding_matrix')),
            tuple(OP_RESHAPE_HEADS): lambda d,**kw: reshape_heads(d, num_heads=kw.get('num_heads')),
            tuple(OP_MERGE_HEADS): lambda d,**kw: merge_heads(d),
            tuple(OP_TRANSPOSE): transpose, tuple(OP_INVERSE): inverse, tuple(OP_TRACE): trace,
            tuple(OP_DETERMINANT): matrix_determinant, tuple(OP_EIGENVALUES): matrix_eigenvalues,
            tuple(OP_EXP_SMOOTHING): exponential_smoothing, tuple(OP_NORMALIZE_01): normalize,
            tuple(OP_APPLY_ROPE): self._execute_apply_rope,
            tuple(OP_GET_Q_ROT): get_q_rot, tuple(OP_GET_K_ROT): get_k_rot,
            tuple(OP_SCALED_DOT_PROD_ATTN): self._execute_sdpa,
            tuple(OP_UPDATE_KV_CACHE): lambda d,**kw: update_kv_cache(kw.get('cache'), kw.get('new_values'), kw.get('start_pos')),
            tuple(OP_CREATE_CAUSAL_MASK): lambda d,**kw: create_causal_mask(d, size=kw.get('size')),
            tuple(OP_ATTENTION_MULTIHEAD): lambda d,**kw: multi_head_attention(d, **kw),
            tuple(OP_REPEAT_KV_HEADS): lambda d,**kw: repeat_kv_heads(d, repeats=kw.get('repeats')),

            # Control Flow / Meta / Debug (existing definitions omitted for brevity)
            tuple(OP_IF): self._op_conditional_if,
            tuple(OP_LOOP_MULT): self._op_loop_multiply,
            tuple(OP_CHOICE): self._op_choice_select,
            tuple(OP_PRINT): self._op_output_print,
            tuple(OP_TRIGGER_REASON): self._op_trigger_reason,
            tuple(OP_DFS): self._op_graph_dfs,
            tuple(OP_MAKE_TUPLE): lambda d,**kw: make_tuple(d, **kw),
            # --- Tuple Access ---
            tuple(OP_GET_TUPLE_ELEM_0): self._op_get_tuple_elem(0),
            tuple(OP_GET_TUPLE_ELEM_1): self._op_get_tuple_elem(1),
            tuple(OP_GET_TUPLE_ELEM_2): self._op_get_tuple_elem(2),
            # --- ДОБАВЛЕНО: Регистрация обработчика для элемента 3 ---
            tuple(OP_GET_TUPLE_ELEM_3): self._op_get_tuple_elem(3),
            # --- КОНЕЦ ДОБАВЛЕНИЯ ---
            tuple(OP_STORE): self._op_store_context,
            tuple(OP_LOAD): self._op_load_context,
            tuple(OP_LOAD_INITIAL_INPUT): self._op_load_initial_input,
            tuple(OP_DEBUG_CONTEXT): self._op_debug_context,

            # Quantum Placeholders (existing definitions omitted for brevity)
            tuple(OP_QUANTUM_HADAMARD): lambda d,**kw: self._quantum_op_placeholder(d,"hadamard",**kw),
            tuple(OP_QUANTUM_PAULI_X): lambda d,**kw: self._quantum_op_placeholder(d,"pauli_x",**kw),
            tuple(OP_QUANTUM_CNOT): lambda d,**kw: self._quantum_op_placeholder(d,"cnot",**kw),
            tuple(OP_QUANTUM_MEASURE): lambda d,**kw: self._quantum_op_placeholder(d,"measure",**kw),
            tuple(OP_QUANTUM_SUPERPOS): lambda d,**kw: self._quantum_op_placeholder(d,"superposition",**kw),
            tuple(OP_QUANTUM_ENTANGLE): lambda d,**kw: self._quantum_op_placeholder(d,"entanglement",**kw),
        }
        self.core_ops.update(standard_ops)
        print(f"Registered {len(standard_ops)} standard operations.")
    # --- Регистрация специфичных для модели операций (без изменений) ---
    def _register_model_specific_ops(self):
        """Pytaetsja importirovat' i registrirovat' operacii iz veector_models."""
        registered_count = 0
        # --- Registracija Qwen2 Ops ---
        try:
            qwen2_ops_module = importlib.import_module("veector_models.qwen2.ops")
            qwen2_ops_dict = getattr(qwen2_ops_module, "qwen2_operations", None)
            if isinstance(qwen2_ops_dict, dict):
                valid_ops = {k: v for k, v in qwen2_ops_dict.items() if isinstance(k, tuple) and len(k) == 3}
                invalid_keys = [k for k in qwen2_ops_dict if k not in valid_ops]
                if invalid_keys:
                     print(f"  WARN: Invalid keys found in qwen2_operations: {invalid_keys}")
                if valid_ops:
                     self.core_ops.update(valid_ops)
                     print(f"  Successfully registered {len(valid_ops)} operations for Qwen2.")
                     registered_count += len(valid_ops)
                else:
                     print("  WARN: No valid operations found in qwen2_operations dictionary.")
            else:
                print("  WARN: Found veector_models.qwen2.ops but 'qwen2_operations' dictionary is missing or invalid.")
        except ImportError:
            print("  INFO: veector_models.qwen2.ops module not found, skipping Qwen2 ops registration.")
        except Exception as e:
            print(f"  ERROR during Qwen2 ops registration: {e}")
            traceback.print_exc()
        # --- End Qwen2 Ops ---

        if registered_count > 0:
            print(f"Registered {registered_count} model-specific operations.")

    # --- Хелперы для мета-операций (без изменений) ---
    def _op_get_tuple_elem(self, index: int):
        """Возвращает функцию для извлечения элемента кортежа по индексу."""
        def getter(current_data: Any, **kw) -> Any:
            step_context = kw.get('_step_context', {})
            provenance = kw.get('_provenance', {})
            if isinstance(current_data, tuple):
                if 0 <= index < len(current_data):
                    return current_data[index]
                else:
                    error_msg = f"GET_TUPLE_ELEM_{index} failed: index out of bounds (len: {len(current_data)})"
                    provenance["error"] = error_msg
                    print(f"ERROR: {error_msg}")
                    return None
            else:
                error_msg = f"GET_TUPLE_ELEM_{index} failed: current_data is not a tuple (type: {type(current_data)})"
                provenance["error"] = error_msg
                print(f"ERROR: {error_msg}")
                return None
        return getter

    def _op_store_context(self, current_data: Any, **kw) -> Any:
        """Сохраняет current_data в контекст шага."""
        step_context = kw.get('_step_context')
        meta_args = kw.get('_meta_args')
        provenance = kw.get('_provenance', {})
        if step_context is None or meta_args is None:
             provenance["error"] = "STORE failed: Missing internal context/args."
             print("ERROR: STORE failed: Missing internal context/args.")
             return current_data
        var_name = meta_args[0] if meta_args and isinstance(meta_args[0], str) else None
        if not var_name:
            provenance["error"] = "OP_STORE needs a string variable name as meta_arg[0]."
            print("ERROR: OP_STORE needs a string variable name as meta_arg[0].")
            return current_data
        step_context[var_name] = current_data
        print(f"  [CORE DBG] Stored '{var_name}' in step_context.")
        return current_data

    def _op_load_context(self, current_data: Any, **kw) -> Any:
        """Загружает значение из контекста шага в current_data."""
        step_context = kw.get('_step_context')
        meta_args = kw.get('_meta_args')
        provenance = kw.get('_provenance', {})
        if step_context is None or meta_args is None:
             provenance["error"] = "LOAD failed: Missing internal context/args."
             print("ERROR: LOAD failed: Missing internal context/args.")
             return None
        var_name = meta_args[0] if meta_args and isinstance(meta_args[0], str) else None
        if not var_name:
            provenance["error"] = "OP_LOAD needs a string variable name as meta_arg[0]."
            print("ERROR: OP_LOAD needs a string variable name as meta_arg[0].")
            return None
        if var_name in step_context:
            loaded_data = step_context[var_name]
            print(f"  [CORE DBG] Loaded '{var_name}' from step_context.")
            return loaded_data
        else:
            error_msg = f"LOAD failed: Var '{var_name}' not found in context. Keys: {list(step_context.keys())}"
            provenance["error"] = error_msg
            print(f"ERROR: {error_msg}")
            return None

    def _op_load_initial_input(self, current_data: Any, **kw) -> Any:
        """Загружает '_initial_input' из контекста шага."""
        step_context = kw.get('_step_context')
        provenance = kw.get('_provenance', {})
        if step_context is None:
             provenance["error"] = "LOAD_INITIAL_INPUT failed: Missing internal context."
             print("ERROR: LOAD_INITIAL_INPUT failed: Missing internal context.")
             return None
        initial_input = step_context.get('_initial_input')
        print("  [CORE DBG] Loaded '_initial_input' from step_context.")
        return initial_input

    def _op_debug_context(self, current_data: Any, **kw) -> Any:
        """Печатает ключи текущего контекста шага."""
        step_context = kw.get('_step_context')
        if step_context is not None:
            print(f"  [CORE DBG] DEBUG_CONTEXT: Keys={list(step_context.keys())}")
        else:
             print("  [CORE DBG] DEBUG_CONTEXT: step_context is None.")
        return current_data

    # --- Остальные хелперы и методы (без изменений) ---
    def _execute_apply_rope(self, current_data, **kw):
        if 'apply_rope' not in globals():
             print("ERROR: apply_rope function not found in operations module.")
             return None, None
        return apply_rope(
             q=kw.get('q'), k=kw.get('k'), position_ids=kw.get('position_ids'),
             rope_theta=kw.get('rope_theta', 10000.0),
             num_heads=kw.get('num_heads'), head_dim=kw.get('head_dim')
        )

    def _execute_sdpa(self, current_data, **kw):
        if 'scaled_dot_product_attention' not in globals():
             print("ERROR: scaled_dot_product_attention function not found in operations module.")
             return None
        return scaled_dot_product_attention(**kw)

    def _log_memory(self, stage: str):
        try:
            process = psutil.Process(os.getpid())
            mem_info = process.memory_info()
            vmem = psutil.virtual_memory()
            print(f"  [MEM_LOG] {stage}: RSS={mem_info.rss / (1024**2):.2f} MB, RAM Used={vmem.percent:.1f}%")
        except NameError: pass
        except Exception as e: print(f"  [MEM_LOG] Error getting memory usage: {e}")

    def _get_resource_status(self) -> Dict:
        mem_percent = 0; cpu_percent = 0; gpu_mem_percent = 0
        try:
            if 'psutil' in globals():
                mem = psutil.virtual_memory(); mem_percent = mem.percent
                cpu_percent = psutil.cpu_percent()
        except Exception: pass
        if TORCH_AVAILABLE and torch.cuda.is_available():
             try:
                 props = torch.cuda.get_device_properties(0)
                 allocated = torch.cuda.memory_allocated()
                 gpu_mem_percent = (allocated / props.total_memory) * 100 if props.total_memory > 0 else 0
             except Exception: pass
        return {"memory_percent": mem_percent, "cpu_percent": cpu_percent,
                "gpu_memory_percent": gpu_mem_percent, "battery_percent": 100 }

    # --- Tensor Creation and Validation Wrappers ---
    def create_tensor(self, *args, **kwargs) -> List:
        if 'create_tensor' not in globals():
            print("Error: create_tensor function not available."); return []
        return create_tensor(*args, **kwargs)

    def validate_tensor(self, tensor_structure: List) -> bool:
        if 'validate_tensor' not in globals():
            print("Error: validate_tensor function not available."); return False
        return validate_tensor(tensor_structure)

    # ИЗМЕНЕНО: Добавляем validate_tensor_tuple как метод-обертку
    def validate_tensor_tuple(self, meta_tuple: Tuple) -> bool:
        if 'validate_tensor_tuple' not in globals():
            print("Error: validate_tensor_tuple function not available."); return False
        return validate_tensor_tuple(meta_tuple)

    def get_tensor_hash(self, meta_tuple: Tuple) -> str:
        if 'get_tensor_hash' not in globals():
            print("Error: get_tensor_hash function not available."); return f"error_hash_{random.random()}"
        try:
            return get_tensor_hash(meta_tuple)
        except ValueError as e:
            print(f"Error hashing metadata tuple: {e}")
            return f"error_hash_{random.random()}"

    # --- Database Interaction Wrappers ---
    def save_tensor(self, tensor_structure: List) -> Optional[str]:
         """Сохраняет тензор (структуру списка) в БД."""
         tensor_id = None
         try:
             if not self.validate_tensor(tensor_structure):
                 print("Error (save_tensor): Attempted to save invalid tensor list structure.")
                 return None

             meta_dict = get_tensor_metadata(tensor_structure)
             meta_tuple_to_save = meta_dict.get("_encoded_metadata_v1_")

             # ИЗМЕНЕНО: Используем self.validate_tensor_tuple
             if not meta_tuple_to_save or not self.validate_tensor_tuple(meta_tuple_to_save):
                  print("Error (save_tensor): Cannot find or validate embedded metadata tuple for saving.")
                  return None

             data_payload = tensor_structure[5] if len(tensor_structure) == 6 else None
             tensor_id = self.db.insert_veector_tensor(meta_tuple_to_save, data_payload)
             return tensor_id
         except Exception as e:
             # ИЗМЕНЕНО: Печатаем traceback для NameError
             print(f"Error during save_tensor wrapper call (tensor_id={tensor_id}): {e}")
             traceback.print_exc()
             return None

    def load_tensor(self, doc_id: str, load_knowledge: bool = False, use_mmap: bool = True) -> Optional[List]:
         """Загружает тензор (структуру списка) из БД."""
         try:
            result_structure = self.db.get_veector_tensor(
                doc_id, load_knowledge_data=load_knowledge, use_mmap=use_mmap
            )
            if result_structure is not None and not self.validate_tensor(result_structure):
                print(f"Warning: load_tensor received invalid list structure from DB for {doc_id}.")
                return None
            return result_structure
         except Exception as e:
             print(f"Error during load_tensor wrapper call for {doc_id}: {e}")
             traceback.print_exc()
             return None

    # --- load_knowledge_tensor_data (Без изменений) ---
    def load_knowledge_tensor_data(self, knowledge_id: str) -> Optional[Any]:
        """Загружает и деквантует данные тензора знаний, использует кеш."""
        if knowledge_id in self.knowledge_cache:
            self._update_cache_access(knowledge_id)
            return self.knowledge_cache[knowledge_id]

        loaded_structure = self.load_tensor(knowledge_id, load_knowledge=True)
        if not loaded_structure:
            print(f"  ERROR: Failed to load structure for knowledge tensor {knowledge_id}.")
            return None

        try:
            meta_dict = get_tensor_metadata(loaded_structure)
            if len(loaded_structure) != 6:
                 print(f"  ERROR: Knowledge tensor {knowledge_id} loaded structure has no blob data (len={len(loaded_structure)}).")
                 return None
            loaded_data = loaded_structure[5]
        except Exception as e:
            print(f"  ERROR: Error extracting meta/data from loaded structure for {knowledge_id}: {e}")
            return None

        if loaded_data is None:
            print(f"  ERROR: Loaded blob data is None for {knowledge_id}.")
            return None

        final_data = loaded_data
        original_dtype_str = meta_dict.get("dtype", "")
        is_int8 = ('int8' in str(original_dtype_str).lower())
        should_cache = True

        if is_int8:
            scale = meta_dict.get("quantization_scale")
            if scale is None or not isinstance(scale, (float, int)) or scale <= 0:
                print(f"  ERROR: Missing/invalid 'quantization_scale' ({scale}) for INT8 tensor {knowledge_id}. Cannot dequantize!")
                return None

            try:
                scale_float16 = np.float16(scale)
                if loaded_data.dtype != np.int8:
                    print(f"  WARN: Data for {knowledge_id} marked INT8 but dtype is {loaded_data.dtype}. Attempting cast.")
                    loaded_data = loaded_data.astype(np.int8)
                dequantized_data = loaded_data.astype(np.float16) * scale_float16
                final_data = dequantized_data
                tags = meta_dict.get("tags", [])
                if TAG_COMP_EMBEDDING in tags or TAG_COMP_LM_HEAD in tags:
                    should_cache = False
            except Exception as dequant_e:
                print(f"  ERROR during dequantization for {knowledge_id}: {dequant_e}")
                traceback.print_exc()
                return None

        if final_data is not None and should_cache:
            self._add_to_cache(knowledge_id, final_data, is_knowledge=True)

        return final_data

    # --- Caching Methods (Без изменений) ---
    def _update_cache_access(self, key: Union[str, Tuple]):
        self.cache_timestamps[key] = time.time()
        self.cache_access_count[key] = self.cache_access_count.get(key, 0) + 1
    def _evict_cache(self):
        compute_keys=list(self.compute_cache.keys())
        knowledge_keys=list(self.knowledge_cache.keys())
        total_items = len(compute_keys) + len(knowledge_keys)
        if total_items <= self.cache_size: return
        num_to_evict = total_items - self.cache_size
        all_keys_metrics=[]
        try:
            sorter=lambda k: self.cache_access_count.get(k,0) if self.eviction_strategy=="LFU" else self.cache_timestamps.get(k,0)
            all_keys_metrics=[(k, sorter(k)) for k in compute_keys+knowledge_keys]
            all_keys_metrics.sort(key=lambda item: item[1])
            keys_to_evict=[item[0] for item in all_keys_metrics[:num_to_evict]]
            for key in keys_to_evict:
                evicted_item=False
                if key in self.compute_cache: del self.compute_cache[key]; evicted_item=True
                if key in self.knowledge_cache: del self.knowledge_cache[key]; evicted_item=True
                if evicted_item:
                     if key in self.cache_timestamps: del self.cache_timestamps[key]
                     if key in self.cache_access_count: del self.cache_access_count[key]
        except Exception as e: print(f"Cache eviction error: {e}")

    def _add_to_cache(self, key: Union[str, Tuple], value: Any, is_knowledge: bool = False):
         cache = self.knowledge_cache if is_knowledge else self.compute_cache
         cache[key]=value
         self._update_cache_access(key)
         self._evict_cache()

    def clear_cache(self, clear_knowledge: bool = True, clear_compute: bool = True):
         cleared_str = []
         if clear_compute: self.compute_cache.clear(); cleared_str.append("Compute")
         if clear_knowledge: self.knowledge_cache.clear(); cleared_str.append("Knowledge")
         if cleared_str:
             self.cache_timestamps.clear(); self.cache_access_count.clear()
             print(f"Cache cleared: {', '.join(cleared_str)}")


# --- Op Sequence Execution (v0.7.16 - Fixed Arg Resolution for Placeholder '_') ---
    def _execute_op_sequence(
        self,
        ops_sequence: List[Any],
        initial_data: Any,
        knowledge_params_for_ops: Dict[str, Any],
        **kw_context # Внешний контекст (например, position_ids)
    ) -> Tuple[Any, List[Dict], Dict]:
        """ Executes op sequence. v0.7.16 - Fixed Arg Resolution for Placeholder '_' """
        current_data = initial_data
        step_provenance_list = []
        step_context = {'_initial_input': initial_data, **kw_context}
        # _log_core_tensor_stats("Initial Data", current_data) # Optional log

        # print(f"DEBUG CORE: Starting execution of sequence with {len(ops_sequence)} steps.") # Keep if needed

        for i, op_command in enumerate(ops_sequence):
            step_provenance = {"step": i}
            step_start = time.time()
            op_tuple = None; op_code_list = None; op_func = None
            op_call_args_from_processor: Dict = {}
            meta_args = []
            valid_command = False

            # print(f"\nDEBUG CORE Step {i}: Processing op_command: {op_command}") # Keep if needed

            # --- Parsing command (no changes needed here) ---
            if not isinstance(op_command, list) or not op_command:
                error_msg = f"Command at step {i} is not a non-empty list: {op_command}"
                step_provenance["error"] = error_msg; step_provenance_list.append(step_provenance)
                return None, step_provenance_list, step_context

            op_code_source = op_command[0]
            args_source = op_command[1:]

            # print(f"DEBUG CORE Step {i}: op_code_source = {op_code_source}") # Keep if needed
            # print(f"DEBUG CORE Step {i}: args_source = {args_source}") # Keep if needed

            if isinstance(op_code_source, list) and len(op_code_source) == 3 and all(isinstance(x, int) for x in op_code_source):
                op_code_list = op_code_source; op_tuple = tuple(op_code_list)
                if len(args_source) == 1 and isinstance(args_source[0], dict):
                    op_call_args_from_processor = args_source[0]
                    valid_command = True
                    # print(f"DEBUG CORE Step {i}: Parsed args as dict: op_call_args_from_processor = {op_call_args_from_processor}") # Keep if needed
                elif not args_source:
                    valid_command = True
                    # print(f"DEBUG CORE Step {i}: Parsed as op without args.") # Keep if needed
                elif all(isinstance(arg, (str, int, float, bool, list, tuple)) for arg in args_source):
                     is_meta_op_expecting_args = op_tuple in (tuple(OP_STORE), tuple(OP_LOAD), tuple(OP_LOAD_INITIAL_INPUT), tuple(OP_DEBUG_CONTEXT))
                     if is_meta_op_expecting_args:
                         meta_args = list(args_source); valid_command = True
                         # print(f"DEBUG CORE Step {i}: Parsed as meta-op with meta_args: {meta_args}") # Keep if needed
                     else: error_msg = f"Invalid arguments format for non-meta op {op_tuple} at step {i}: {args_source}"; step_provenance["error"] = error_msg; valid_command = False
                else: error_msg = f"Cannot parse arguments format at step {i}: {args_source}"; step_provenance["error"] = error_msg; valid_command = False
            else: error_msg = f"Cannot parse OP CODE at step {i}: {op_code_source}"; step_provenance["error"] = error_msg; valid_command = False

            if not valid_command or op_code_list is None:
                if "error" not in step_provenance: step_provenance["error"] = f"Invalid command structure or OP CODE at step {i}: {op_command}"
                step_provenance_list.append(step_provenance)
                return None, step_provenance_list, step_context

            if op_tuple: step_provenance["op"] = op_tuple
            if meta_args: step_provenance["meta_args"] = meta_args
            if op_call_args_from_processor: step_provenance["op_args"] = op_call_args_from_processor
            # --- End Parsing ---

            # --- Execution ---
            op_result = None
            try:
                op_func = self.core_ops.get(op_tuple)
                if not op_func: raise ValueError(f"Operation {op_tuple} not found in core_ops.")

                # --- ИСПРАВЛЕННАЯ ЛОГИКА v2: Учитываем плейсхолдер "_" ---
                final_op_args = {} # Собираем финальные аргументы здесь
                for arg_name, value_source in op_call_args_from_processor.items():
                    if isinstance(value_source, str):
                        # СНАЧАЛА проверяем на плейсхолдер текущих данных
                        if value_source == '_':
                            final_op_args[arg_name] = current_data # Подставляем current_data
                            # print(f"DEBUG CORE Step {i}: Resolved '{arg_name}' as current_data placeholder") # Optional
                        # ПОТОМ проверяем, ссылка ли это на знания или контекст
                        elif value_source in knowledge_params_for_ops:
                            final_op_args[arg_name] = knowledge_params_for_ops[value_source]
                            # print(f"DEBUG CORE Step {i}: Resolved '{arg_name}' from knowledge") # Optional
                        elif value_source in step_context:
                            final_op_args[arg_name] = step_context[value_source]
                            # print(f"DEBUG CORE Step {i}: Resolved '{arg_name}' from context") # Optional
                        else:
                            # Если строка не плейсхолдер и не ссылка, считаем её литералом
                            final_op_args[arg_name] = value_source
                            # print(f"DEBUG CORE Step {i}: Kept literal string '{arg_name}'") # Optional
                    else:
                        # Если значение - не строка, используем как есть (литерал)
                        final_op_args[arg_name] = value_source
                        # print(f"DEBUG CORE Step {i}: Kept literal non-string '{arg_name}'") # Optional
                # --- КОНЕЦ ИСПРАВЛЕННОЙ ЛОГИКИ v2 ---

                # Собираем полный набор аргументов для вызова функции
                call_kwargs = {**final_op_args} # Начинаем с обработанных аргументов операции
                # Добавляем внутренние переменные контекста
                call_kwargs.update({'_step_context': step_context, '_meta_args': meta_args, '_provenance': step_provenance})

                # --- DEBUG PRINT: Print the final arguments passed to the op function ---
                # print(f"DEBUG CORE Step {i}: Calling op_func {op_tuple} with final call_kwargs keys: {list(call_kwargs.keys())}") # Keep if needed
                # Особенно важно проверить для MLP и ADD:
                # if op_tuple == tuple(OP_QWEN2_MLP):
                #     print(f"DEBUG CORE Step {i} (MLP CALL): hidden_act in call_kwargs = '{call_kwargs.get('hidden_act')}'")
                # if op_tuple == tuple(OP_ADD):
                #     print(f"DEBUG CORE Step {i} (ADD CALL): input_a type = {type(call_kwargs.get('input_a'))}, input_b type = {type(call_kwargs.get('input_b'))}")

                # Вызываем функцию операции
                op_result = op_func(current_data, **call_kwargs)

                # Обновляем current_data, если операция не мета-операция сохранения
                if op_tuple not in (tuple(OP_STORE), tuple(OP_DEBUG_CONTEXT)):
                    current_data = op_result
                    # _log_core_tensor_stats(f"Step {i} AFTER {op_tuple}", current_data) # Optional
                # else:
                    # _log_core_tensor_stats(f"Step {i} AFTER {op_tuple} (data unchanged)", current_data) # Optional

                # Проверяем, сообщила ли операция об ошибке через provenance
                if "error" in step_provenance and step_provenance["error"]:
                     print(f"--- Error reported by Op {op_tuple} at Step {i} ---"); print(f"    Error: {step_provenance['error']}"); print(f"--- End Error Report ---")
                     step_provenance["duration_ms"] = (time.time() - step_start) * 1000; step_provenance_list.append(step_provenance)
                     return None, step_provenance_list, step_context

            except Exception as e:
                error_msg = f"Op {op_tuple} execution failed at step {i}: {e}"
                step_provenance["error"] = str(e); print(f"--- Exception during Op {op_tuple} execution (Step {i}) ---"); traceback.print_exc(); print(f"--- End Exception ---")
                step_provenance["duration_ms"] = (time.time() - step_start) * 1000; step_provenance_list.append(step_provenance)
                return None, step_provenance_list, step_context
            # --- End Execution ---

            step_provenance["duration_ms"] = (time.time() - step_start) * 1000
            step_provenance_list.append(step_provenance)

            # Проверка статуса ошибки после выполнения шага (на всякий случай)
            if isinstance(current_data, dict) and current_data.get('status') == 'error':
                error_msg = f"Op {op_tuple} returned error status: {current_data.get('error')}"
                step_provenance_list[-1]["error"] = error_msg; print(f"ERROR: {error_msg}")
                return None, step_provenance_list, step_context
            # --- End Step Finish ---

        # print(f"DEBUG CORE: Finished execution of sequence.") # Keep if needed
        return current_data, step_provenance_list, step_context

    # --- Knowledge Selection (v0.7.9 - Без изменений) ---
    def _select_knowledge_tensors(self, processor_structure: List, context: Dict) -> Dict[str, str]:
        selected_knowledge_map = {}
        try:
            interface = get_tensor_interface(processor_structure) or {}
            processor_tags_list = get_tensor_tags(processor_structure)
            processor_coord = get_tensor_coord(processor_structure)
            if not processor_coord: raise ValueError("Processor coordinates not found in list structure")
        except Exception as e: print(f"Error selecting knowledge: Cannot access metadata from list structure: {e}"); return {}

        knowledge_needs = interface.get("knowledge_needed", [])
        if not knowledge_needs: return {}

        required_nest = context.get("required_nest")
        target_knowledge_group_from_context = context.get("target_knowledge_group")
        final_target_group = None
        if target_knowledge_group_from_context is not None:
            final_target_group = target_knowledge_group_from_context
        else:
            model_tag = None
            for tag in processor_tags_list:
                if TAG_MODEL_QWEN2 <= tag <= TAG_MODEL_DEEPSEEK + 7: model_tag = tag; break
            if model_tag == TAG_MODEL_QWEN2: final_target_group = GROUP_IDX_QWEN_KNOWLEDGE
            elif model_tag == TAG_MODEL_DEEPSEEK: final_target_group = GROUP_IDX_DEEPSEEK_KNOWLEDGE
            if final_target_group is None: print(f"  [SelectKnowledge] WARN: Could not infer target knowledge group from processor tags {processor_tags_list}. Searching without group filter.")

        target_nest = required_nest if isinstance(required_nest, int) else processor_coord.nest
        db_coord_filter = {};
        if final_target_group is not None: db_coord_filter["group"] = final_target_group
        db_coord_filter["nest"] = target_nest

        all_candidate_structures: Dict[str, List] = self.db.find_active_tensors(tensor_type="knowledge", coord_filter=db_coord_filter)

        for need in knowledge_needs:
            param_name = need.get("param_name"); base_tags_needed = need.get("tags", []); is_optional = need.get("optional", False)
            if not param_name or not base_tags_needed: print(f"  WARN: Invalid knowledge need definition: {need}"); continue

            best_candidate_id = None; query_tags_set = set(base_tags_needed); found_match_for_need = False
            for cand_id, cand_structure in all_candidate_structures.items():
                 try:
                     tensor_tags_list = get_tensor_tags(cand_structure)
                     if query_tags_set.issubset(set(tensor_tags_list)):
                         best_candidate_id = cand_id; found_match_for_need = True; break
                 except Exception as e: print(f"    Warn: Error processing candidate {cand_id} for need '{param_name}': {e}"); continue

            if found_match_for_need: selected_knowledge_map[param_name] = best_candidate_id
            elif not is_optional:
                err_msg = f"Missing REQUIRED knowledge for '{param_name}' with tags {query_tags_set} in G={final_target_group}, N={target_nest}. Candidates checked: {len(all_candidate_structures)}."
                print(f"ERROR: {err_msg}"); raise ValueError(err_msg)

        return selected_knowledge_map

    # --- Early Exit Check (Без изменений) ---
    def _check_early_exit(self, tensor_structure: List, result_data: Any, context: Dict) -> bool:
        try: exit_gates = get_tensor_exit_gates(tensor_structure) or []
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

    # --- compute Method (Без изменений) ---
    def compute(self, processor_id: str, context: Optional[Dict] = None) -> Dict:
        start_time = time.time(); context = context if context is not None else {}
        provenance = {"processor_id": processor_id, "steps": [], "timestamp_start": datetime.now().isoformat(), "context_received": {k: v for k, v in context.items() if k != 'input_data'}}
        final_step_context = {}

        input_data = context.get("input_data"); input_hash_str = "no_input"
        if input_data is not None:
             try:
                 if isinstance(input_data, np.ndarray): input_hash_str = hashlib.sha256(input_data.tobytes()).hexdigest()[:8]
                 else: input_hash_str = hashlib.sha256(pickle.dumps(input_data)).hexdigest()[:8]
             except Exception: input_hash_str = f"unhashable_{type(input_data).__name__}"
        state_id = context.get("state_id"); required_nest = context.get("required_nest", "default")
        cache_key = (processor_id, required_nest, state_id if state_id else input_hash_str)

        if cache_key in self.compute_cache:
            cached_data = self.compute_cache[cache_key]; self._update_cache_access(cache_key)
            provenance["status"] = "cached"; provenance["timestamp_end"] = datetime.now().isoformat(); provenance["duration_ms"] = (time.time() - start_time) * 1000
            return {"data": cached_data, "provenance": provenance, "status": "completed", "step_context": {}}

        processor_structure = self.load_tensor(processor_id, load_knowledge=False)
        if not processor_structure:
            error_msg = f"Processor {processor_id} structure list not found or invalid."; provenance["error"] = error_msg; provenance["status"] = "error"
            return {"data": None, "provenance": provenance, "status": "error", "step_context": {}}

        try:
            if not self.validate_tensor(processor_structure): raise ValueError("Invalid processor list structure.")
            tensor_type_str = get_tensor_type(processor_structure); status_str = get_tensor_status(processor_structure)
            coord_obj = get_tensor_coord(processor_structure); meta_dict = get_tensor_metadata(processor_structure)
            evo_version = meta_dict.get("evolutionary_version", 1)
            if tensor_type_str not in ["processor", "converter"]: raise ValueError(f"Tensor {processor_id} is type '{tensor_type_str}', not processor/converter.")
            if status_str == "archived": raise ValueError(f"Processor {processor_id} is archived.")
            if not coord_obj: raise ValueError(f"Cannot get coordinates for {processor_id}")
            provenance.update({"coord": str(coord_obj), "evo_version": evo_version})
        except Exception as e:
            error_msg = f"Error processing loaded structure for {processor_id}: {e}"; provenance["error"] = error_msg; provenance["status"] = "error"
            return {"data": None, "provenance": provenance, "status": "error", "step_context": {}}

        ops_sequences_dict = get_processor_ops_sequences(processor_structure) or {}
        precision_key = 'default'
        if isinstance(required_nest, int):
            if required_nest == 0: precision_key = TAG_PREC_INT8
            elif required_nest == 1: precision_key = TAG_PREC_FLOAT16
            elif required_nest == 2: precision_key = TAG_PREC_FLOAT32
            elif 20 <= required_nest <= 29: precision_key = required_nest
        elif required_nest != "default": precision_key = required_nest
        ops_sequence = ops_sequences_dict.get(precision_key)
        if ops_sequence is None and precision_key != 'default': ops_sequence = ops_sequences_dict.get('default')

        if not ops_sequence:
            result_data = context.get("input_data"); provenance["status"] = "completed (no ops)"; provenance["timestamp_end"] = datetime.now().isoformat(); provenance["duration_ms"] = (time.time() - start_time) * 1000
            self._add_to_cache(cache_key, result_data)
            return {"data": result_data, "provenance": provenance, "status": "completed", "step_context": {}}

        provenance["ops_sequence_length"] = len(ops_sequence); provenance["precision_key_used"] = str(precision_key)

        load_start_time = time.time(); loaded_knowledge_by_param_name = {}; knowledge_ids_loaded = []
        try:
            selected_knowledge_map = self._select_knowledge_tensors(processor_structure, context)
            for param_name, knowledge_id in selected_knowledge_map.items():
                 data = self.load_knowledge_tensor_data(knowledge_id)
                 if data is None: raise ValueError(f"Failed load knowledge blob {knowledge_id} (for param '{param_name}') which was selected.")
                 loaded_knowledge_by_param_name[param_name] = data; knowledge_ids_loaded.append(knowledge_id)
        except Exception as load_err:
            error_msg = f"Knowledge selection/loading failed: {load_err}"; provenance["error"] = error_msg; provenance["status"] = "error"
            return {"data": None, "provenance": provenance, "status": "error", "step_context": {}}

        provenance["knowledge_load_ms"] = (time.time() - load_start_time) * 1000; provenance["loaded_knowledge_ids"] = knowledge_ids_loaded

        current_step = 0; current_data = None; initial_step_context_from_state = {}; state_id_used = context.get("state_id")
        if state_id_used:
             intermediate_state_data = self.load_knowledge_tensor_data(state_id_used)
             if not isinstance(intermediate_state_data, dict):
                 error_msg = f"Invalid state data format for {state_id_used}. Expected dict."; provenance["error"] = error_msg; provenance["status"] = "error"
                 return {"data": None, "provenance": provenance, "status": "error", "step_context": {}}
             current_data = intermediate_state_data.get("data"); current_step = intermediate_state_data.get("next_step", 0); initial_step_context_from_state = intermediate_state_data.get("step_context", {})
             provenance["resumed_from_state"] = state_id_used; provenance["resumed_from_step"] = current_step
        else: current_data = context.get("input_data")

        filters = get_tensor_filters(processor_structure) or []
        if filters and current_data is not None: pass # Placeholder

        max_steps_per_call = context.get("max_steps", len(ops_sequence)); steps_executed_this_call = 0; exec_start_time = time.time(); ops_to_execute = []; result_data = None

        if 0 <= current_step < len(ops_sequence):
            end_step = min(current_step + max_steps_per_call, len(ops_sequence)); ops_to_execute = ops_sequence[current_step : end_step]

        if ops_to_execute:
            try:
                exec_context = {**initial_step_context_from_state, "_current_processor_id": processor_id, **context}
                result_data, step_provenance_list, final_step_context = self._execute_op_sequence(ops_to_execute, current_data, loaded_knowledge_by_param_name, **exec_context)
                provenance["steps"].extend(step_provenance_list); steps_executed_this_call = len(step_provenance_list)

                if step_provenance_list and "error" in step_provenance_list[-1] and step_provenance_list[-1]["error"]:
                     error_msg = f"Op sequence failed at step {current_step + steps_executed_this_call - 1}: {step_provenance_list[-1].get('error','Unknown error')}"
                     provenance["error"] = error_msg; provenance["status"] = "error";
                     return {"data": None, "provenance": provenance, "status": "error", "step_context": final_step_context}
                elif result_data is None and steps_executed_this_call > 0:
                     error_msg = f"Op sequence returned None data at step {current_step + steps_executed_this_call - 1} without explicit error."
                     provenance["error"] = error_msg; provenance["status"] = "error"
                     if "error" not in step_provenance_list[-1]: step_provenance_list[-1]["error"] = error_msg
                     return {"data": None, "provenance": provenance, "status": "error", "step_context": final_step_context}
                else: current_step += steps_executed_this_call
            except Exception as exec_e:
                error_msg = f"Error during op sequence execution: {exec_e}"; provenance["error"] = error_msg; provenance["status"] = "error"; traceback.print_exc()
                return {"data": None, "provenance": provenance, "status": "error", "step_context": final_step_context}
        else: result_data = current_data; final_step_context = initial_step_context_from_state

        provenance["execution_ms"] = (time.time() - exec_start_time) * 1000

        if current_step < len(ops_sequence):
            state_to_save = {"data": result_data, "next_step": current_step, "step_context": final_step_context}
            state_coord = TensorCoordinate(layer=STATE_TENSOR_LAYER, group=coord_obj.group, nest=coord_obj.nest, x=random.randint(10000,99999), y=coord_obj.y, z=coord_obj.z)
            try:
                state_tensor_structure = self.create_tensor(coord=state_coord, tensor_type="state", knowledge_data=state_to_save, status="temporary", parents=[processor_id])
                state_id_new = self.save_tensor(state_tensor_structure)
                if not state_id_new: raise RuntimeError("Failed to save state tensor.")
                provenance["status"] = "pending"; provenance["next_state_id"] = state_id_new; provenance["timestamp_end"] = datetime.now().isoformat(); provenance["duration_ms"] = (time.time() - start_time) * 1000
                return {"data": None, "provenance": provenance, "status": "pending", "state_id": state_id_new, "step_context": final_step_context}
            except Exception as state_e:
                error_msg = f"Failed create/save state: {state_e}"; provenance["error"] = error_msg; provenance["status"] = "error"
                return {"data": None, "provenance": provenance, "status": "error", "step_context": final_step_context}

        if self._check_early_exit(processor_structure, result_data, context):
            provenance["status"] = "early_exit"; provenance["timestamp_end"] = datetime.now().isoformat(); provenance["duration_ms"] = (time.time() - start_time) * 1000
            self._add_to_cache(cache_key, result_data);
            return {"data": result_data, "provenance": provenance, "status": "completed", "step_context": final_step_context}

        provenance["status"] = "completed"; provenance["timestamp_end"] = datetime.now().isoformat(); provenance["duration_ms"] = (time.time() - start_time) * 1000
        self._add_to_cache(cache_key, result_data);
        return {"data": result_data, "provenance": provenance, "status": "completed", "step_context": final_step_context}

    # --- Spawning / Evolution (Без изменений) ---
    def spawn_tensor(self, parent_id: str, strategy: str = "inherit", context: Optional[Dict] = None) -> Optional[str]:
        parent_structure = self.load_tensor(parent_id, load_knowledge=False)
        if not parent_structure: print(f"Error (spawn): Parent {parent_id} not found or invalid."); return None
        try:
            parent_meta_dict = get_tensor_metadata(parent_structure); parent_type_str = parent_meta_dict.get("tensor_type")
            parent_coord_obj = get_tensor_coord(parent_structure); parent_tags = parent_meta_dict.get("tags", [])
            parent_interface = parent_meta_dict.get("interface"); parent_ops_sequences = parent_meta_dict.get("ops_sequences")
            parent_filters = get_tensor_filters(parent_structure); parent_exit_gates = get_tensor_exit_gates(parent_structure)
            parent_dtype_str = parent_meta_dict.get("dtype"); parent_shape = parent_meta_dict.get("shape")
            if not parent_coord_obj: raise ValueError("Cannot get parent coordinates")
            if not parent_type_str: raise ValueError("Cannot get parent tensor type")
        except Exception as e: print(f"Error (spawn): Failed extraction from parent {parent_id}: {e}"); return None

        child_coord = self._generate_next_coords(parent_coord_obj); child_metadata_extra = {"spawn_strategy": strategy}
        child_type = parent_type_str; child_knowledge_data = None; child_tags = list(parent_tags)
        child_interface = parent_interface.copy() if parent_interface else None; child_ops_sequences = parent_ops_sequences.copy() if parent_ops_sequences else None
        child_filters = list(parent_filters) if parent_filters else None; child_exit_gates = list(parent_exit_gates) if parent_exit_gates else None
        child_dtype = parent_dtype_str; child_shape = parent_shape

        try:
            parent_has_blob = parent_meta_dict.get("has_blob_data", False)
            if parent_type_str == "knowledge":
                if strategy == "inherit" or not parent_has_blob: child_knowledge_data = None
                elif strategy == "mutate_knowledge":
                    parent_data = self.load_knowledge_tensor_data(parent_id)
                    if parent_data is not None and isinstance(parent_data, np.ndarray):
                        scale = context.get("mutation_scale", 0.05) if context else 0.05
                        noise = np.random.normal(0, scale, parent_data.shape).astype(parent_data.dtype); child_knowledge_data = parent_data + noise
                        child_dtype = child_knowledge_data.dtype; child_shape = child_knowledge_data.shape
                        child_tags = [t for t in child_tags if not (TAG_PREC_FLOAT32 <= t <= TAG_PREC_INT4)]
                        new_prec_tag = DTYPE_MAPPING.get(child_dtype, TAG_PREC_FLOAT16); child_tags.append(new_prec_tag)
                    else: raise ValueError("Cannot mutate missing/invalid knowledge data.")
                elif strategy == "distill_knowledge":
                    parent_data = self.load_knowledge_tensor_data(parent_id)
                    if parent_data is not None:
                        target_format_str = "int8"
                        if np.issubdtype(parent_data.dtype, np.floating):
                            abs_max = np.max(np.abs(parent_data)); scale = abs_max / 127.0 if abs_max > 1e-9 else 1.0
                            scale = max(scale, 1e-9); distilled_data = np.round(parent_data / scale).astype(np.int8)
                            child_knowledge_data = distilled_data; child_dtype = np.int8; child_shape = distilled_data.shape
                            new_prec_tag = TAG_PREC_INT8; child_tags = [t for t in child_tags if not (TAG_PREC_FLOAT32 <= t <= TAG_PREC_INT4)]
                            child_tags.append(new_prec_tag); child_metadata_extra["quantization_scale"] = float(scale)
                        else: raise ValueError(f"Cannot distill non-float data to {target_format_str}")
                    else: raise ValueError("Cannot distill missing knowledge data.")
        except Exception as e: print(f"Error applying spawn strategy '{strategy}': {e}"); traceback.print_exc(); return None

        try:
            child_tensor_structure = self.create_tensor(
                 coord=child_coord, tensor_type=child_type, tags=list(set(child_tags)), interface=child_interface,
                 ops_sequences=child_ops_sequences, filters=child_filters, exit_gates=child_exit_gates,
                 knowledge_data=child_knowledge_data, dtype=child_dtype, shape=child_shape, name_id=-1,
                 parents=[parent_id], evolutionary_version=1, status="active", metadata_extra=child_metadata_extra
            )
            child_id = self.save_tensor(child_tensor_structure)
            if child_id: print(f"Spawn successful: Child {child_id} created from parent {parent_id} at {child_coord}")
            else: print(f"Error (spawn): Failed to save child tensor.")
            return child_id
        except Exception as e: print(f"Error during child tensor creation/saving: {e}"); traceback.print_exc(); return None

    # --- Вспомогательные и специальные методы (без изменений) ---
    def _distill_knowledge_data(self, data: Any, target_format: str) -> Any:
        if isinstance(data, np.ndarray):
            if target_format == "int8" and np.issubdtype(data.dtype, np.floating):
                scale = np.max(np.abs(data))/127.0 if np.max(np.abs(data))>1e-9 else 1.0
                return np.round(data/max(scale, 1e-9)).astype(np.int8)
            elif target_format == "float16": return data.astype(np.float16)
            elif target_format == "bfloat16": print("Warn: Numpy has no bfloat16, returning float16."); return data.astype(np.float16)
        return data

    def _generate_next_coords(self, c: TensorCoordinate) -> TensorCoordinate:
        if not isinstance(c, TensorCoordinate): raise TypeError("Input must be a TensorCoordinate object.")
        return TensorCoordinate(layer=c.layer, group=c.group, nest=c.nest, x=c.x, y=c.y, z=c.z + 1)

    def _op_conditional_if(self, data: Any, **kw) -> Any:
        condition = bool(data[0]) if isinstance(data,(list,np.ndarray)) and len(data)>0 else bool(data)
        true_val = kw.get('true_branch'); false_val = kw.get('false_branch')
        return true_val if condition else false_val

    def _op_loop_multiply(self, data: Any, **kw) -> Any:
        value_to_repeat = data; n_repeats = kw.get('n', 1)
        try: n = int(n_repeats); n = max(n, 0)
        except (ValueError, TypeError): n = 1
        if isinstance(value_to_repeat, (int, float, complex, np.number)): return value_to_repeat * n
        elif isinstance(value_to_repeat, np.ndarray): return np.tile(value_to_repeat, n)
        elif isinstance(value_to_repeat, str): return value_to_repeat * n
        elif isinstance(value_to_repeat, list): return value_to_repeat * n
        else: return [value_to_repeat] * n

    def _op_choice_select(self, data: Any, **kw) -> Any:
        index = kw.get('index', 0); options = kw.get('options', [])
        if not isinstance(options, list): print("Error (CHOICE): 'options' must be a list."); return None
        try:
            idx = int(index)
            if 0 <= idx < len(options): return options[idx]
            else: print(f"Error (CHOICE): Index {idx} out of bounds for options list (len {len(options)})."); return None
        except (ValueError, TypeError): print(f"Error (CHOICE): Invalid index type ({type(index)})."); return None

    def _op_trigger_reason(self, data: Any, **kw) -> Any: return data
    def _op_graph_dfs(self, data: Any, **kw) -> Any: print("Warning: Graph DFS operation not implemented."); return []
    def _op_output_print(self, data: Any, **kw):
        label = kw.get("label", "PRINT"); print(f"--- {label} (Step Data) ---"); print(data); print(f"--- End {label} ---")
        return data
    def _quantum_op_placeholder(self, data: Any, op_type: str, **kw) -> Any:
        print(f"WARN: Quantum op '{op_type}' called, but Qiskit not fully integrated."); return data
    def _get_tensor_manager(self) -> Optional[Any]: print("Warning: _get_tensor_manager called, not implemented."); return None
    def handle_task( self, task_description: Any, input_data: Any, resources: Dict, priority: float ) -> Optional[Dict]:
        print("Warning: handle_task called, but TensorManager is not implemented."); return None

    # --- Публичные Методы для Информации (без изменений) ---
    def get_available_processors(self, filter_dict=None) -> List[str]:
        try:
            active_processors_dict = self.db.find_active_tensors(tensor_type="processor", coord_filter=filter_dict)
            return list(active_processors_dict.keys())
        except Exception as e: print(f"Error finding available processors: {e}"); return []

    def get_tensor_structure(self, tensor_id: str) -> Optional[List]:
        return self.load_tensor(tensor_id, load_knowledge=False)

# --- Example Usage Placeholder (Без изменений) ---
if __name__ == "__main__":
    print("\\n--- Veector Core Example ---")
    try:
        print("--- Example needs update/implementation ---")
    except Exception as ex:
        print(f"Error in example: {ex}")
    print("\\n--- Example Finished ---")
