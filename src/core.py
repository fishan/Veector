# FILE: core.py
# Description: Core execution engine for the Veector system.
# Author: [Your Name/Project Name]
# Date: 2025-03-27 (Based on discussion v12 - STRICT FINAL ATTEMPT v2)
# Version: 0.3.12 (TypeError Fix, Full Ops, User Formatting)

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
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union # Ensure typing is imported

# --- Optional Imports ---
try:
    import torch # Used for GPU memory check and potential future GPU ops
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # print("Warning: PyTorch not found. GPU features disabled.") # User formatting preference

try:
    import ipfshttpclient
    IPFS_AVAILABLE = True
except ImportError:
    IPFS_AVAILABLE = False
    # print("Warning: ipfshttpclient not found. IPFS features disabled.") # User formatting preference

try:
    # For Quantum Ops (Optional)
    from qiskit import QuantumCircuit
    # from qiskit_aer import AerSimulator # Newer import? Check Qiskit version
    from qiskit.providers.aer import Aer # Older import style
    from qiskit import execute # Older execute style
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False
    # print("Warning: Qiskit or Qiskit Aer not found. Quantum operations disabled/placeholder.") # User formatting preference


# --- Veector Project Imports ---
# These imports must succeed for the core to function.
try:
    from veectordb import VeectorDB
    from tensors import (
        TensorCoordinate, create_tensor, validate_tensor, get_tensor_coord,
        get_tensor_op_channels, get_tensor_default_op, get_tensor_filters,
        get_tensor_exit_gates, get_tensor_metadata, get_tensor_parents,
        get_tensor_status, get_tensor_hash, get_tensor_type,
        get_processor_ops_sequence, get_processor_required_knowledge_tags,
        get_processor_param_mapping, get_knowledge_compatibility_tags,
        get_tensor_input_channels, get_tensor_output_channels # Ensure these are imported
    )
    # Import operations - Functions need adaptation for **kw or specific args
    from operations import * # Import all for now
    from memory import Memory # Optional memory module
except ImportError as e:
    print(f"FATAL ERROR: Could not import core Veector components: {e}.")
    # Define minimal dummies to allow basic script execution without crashing
    # --- Corrected Dummy Classes Syntax ---
    class VeectorDB:
         def __init__(self, *args, **kwargs):
              print("Dummy VeectorDB")
              self.data={}
         def find_active_tensors(self, *args, **kwargs): return {} # Dummy method
         def _load_blob(self, *args, **kwargs): return None # Dummy method
         def insert_veector_tensor(self, *args, **kwargs): return "dummy_id" # Dummy method
         def get_veector_tensor(self, *args, **kwargs): return None # Dummy method
         def archive_tensor(self, *args, **kwargs): pass # Dummy method
         def update_tensor_metadata(self, *args, **kwargs): pass # Dummy method

    class TensorCoordinate:
         def __init__(self, *args, **kwargs):
             pass
         def to_string(self): return "dummy_coord"
         @classmethod
         def from_string(cls, s): return cls()
    # --- End Corrected Dummy Classes ---

    def validate_tensor(t): return isinstance(t, list) and len(t)>4
    def get_tensor_hash(t): return "dummy_hash"
    def get_tensor_type(t): return t[4].get('tensor_type') if validate_tensor(t) else None
    def get_tensor_metadata(t): return t[4] if validate_tensor(t) else {}
    def get_tensor_coord(t): return t[0] if validate_tensor(t) else None
    def get_processor_required_knowledge_tags(t): return []
    def get_processor_ops_sequence(t): return None
    def get_tensor_default_op(t): return []
    def get_processor_param_mapping(t): return {}
    def get_tensor_filters(t): return []
    def get_tensor_exit_gates(t): return []
    def get_tensor_op_channels(t): return [[], [], []]
    def get_tensor_parents(t): return []
    def get_tensor_input_channels(t): return [] # Added dummy
    def get_tensor_output_channels(t): return [] # Added dummy
    def create_tensor(*args, **kwargs): return []
    # --- Corrected Dummy Operations Syntax (Each Def/Return on New Line) ---
    def relu(d):
        return d
    def softmax(d):
        return d
    def mod(d1, d2):
        return d1
    def floor(d):
        return d
    def ceil(d):
        return d
    def arcsin(d):
        return d
    def arccos(d):
        return d
    def arctan(d):
        return d
    def xor(d1, d2):
        return d1
    def nand(d1, d2):
        return d1
    def nor(d1, d2):
        return d1
    def random_uniform(d1, d2):
        return 0.5
    def random_normal(d1, d2):
        return 0
    def median(d):
        return d
    def mean(d):
        return d
    def std_dev(d):
        return d
    def dropout(d, rate=0):
        return d
    def batch_norm(d):
        return d
    def sigmoid(d):
        return d
    def leaky_relu(d, alpha=0):
        return d
    def gelu(d):
        return d
    def exponential_smoothing(d, alpha=0):
        return d
    def normalize(d):
        return d
    def interpolate(d, nl):
        return d
    def layer_normalization(d):
        return d
    def matrix_determinant(d):
        return 1
    def matrix_eigenvalues(d):
        return d
    def transpose(d):
        return d
    def inverse(d):
        return d
    def trace(d):
        return d
    def multi_head_attention(d, **kw): # Keep **kw for flexibility with optional args
        return d
    def matrix_multiply(d, **kw): # Keep **kw for flexibility with optional args
        return d
    def convolution(d, **kw): # Keep **kw for flexibility with optional args
        return d
    class Memory:
        pass
    # --- End Corrected Dummy Operations ---


# --- Constants ---
DEFAULT_CACHE_SIZE = 1000
DEFAULT_EVICTION_STRATEGY = "LRU"
DEFAULT_IPFS_ADDRESS = '/ip4/127.0.0.1/tcp/5001'
STATE_TENSOR_LAYER = -2

# --- Operation Code Constants (Formatted One Per Line) ---
OP_SUM = [0, 0, 0]
OP_SUBTRACT = [0, 0, 1]
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
OP_GELU = [40, 5, 0]
OP_EXP_SMOOTHING = [19, 0, 0]
OP_NORMALIZE_01 = [20, 0, 0]
OP_INTERPOLATE = [20, 1, 0]
OP_LAYER_NORM = [40, 1, 0]
OP_BATCH_NORM = [40, 4, 0]
OP_DROPOUT = [40, 3, 0]
OP_MATRIX_MULTIPLY = [30, 0, 0]
OP_DETERMINANT = [30, 1, 0]
OP_EIGENVALUES = [30, 2, 0]
OP_CONVOLUTION = [30, 3, 0]
OP_TRANSPOSE = [30, 4, 0]
OP_INVERSE = [30, 5, 0]
OP_TRACE = [30, 6, 0]
OP_ATTENTION_MULTIHEAD = [40, 2, 0]
OP_QUANTUM_HADAMARD = [50, 0, 0]
OP_QUANTUM_PAULI_X = [50, 0, 1]
OP_QUANTUM_CNOT = [50, 1, 0]
OP_QUANTUM_MEASURE = [50, 2, 0]
OP_QUANTUM_SUPERPOS = [50, 3, 0]
OP_QUANTUM_ENTANGLE = [50, 4, 0]


class Veector:
    """
    The core execution engine for the Veector system.
    Orchestrates tensor computation based on dynamic configurations.
    """
    def __init__(self,
                 db_dir: Union[str, Path] = "../data/db",
                 cache_size: int = DEFAULT_CACHE_SIZE,
                 eviction_strategy: str = DEFAULT_EVICTION_STRATEGY,
                 use_memory_module: bool = False,
                 p2p_node: Optional[Any] = None,
                 ipfs_enabled: bool = True,
                 ipfs_address: str = DEFAULT_IPFS_ADDRESS):

        print("Initializing Veector Core...")
        self.db_dir = Path(db_dir)
        try:
            self.db = VeectorDB(db_dir=self.db_dir)
            print("VeectorDB initialized.")
        except NameError:
             print("FATAL ERROR: VeectorDB class not found. Check imports.")
             raise
        except Exception as e:
             print(f"FATAL ERROR: Failed to initialize VeectorDB: {e}")
             raise

        self.p2p_node = p2p_node
        self.ipfs_client = None
        if ipfs_enabled and IPFS_AVAILABLE and ipfs_address:
            try:
                self.ipfs_client = ipfshttpclient.connect(addr=ipfs_address, timeout=10)
                print(f"IPFS client connected to {ipfs_address}.")
                ipfs_enabled = True
            except Exception as e:
                print(f"Warning: Failed to connect to IPFS at {ipfs_address}: {e}. IPFS disabled.")
                ipfs_enabled = False
        else:
             if ipfs_enabled and not IPFS_AVAILABLE: print("Warning: IPFS requested but ipfshttpclient not installed. IPFS disabled.")
             ipfs_enabled = False
        self.ipfs_enabled = ipfs_enabled

        # Caching
        self.compute_cache: Dict[Tuple, Any] = {}
        self.knowledge_cache: Dict[str, Any] = {}
        self.cache_size = max(10, cache_size)
        self.eviction_strategy = eviction_strategy.upper() if eviction_strategy.upper() in ["LRU", "LFU"] else "LRU"
        self.cache_access_count: Dict[Union[Tuple, str], int] = {}
        self.cache_timestamps: Dict[Union[Tuple, str], float] = {}
        print(f"Cache initialized: Size={self.cache_size}, Strategy={self.eviction_strategy}")

        # Optional Modules
        self.memory_module = Memory() if use_memory_module and 'Memory' in globals() else None
        # self.evolution_engine = Evolution(self) # Initialize later if needed

        # --- Core Operations Dictionary (ADDED FULL DICTIONARY - Lambdas Corrected) ---
        # Assumes functions in operations.py accept data + named args from mapping
        self.core_ops: Dict[Tuple[int, ...], callable] = {
            # Basic Arithmetic
            tuple(OP_SUM): lambda data, **kw: np.sum(data).astype(data.dtype) if isinstance(data, np.ndarray) else data,
            tuple(OP_SUBTRACT): lambda data, **kw: data[0] - data[1] if isinstance(data, (list, np.ndarray)) and len(data) > 1 else data,
            tuple(OP_MULTIPLY): lambda data, **kw: data[0] * data[1] if isinstance(data, (list, np.ndarray)) and len(data) > 1 else data,
            tuple(OP_DIVIDE): lambda data, **kw: data[0] / data[1] if isinstance(data, (list, np.ndarray)) and len(data) > 1 and np.all(data[1]!=0) else data,
            tuple(OP_SQRT): lambda data, **kw: np.sqrt(data),
            tuple(OP_POWER): lambda data, **kw: np.power(data[0], data[1]) if isinstance(data, (list, np.ndarray)) and len(data) > 1 else data,
            tuple(OP_ABS): lambda data, **kw: np.abs(data),
            tuple(OP_MOD): lambda data, **kw: mod(data[0], data[1]) if isinstance(data, (list, np.ndarray)) and len(data) > 1 else data,
            tuple(OP_FLOOR): lambda data, **kw: floor(data),
            tuple(OP_CEIL): lambda data, **kw: ceil(data),
            # Trigonometry
            tuple(OP_SIN): lambda data, **kw: np.sin(data), tuple(OP_COS): lambda data, **kw: np.cos(data), tuple(OP_TAN): lambda data, **kw: np.tan(data),
            tuple(OP_COT): lambda data, **kw: 1 / np.tan(data) if np.all(np.tan(data)!=0) else np.full_like(data, np.nan),
            tuple(OP_ASIN): lambda data, **kw: arcsin(data), tuple(OP_ACOS): lambda data, **kw: arccos(data), tuple(OP_ATAN): lambda data, **kw: arctan(data),
            # Logic
            tuple(OP_GREATER): lambda data, **kw: data[0] > data[1] if isinstance(data, (list, np.ndarray)) and len(data)>1 else False,
            tuple(OP_EQUAL): lambda data, **kw: data[0] == data[1] if isinstance(data, (list, np.ndarray)) and len(data)>1 else False,
            tuple(OP_AND): lambda data, **kw: bool(data[0]) and bool(data[1]) if isinstance(data, (list, np.ndarray)) and len(data)>1 else False,
            tuple(OP_OR): lambda data, **kw: bool(data[0]) or bool(data[1]) if isinstance(data, (list, np.ndarray)) and len(data)>1 else False,
            tuple(OP_NOT): lambda data, **kw: not bool(data),
            tuple(OP_XOR): lambda data, **kw: xor(data[0], data[1]) if isinstance(data, (list, np.ndarray)) and len(data)>1 else data,
            tuple(OP_NAND): lambda data, **kw: nand(data[0], data[1]) if isinstance(data, (list, np.ndarray)) and len(data)>1 else data,
            tuple(OP_NOR): lambda data, **kw: nor(data[0], data[1]) if isinstance(data, (list, np.ndarray)) and len(data)>1 else data,
            # Control Flow
            tuple(OP_IF): self._op_conditional_if, tuple(OP_LOOP_MULT): self._op_loop_multiply, tuple(OP_CHOICE): self._op_choice_select,
            # Random
            tuple(OP_RAND_UNIFORM): lambda data, **kw: random_uniform(data[0], data[1]) if isinstance(data, (list, np.ndarray)) and len(data)>1 else np.random.rand(),
            tuple(OP_RAND_NORMAL): lambda data, **kw: random_normal(data[0], data[1]) if isinstance(data, (list, np.ndarray)) and len(data)>1 else np.random.randn(),
            tuple(OP_MEDIAN): lambda data, **kw: median(data),
            # Output
            tuple(OP_PRINT): self._op_output_print,
            # Identity
            tuple(OP_IDENTITY): lambda data, **kw: data,
            # Meta
            tuple(OP_TRIGGER_REASON): self._op_trigger_reason,
            # Graph
            tuple(OP_DFS): self._op_graph_dfs,
            # Statistics
            tuple(OP_MEAN): lambda data, **kw: mean(data), tuple(OP_STDDEV): lambda data, **kw: std_dev(data),
            # Activations
            tuple(OP_RELU): lambda data, **kw: relu(data), tuple(OP_SIGMOID): lambda data, **kw: sigmoid(data), tuple(OP_SOFTMAX): lambda data, **kw: softmax(data),
            tuple(OP_LEAKY_RELU): lambda data, **kw: leaky_relu(data, alpha=kw.get('alpha', 0.01)), tuple(OP_GELU): lambda data, **kw: gelu(data),
            # Smoothing
            tuple(OP_EXP_SMOOTHING): lambda data, **kw: exponential_smoothing(data, alpha=kw.get('alpha', 0.5)),
            # Normalization / NN Utils
            tuple(OP_NORMALIZE_01): lambda data, **kw: normalize(data),
            tuple(OP_INTERPOLATE): lambda data, **kw: interpolate(data, new_length=kw['new_length']) if 'new_length' in kw else print("Error: interpolate needs 'new_length'"),
            tuple(OP_LAYER_NORM): lambda data, **kw: layer_normalization(data),
            tuple(OP_BATCH_NORM): lambda data, **kw: batch_norm(data),
            tuple(OP_DROPOUT): lambda data, **kw: dropout(data, rate=kw.get('rate', 0.1)),
            # Matrix/Tensor Ops --- CORRECTED LAMBDAS ---
            tuple(OP_MATRIX_MULTIPLY): lambda data, **kw: matrix_multiply(data, kw['weights']) if 'weights' in kw else print("Error: 'weights' kwarg missing for matmul"),
            tuple(OP_DETERMINANT): lambda data, **kw: matrix_determinant(data),
            tuple(OP_EIGENVALUES): lambda data, **kw: matrix_eigenvalues(data),
            tuple(OP_CONVOLUTION): lambda data, **kw: convolution(data, kw['kernel']) if 'kernel' in kw else print("Error: 'kernel' kwarg missing for conv"),
            tuple(OP_TRANSPOSE): lambda data, **kw: transpose(data),
            tuple(OP_INVERSE): lambda data, **kw: inverse(data),
            tuple(OP_TRACE): lambda data, **kw: trace(data),
            # Attention --- CORRECTED LAMBDA ---
            tuple(OP_ATTENTION_MULTIHEAD): lambda data, **kw: multi_head_attention(data, kw['key'], kw['value'], **{k:v for k,v in kw.items() if k not in ['key','value']}) if 'key' in kw and 'value' in kw else print("Error: 'key'/'value' missing for attention"),
            # Quantum (Placeholders)
            tuple(OP_QUANTUM_HADAMARD): lambda data, **kw: self._quantum_op_placeholder(data, "hadamard", **kw),
            tuple(OP_QUANTUM_PAULI_X): lambda data, **kw: self._quantum_op_placeholder(data, "pauli_x", **kw),
            tuple(OP_QUANTUM_CNOT): lambda data, **kw: self._quantum_op_placeholder(data, "cnot", **kw),
            tuple(OP_QUANTUM_MEASURE): lambda data, **kw: self._quantum_op_placeholder(data, "measure", **kw),
            tuple(OP_QUANTUM_SUPERPOS): lambda data, **kw: self._quantum_op_placeholder(data, "superposition", **kw),
            tuple(OP_QUANTUM_ENTANGLE): lambda data, **kw: self._quantum_op_placeholder(data, "entanglement", **kw),
        }
        print(f"Initialized {len(self.core_ops)} core operations.")
        self._log_memory("Veector Initialized")

    # --- Logging & Monitoring Methods ---
    # [Unchanged]
    def _log_memory(self, stage: str):
        try: process = psutil.Process(os.getpid()); ram_usage = process.memory_info().rss / 1024**2; print(f"Mem({stage}): RAM {ram_usage:.1f}MB", end='');
        except Exception as e: print(f"Mem log warning: {e}"); return
        if TORCH_AVAILABLE and torch.cuda.is_available():
             try: allocated = torch.cuda.memory_allocated()/1024**2; reserved = torch.cuda.memory_reserved()/1024**2; print(f" | GPU Alloc {allocated:.1f}MB, Reserv {reserved:.1f}MB")
             except Exception as e: print(f" | GPU Mem log warning: {e}")
        else: print()
    def _get_resource_status(self) -> Dict:
        mem_percent=0; cpu_percent=0; gpu_mem_percent=0
        try: mem=psutil.virtual_memory(); mem_percent=mem.percent; cpu_percent=psutil.cpu_percent()
        except Exception: pass
        if TORCH_AVAILABLE and torch.cuda.is_available():
             try: props=torch.cuda.get_device_properties(0); allocated=torch.cuda.memory_allocated(); gpu_mem_percent = (allocated / props.total_memory) * 100 if props.total_memory > 0 else 0
             except Exception: pass
        return {"memory_percent": mem_percent, "cpu_percent": cpu_percent, "gpu_memory_percent": gpu_mem_percent, "battery_percent": 100 }

    # --- Tensor Creation and Validation Wrappers ---
    # [Unchanged]
    def create_tensor(self, *args, **kwargs) -> List: return create_tensor(*args, **kwargs) if 'create_tensor' in globals() else []
    def validate_tensor(self, tensor: List) -> bool: return validate_tensor(tensor) if 'validate_tensor' in globals() else False
    def get_tensor_hash(self, tensor_structure: List) -> str: return get_tensor_hash(tensor_structure) if 'get_tensor_hash' in globals() else f"error_hash_{random.random()}"

    # --- Database Interaction Wrappers ---
    # [Unchanged]
    def save_tensor(self, tensor_structure: List) -> Optional[str]:
         try:
             if not self.validate_tensor(tensor_structure): print("Error: Save invalid tensor."); return None
             return self.db.insert_veector_tensor(tensor_structure)
         except Exception as e: print(f"Error saving tensor: {e}"); return None
    def load_tensor(self, doc_id: str, load_knowledge: bool = False) -> Optional[List]:
         try: return self.db.get_veector_tensor(doc_id, load_knowledge_data=load_knowledge)
         except Exception as e: print(f"Error loading tensor {doc_id}: {e}"); return None
    def load_knowledge_tensor_data(self, knowledge_id: str) -> Optional[Any]:
         if knowledge_id in self.knowledge_cache: self._update_cache_access(knowledge_id); return self.knowledge_cache[knowledge_id]
         structure = self.db.get_veector_tensor(knowledge_id, load_knowledge_data=False)
         if structure and get_tensor_type(structure) == "knowledge":
             metadata = get_tensor_metadata(structure); blob_cid = metadata.get("knowledge_blob_cid")
             if blob_cid:
                 data = self.db._load_blob(blob_cid)
                 if data is not None: self._add_to_cache(knowledge_id, data, is_knowledge=True); return data
                 else: print(f"Warn: Failed load blob {blob_cid} for {knowledge_id}")
             else: print(f"Warn: Knowledge {knowledge_id} no blob CID.")
         else: print(f"Error: Knowledge {knowledge_id} not found/invalid type.")
         return None

    # --- Caching Methods ---
    # [Unchanged]
    def _update_cache_access(self, key: Union[str, Tuple]): self.cache_timestamps[key] = time.time(); self.cache_access_count[key] = self.cache_access_count.get(key, 0) + 1
    def _evict_cache(self):
        compute_keys=list(self.compute_cache.keys()); knowledge_keys=list(self.knowledge_cache.keys()); total_items=len(compute_keys)+len(knowledge_keys)
        if total_items < self.cache_size: return
        num_to_evict=total_items-self.cache_size+1; all_keys_metrics=[]
        try:
            sorter=lambda k: self.cache_access_count.get(k,0) if self.eviction_strategy=="LFU" else self.cache_timestamps.get(k,0); all_keys_metrics=[(k, sorter(k)) for k in compute_keys+knowledge_keys]; all_keys_metrics.sort(key=lambda item: item[1]); keys_to_evict=[item[0] for item in all_keys_metrics[:num_to_evict]]; evicted_count=0
            for key in keys_to_evict:
                evicted_item=False
                if key in self.compute_cache: del self.compute_cache[key]; evicted_item=True
                if key in self.knowledge_cache: del self.knowledge_cache[key]; evicted_item=True
                if evicted_item:
                     if key in self.cache_timestamps: del self.cache_timestamps[key]
                     if key in self.cache_access_count: del self.cache_access_count[key]
                     evicted_count+=1
        except Exception as e: print(f"Cache eviction error: {e}")
    def _add_to_cache(self, key: Union[str, Tuple], value: Any, is_knowledge: bool = False): self._evict_cache(); cache = self.knowledge_cache if is_knowledge else self.compute_cache; cache[key]=value; self._update_cache_access(key)
    def clear_cache(self, clear_knowledge: bool = True, clear_compute: bool = True):
         if clear_compute: self.compute_cache.clear()
         if clear_knowledge: self.knowledge_cache.clear()
         self.cache_timestamps.clear(); self.cache_access_count.clear(); print("Caches cleared.")

    # --- Core Computation Logic ---

    def _select_knowledge_tensors(self, required_tags: List[Union[int, Tuple]], context: Dict) -> Dict[Union[int, Tuple], str]:
        """Finds IDs of suitable knowledge tensors based on numerical tags and context."""
        # [Implementation unchanged]
        print(f"Selecting knowledge for tags: {required_tags} (context: {context})")
        selected_ids = {}; required_nest = context.get("required_nest"); all_candidates = self.db.find_active_tensors(tensor_type="knowledge"); found_tags = set()
        for tag in required_tags:
            best_candidate_id = None; best_candidate_nest = -999; lookup_tag = tuple(tag) if isinstance(tag, list) else tag
            for cand_id, structure in all_candidates.items():
                 meta = get_tensor_metadata(structure); compat_tags = meta.get("compatibility_tags", [])
                 if lookup_tag in compat_tags:
                     coord = get_tensor_coord(structure);
                     if not coord: continue
                     current_nest = coord.nest
                     if required_nest is not None:
                          if current_nest == required_nest: best_candidate_id = cand_id; best_candidate_nest = current_nest; break
                          elif current_nest <= required_nest and current_nest > best_candidate_nest: best_candidate_nest = current_nest; best_candidate_id = cand_id
                     elif current_nest > best_candidate_nest: best_candidate_nest = current_nest; best_candidate_id = cand_id
            if best_candidate_id: selected_ids[lookup_tag] = best_candidate_id; found_tags.add(lookup_tag); print(f"  Selected tag {lookup_tag}: {best_candidate_id} (nest={best_candidate_nest})")
            else: print(f"Warn: No knowledge for tag {lookup_tag} matching context.")
        if not set(required_tags).issubset(found_tags): print(f"Warn: Missing knowledge for tags: {set(required_tags)-found_tags}")
        return selected_ids

    def _execute_op_sequence(self,
                             ops_sequence: List[List[int]],
                             initial_data: Any,
                             loaded_knowledge_by_id: Dict[str, Any], # {knowledge_id: data}
                             param_mapping: Dict[Any, str], # {tag_id: param_name}
                             **kw_context # Pass context map
                             ) -> Tuple[Any, List[Dict]]:
        """Executes a sequence of operations using loaded knowledge tensors."""
        # [Mapping logic using context map]
        current_data = initial_data; step_provenance_list = []
        selected_knowledge_ids_map_by_tag = kw_context.get("_selected_knowledge_ids_map", {})

        knowledge_params_for_ops = {}
        for tag_id, param_name in param_mapping.items():
             knowledge_id = selected_knowledge_ids_map_by_tag.get(tag_id)
             if knowledge_id and knowledge_id in loaded_knowledge_by_id:
                  knowledge_params_for_ops[param_name] = loaded_knowledge_by_id[knowledge_id]
             else:
                  print(f"Warn exec: Knowledge for tag {tag_id} (map->{param_name}) not loaded.")

        for i, op_code_list in enumerate(ops_sequence):
            op_tuple = tuple(op_code_list); op_func = self.core_ops.get(op_tuple); step_provenance = {"step": i, "op": op_tuple}; step_start = time.time()
            if not op_func:
                error_msg = f"Op {op_tuple} not found"; print(f"Error: {error_msg}")
                step_provenance["error"] = error_msg; step_provenance_list.append(step_provenance)
                return None, step_provenance_list
            try:
                # Call operation with data and mapped knowledge
                current_data = op_func(current_data, **knowledge_params_for_ops) # Pass mapped knowledge as keywords
                step_provenance["duration_ms"] = (time.time() - step_start) * 1000
                if current_data is None:
                     print(f"Warn: Op {op_tuple} returned None.")
            except TypeError as te: # Catch TypeError (likely missing/wrong arguments)
                 error_msg = f"Op {op_tuple} failed (TypeError): {te}"
                 print(f"Error executing {op_tuple}: {te}")
                 if "missing" in str(te) and "argument" in str(te): error_msg += f" - Check func signature and param_mapping."
                 elif "unexpected keyword argument" in str(te): error_msg += f" - Check function signature vs **kw / param_mapping."
                 step_provenance["error"] = error_msg; step_provenance_list.append(step_provenance)
                 return None, step_provenance_list # Abort on TypeError
            except Exception as e: # Catch other exceptions
                error_msg = f"Op {op_tuple} failed: {e}"; print(f"Error executing {op_tuple}: {e}")
                step_provenance["error"] = str(e); step_provenance_list.append(step_provenance)
                return None, step_provenance_list # Abort
            step_provenance_list.append(step_provenance)
        return current_data, step_provenance_list

    def _check_early_exit(self, tensor_structure: List, result_data: Any, context: Dict) -> bool:
        """Checks early exit conditions (placeholder)."""
        # [Implementation unchanged]
        exit_gates = get_tensor_exit_gates(tensor_structure);
        if not exit_gates: return False
        check_context = {**context, "current_result": result_data}
        for gate_ref in exit_gates:
            try:
                gate_triggered = False
                if isinstance(gate_ref, str): gate_result = self.compute(gate_ref, context={"input_data": check_context}); gate_triggered = gate_result.get("status") == "completed" and isinstance(gate_result.get("data"), bool) and gate_result["data"]
                elif callable(gate_ref): gate_triggered = gate_ref(check_context)
                else: print(f"Warning: Invalid exit gate format: {gate_ref}")
                if gate_triggered: print(f"Early exit condition met by gate: {gate_ref}"); return True
            except Exception as e: print(f"Error checking exit gate {gate_ref}: {e}"); continue
        return False

    def compute(self, processor_id: str, context: Optional[Dict] = None) -> Dict:
        """Core computation method. Dynamically loads knowledge and executes operations."""
        # [Corrected Error Handling Logic]
        start_time = time.time(); context = context or {}
        provenance = {"processor_id": processor_id, "steps": [],"timestamp_start": datetime.now().isoformat(),"context_received": {k:v for k,v in context.items() if k!='input_data'}}
        self._log_memory(f"Compute Start: {processor_id}")

        # Cache Key & Check
        input_data = context.get("input_data"); input_hash = hashlib.sha256(pickle.dumps(input_data)).hexdigest()[:8] if input_data is not None else "no_input"; state_id = context.get("state_id", None); required_nest = context.get("required_nest", "default"); cache_key = (processor_id, required_nest, state_id if state_id else input_hash)
        if cache_key in self.compute_cache: cached_data = self.compute_cache[cache_key]; self._update_cache_access(cache_key); provenance["status"] = "cached"; provenance["timestamp_end"] = datetime.now().isoformat(); provenance["duration_ms"] = (time.time() - start_time) * 1000; print(f"Compute Cache Hit for {processor_id}"); return {"data": cached_data, "provenance": provenance, "status": "completed"}

        # Load Processor
        processor_structure = self.load_tensor(processor_id, load_knowledge=False)
        if not processor_structure or get_tensor_type(processor_structure) != "processor": error_msg = f"Processor {processor_id} not found/invalid"; provenance["error"] = error_msg; provenance["status"] = "error"; return {"data": None, "provenance": provenance, "status": "error"}
        coord = get_tensor_coord(processor_structure); metadata = get_tensor_metadata(processor_structure); status = metadata.get("status", "active"); provenance.update({"coord": str(coord), "evo_version": metadata.get("evolutionary_version")})
        if status == "archived": error_msg = f"Processor {processor_id} archived"; provenance["error"] = error_msg; provenance["status"] = "error"; return {"data": None, "provenance": provenance, "status": "error"}

        # Determine Config
        required_tags = get_processor_required_knowledge_tags(processor_structure); ops_sequence = get_processor_ops_sequence(processor_structure)
        if not ops_sequence: default_op = get_tensor_default_op(processor_structure); ops_sequence = [default_op] if default_op and default_op != OP_IDENTITY else []
        param_mapping = get_processor_param_mapping(processor_structure) # { tag_id: param_name }
        if not ops_sequence: result_data = context.get("input_data"); provenance["status"] = "completed (no ops)"; provenance["timestamp_end"] = datetime.now().isoformat(); provenance["duration_ms"] = (time.time() - start_time) * 1000; self._add_to_cache(cache_key, result_data); return {"data": result_data, "provenance": provenance, "status": "completed"}
        provenance["ops_sequence_length"] = len(ops_sequence)

        # Select & Load Knowledge
        load_start_time = time.time()
        selected_knowledge_ids_map_by_tag = self._select_knowledge_tensors(required_tags, context) # { tag_id: knowledge_id }
        loaded_knowledge_by_id = {} # { knowledge_id: data }
        missing_tags = set(required_tags) - set(selected_knowledge_ids_map_by_tag.keys())
        if missing_tags: error_msg = f"Could not find knowledge for tags: {missing_tags}"; provenance["error"] = error_msg; provenance["status"] = "error"; return {"data": None, "provenance": provenance, "status": "error"}
        for tag, knowledge_id in selected_knowledge_ids_map_by_tag.items():
             data = self.load_knowledge_tensor_data(knowledge_id)
             if data is not None: loaded_knowledge_by_id[knowledge_id] = data
             else: error_msg = f"Failed load knowledge {knowledge_id} for tag {tag}"; provenance["error"] = error_msg; provenance["status"] = "error"; return {"data": None, "provenance": provenance, "status": "error"}
        provenance["knowledge_load_ms"] = (time.time() - load_start_time) * 1000
        provenance["loaded_knowledge_ids"] = list(loaded_knowledge_by_id.keys())
        self._log_memory(f"Compute Knowledge Loaded: {processor_id}")

        # Initial Data & State
        current_step = 0; current_data = None; state_id_used = context.get("state_id")
        if state_id_used:
             intermediate_state_data = self.load_knowledge_tensor_data(state_id_used)
             if intermediate_state_data is None or not isinstance(intermediate_state_data, dict): error_msg = f"Failed/invalid state {state_id_used}"; provenance["error"] = error_msg; provenance["status"] = "error"; return {"data": None, "provenance": provenance, "status": "error"}
             current_data = intermediate_state_data.get("data"); current_step = intermediate_state_data.get("next_step", 0); provenance["resumed_from_state"] = state_id_used; provenance["resumed_from_step"] = current_step; print(f"Resuming computation from step {current_step}")
        else: current_data = context.get("input_data")

        # Pre-processing Filters
        filters = get_tensor_filters(processor_structure)
        if filters and current_data is not None: print("Applying pre-processing filters (placeholder)..."); pass # Apply filters

        # Execute Op Sequence
        max_steps_per_call = context.get("max_steps", len(ops_sequence)); steps_executed_this_call = 0
        exec_start_time = time.time(); ops_to_execute = ops_sequence[current_step : current_step + max_steps_per_call]
        result_data = None; step_provenance_list = []
        if ops_to_execute:
            try:
                kw_for_exec = {"_selected_knowledge_ids_map": selected_knowledge_ids_map_by_tag}
                result_data, step_provenance_list = self._execute_op_sequence(
                    ops_to_execute, current_data, loaded_knowledge_by_id, param_mapping, **kw_for_exec
                )
                provenance["steps"].extend(step_provenance_list); steps_executed_this_call = len(step_provenance_list); current_step += steps_executed_this_call

                # Check for failure WITHIN the sequence execution helper
                if result_data is None and step_provenance_list and "error" in step_provenance_list[-1]:
                    # If helper returned None AND set an error in provenance
                    error_msg = f"Op sequence failed at step {current_step - 1}: {step_provenance_list[-1]['error']}"
                    provenance["error"] = error_msg # Use the specific error
                    provenance["status"] = "error"
                    return {"data": None, "provenance": provenance, "status": "error"} # Return immediately

            except Exception as exec_e: # Catch errors during the call to helper itself
                 error_msg = f"Error during op sequence execution call: {exec_e}"; provenance["error"] = error_msg; provenance["status"] = "error"; print(error_msg)
                 return {"data": None, "provenance": provenance, "status": "error"}
        else:
            result_data = current_data # No ops executed
        provenance["execution_ms"] = (time.time() - exec_start_time) * 1000

        # Handle Sequential State
        if current_step < len(ops_sequence):
            state_to_save = {"data": result_data, "next_step": current_step}; state_coord = TensorCoordinate(layer=STATE_TENSOR_LAYER, group=coord.group, nest=coord.nest, x=random.randint(10000,99999))
            state_tensor = self.create_tensor(coord=state_coord, tensor_type="knowledge", knowledge_data=state_to_save, status="temporary", metadata={"origin_processor": processor_id, "origin_coord": str(coord)}); state_id_new = self.save_tensor(state_tensor)
            if not state_id_new: error_msg = "Failed save state"; provenance["error"] = error_msg; provenance["status"] = "error"; return {"data": None, "provenance": provenance, "status": "error"}
            provenance["status"] = "pending"; provenance["next_state_id"] = state_id_new; provenance["timestamp_end"] = datetime.now().isoformat(); provenance["duration_ms"] = (time.time() - start_time) * 1000; print(f"Computation pending, state {state_id_new} for step {current_step}"); return {"data": None, "provenance": provenance, "status": "pending", "state_id": state_id_new}

        # Post-processing Filters & Early Exit
        # TODO: Apply post-filters
        if self._check_early_exit(processor_structure, result_data, context):
            provenance["status"] = "early_exit"; provenance["timestamp_end"] = datetime.now().isoformat(); provenance["duration_ms"] = (time.time() - start_time) * 1000; print(f"Early exit for {processor_id}"); self._add_to_cache(cache_key, result_data); return {"data": result_data, "provenance": provenance, "status": "completed"}

        # Finalize
        provenance["status"] = "completed"; provenance["timestamp_end"] = datetime.now().isoformat(); provenance["duration_ms"] = (time.time() - start_time) * 1000
        self._add_to_cache(cache_key, result_data); self._log_memory(f"Compute End: {processor_id}")

        # Route Output
        output_channels = get_tensor_output_channels(processor_structure)
        if output_channels: print(f"Routing result (completed) to channels: {output_channels} (placeholder)..."); # TODO: Implement routing

        return {"data": result_data, "provenance": provenance, "status": "completed"}

    # --- Spawning / Evolution ---
    # [Unchanged]
    def spawn_tensor(self, parent_id: str, strategy: str = "inherit", context: Optional[Dict] = None) -> Optional[str]:
        """Creates a new child tensor (processor or knowledge) based on a parent."""
        print(f"Spawning child from parent {parent_id} using strategy '{strategy}'...")
        parent_structure = self.load_tensor(parent_id, load_knowledge=False);
        if not parent_structure: print(f"Error spawn: Parent {parent_id} not found."); return None
        parent_metadata = get_tensor_metadata(parent_structure); parent_type = parent_metadata.get("tensor_type"); parent_coord = get_tensor_coord(parent_structure)
        child_coord = self._generate_next_coords(parent_coord); child_metadata = {"spawn_strategy": strategy}; child_type = parent_type; child_knowledge_data = None
        child_ops_sequence = get_processor_ops_sequence(parent_structure); child_required_tags = get_processor_required_knowledge_tags(parent_structure); child_param_mapping = get_processor_param_mapping(parent_structure); child_default_op = get_tensor_default_op(parent_structure); child_filters = get_tensor_filters(parent_structure); child_input_channels = get_tensor_input_channels(parent_structure); child_output_channels = get_tensor_output_channels(parent_structure); child_exit_gates = get_tensor_exit_gates(parent_structure); child_compatibility_tags = get_knowledge_compatibility_tags(parent_structure) if parent_type == "knowledge" else None
        try:
            if parent_type == "knowledge":
                child_type = "knowledge"
                if strategy == "inherit": child_metadata["knowledge_blob_cid"] = parent_metadata.get("knowledge_blob_cid")
                elif strategy == "mutate_knowledge":
                    parent_data = self.load_knowledge_tensor_data(parent_id)
                    if parent_data is not None and isinstance(parent_data, np.ndarray): scale = context.get("mutation_scale", 0.05); noise = np.random.normal(0, scale, parent_data.shape); child_knowledge_data = parent_data + noise.astype(parent_data.dtype); child_metadata["mutation_scale"] = scale; child_metadata["knowledge_blob_cid"] = None
                    else: raise ValueError("Cannot mutate non-NumPy/missing knowledge.")
                elif strategy == "distill_knowledge":
                    parent_data = self.load_knowledge_tensor_data(parent_id)
                    if parent_data is not None:
                        target_format = context.get("target_format", "int8"); distilled_data = self._distill_knowledge_data(parent_data, target_format)
                        if distilled_data is not None: child_knowledge_data = distilled_data; child_metadata["distilled_from"] = parent_id; child_metadata["distill_format"] = target_format; child_compatibility_tags = [t for t in (child_compatibility_tags or []) if "float" not in t] + [target_format]
                        else: raise ValueError("Distillation failed.")
                    else: raise ValueError("Cannot distill missing knowledge.")
                else: print(f"Warn: Strategy '{strategy}' default inherit for knowledge."); child_metadata["knowledge_blob_cid"] = parent_metadata.get("knowledge_blob_cid")
            elif parent_type == "processor":
                child_type = "processor"
                if strategy == "inherit": pass
                elif strategy == "mutate_ops_sequence":
                     if child_ops_sequence:
                          idx = random.randrange(len(child_ops_sequence)); current_op = tuple(child_ops_sequence[idx]); compatible_ops = [op for op in self.core_ops.keys() if op != current_op]
                          if compatible_ops: new_op = random.choice(compatible_ops); child_ops_sequence[idx] = list(new_op); child_metadata["mutation_details"] = f"Replaced op @{idx} with {new_op}"; print(f"  Mutated ops sequence @{idx}")
                          else: print("Warn: No alternative ops.")
                     else: print("Warn: No ops sequence to mutate.")
                elif strategy == "specialize_processor":
                     focus_tags = context.get("focus_tags"); add_filter_id = context.get("add_filter_id")
                     if isinstance(focus_tags, list): child_required_tags = focus_tags; child_metadata["specialization_tags"] = focus_tags; print(f"  Specialized tags: {focus_tags}")
                     if add_filter_id: child_filters = (child_filters or []) + [add_filter_id]; child_metadata["added_filter"] = add_filter_id; print(f"  Specialized filter: {add_filter_id}")
                else: print(f"Warn: Strategy '{strategy}' default inherit for processor.")
            else: raise ValueError(f"Unknown parent type '{parent_type}'")
        except Exception as e: print(f"Error applying spawn strategy '{strategy}': {e}"); return None
        final_child_metadata = {"desc": f"Child of {parent_id} via {strategy}", **child_metadata}
        child_tensor_structure = self.create_tensor( coord=child_coord, tensor_type=child_type, metadata=final_child_metadata, op=child_default_op, ops_sequence=child_ops_sequence, required_knowledge_tags=child_required_tags, param_mapping=child_param_mapping, filters=child_filters, input_channels=child_input_channels, output_channels=child_output_channels, exit_gates=child_exit_gates, knowledge_data=child_knowledge_data, compatibility_tags=child_compatibility_tags, parents=[parent_id], evolutionary_version=1 )
        child_id = self.save_tensor(child_tensor_structure)
        if child_id: print(f"Spawn successful: Created child {child_id} (type: {child_type}) from {parent_id}")
        else: print(f"Error spawn: Failed to save child tensor.")
        return child_id

    def _distill_knowledge_data(self, data: Any, target_format: str) -> Any:
        """Placeholder for knowledge distillation."""
        # [Implementation unchanged]
        print(f"Distilling data to {target_format} (Placeholder)...")
        if isinstance(data, np.ndarray):
             if target_format == "int8" and np.isrealobj(data): scale = np.max(np.abs(data))/127.0 if np.max(np.abs(data))>1e-9 else 1.0; quantized = np.round(data/(scale + 1e-9)).astype(np.int8); print("  Quantized to int8"); return quantized
             elif target_format == "float16": print("  Converting to float16"); return data.astype(np.float16)
        print("  Distillation not applied."); return data

    def _generate_next_coords(self, current_coords: TensorCoordinate) -> TensorCoordinate:
        """Generates next coordinates (simple placeholder)."""
        # [Implementation unchanged]
        return TensorCoordinate(layer=current_coords.layer, group=current_coords.group, nest=current_coords.nest, x=current_coords.x, y=current_coords.y, z=current_coords.z + 1)

    # --- Definitions for Special Operation Methods (ADDED and Formatted) ---
    def _op_conditional_if(self, data: Any, **kw) -> Any:
        """Placeholder: Executes different branches based on condition."""
        print(f"Executing Op: Conditional IF")
        try:
            condition = bool(data[0]) if isinstance(data, (list, np.ndarray)) and len(data)>0 else bool(data)
            true_ref = data[1] if isinstance(data, (list, np.ndarray)) and len(data)>1 else None
            false_ref = data[2] if isinstance(data, (list, np.ndarray)) and len(data)>2 else None
            ref_to_process = true_ref if condition else false_ref
            print(f"  Condition={condition}. Branch ref: {ref_to_process}")
            # TODO: Implement actual execution of the branch
            return ref_to_process # Returning the reference for now
        except Exception as e: print(f"  Error in _op_conditional_if: {e}"); return None

    def _op_loop_multiply(self, data: Any, **kw) -> Any:
        """Placeholder: Repeats an operation or multiplies data."""
        print("Executing Op: Loop Multiply")
        try:
             val = data[0] if isinstance(data, (list, np.ndarray)) and len(data)>0 else data
             n = int(data[1]) if isinstance(data, (list, np.ndarray)) and len(data)>1 else 1
             print(f"  Multiplying {type(val)} by {n}")
             if isinstance(val, (int, float, complex, np.number)): return val * n
             elif isinstance(val, np.ndarray): return val * n
             else: return [val] * n
        except Exception as e: print(f"  Error in _op_loop_multiply: {e}"); return data

    def _op_choice_select(self, data: Any, **kw) -> Any:
        """Placeholder: Selects one option based on an index."""
        print("Executing Op: Choice Select")
        try:
            idx = int(data[0]) if isinstance(data, (list, np.ndarray)) and len(data)>0 else 0
            options = data[1:] if isinstance(data, (list, np.ndarray)) else []
            if 0 <= idx < len(options):
                print(f"  Selected option at index {idx}")
                return options[idx]
            else: print(f"  Index {idx} out of bounds."); return None
        except Exception as e: print(f"  Error in _op_choice_select: {e}"); return None

    def _op_trigger_reason(self, data: Any, **kw) -> Any:
        """Placeholder: Triggers optimization/evolution analysis."""
        processor_id = kw.get("_current_processor_id") # Need compute to pass this
        print(f"Executing Op: Trigger Reason/Optimization for processor {processor_id}?")
        # TODO: Call self.optimize_tensor_structure(processor_id) or similar
        return data

    def _op_graph_dfs(self, data: Any, **kw) -> Any:
        """Placeholder: Performs Depth First Search on a graph."""
        print("Executing Op: Graph DFS")
        try:
            graph_ref = data[0] if isinstance(data, (list, np.ndarray)) else None
            start_node = data[1] if isinstance(data, (list, np.ndarray)) and len(data)>1 else None
            graph = graph_ref if isinstance(graph_ref, dict) else {} # TODO: Load if ID
            path = []; stack = [start_node] if start_node is not None else []; visited = set()
            while stack:
                 node = stack.pop()
                 if node not in visited and node is not None:
                      node_str = str(node); visited.add(node_str); path.append(node)
                      neighbors = graph.get(node_str,[]); stack.extend(reversed(neighbors))
            print(f"  DFS Path from {start_node}: {path}")
            return path
        except Exception as e: print(f"  Error in _op_graph_dfs: {e}"); return []

    def _op_output_print(self, data: Any, **kw):
        """Operation with side effect: print data nicely."""
        print(f"Executing Op: Print Output")
        try:
            if isinstance(data, np.ndarray):
                with np.printoptions(precision=3, suppress=True, threshold=20, edgeitems=5): print(f"  >> Veector Output: {data}")
            elif isinstance(data, dict) and "provenance" in data: print(f"  >> Veector Output (Result): {data.get('data')}")
            else: print(f"  >> Veector Output: {data}")
        except Exception as e: print(f"  >> Veector Output (Error formatting): {e}")
        return data

    # --- Placeholder for Quantum Ops ---
    # [Unchanged]
    def _quantum_op_placeholder(self, data: Any, op_type: str, **kw) -> Any:
         if not QISKIT_AVAILABLE: print(f"Quantum op '{op_type}' skipped: Qiskit not available."); return data
         print(f"Quantum op '{op_type}' called (Placeholder).")
         return data

    # --- Placeholders for TensorManager Interaction ---
    # [Unchanged]
    def _get_tensor_manager(self) -> Optional[Any]: print("TensorManager interaction not implemented."); return None
    def handle_task(self, task_description: Any, input_data: Any, resources: Dict, priority: float) -> Optional[Dict]: print("handle_task needs TensorManager logic."); return None
    def get_available_processors(self, filter_dict=None) -> List[str]: return list(self.db.find_active_tensors(tensor_type="processor", coord_filter=filter_dict).keys())
    def get_tensor_structure(self, tensor_id: str) -> Optional[List]: return self.load_tensor(tensor_id, load_knowledge=False)


# --- Example Usage ---
if __name__ == "__main__":
    # [Example block unchanged from user's provided code - uses STRING tags]
    print("\n--- Veector Core Example (v2 - User Provided Base Example Style) ---")
    script_dir = Path(__file__).parent
    db_path = script_dir / f"../data/veector_core_db_v{time.strftime('%Y%m%d%H%M%S')}" # Use unique timestamp

    if db_path.exists():
        import shutil
        try: shutil.rmtree(db_path)
        except OSError as e: print(f"Error removing directory {db_path}: {e}")

    # Using STRING tags/params as in the user's example for now
    TAG_WEIGHTS_STR = "weights"; TAG_LINEAR_STR = "linear"; TAG_FLOAT32_STR = "float32"; TAG_INT8_STR = "int8"
    PARAM_NAME_WEIGHTS_STR = "weights"

    try:
        vec = Veector(db_dir=db_path, ipfs_enabled=False, p2p_node=None)
    except Exception as init_e:
        print(f"\nFATAL: Veector initialization failed: {init_e}")
        exit(1) # Exit if core cannot initialize

    # --- Create Knowledge Tensors ---
    coord_w0 = TensorCoordinate(layer=0, group=0, nest=0, x=1)
    weights0 = np.round(np.random.rand(5, 3) * 10).astype(np.int8)
    k_tensor0 = vec.create_tensor( coord=coord_w0, tensor_type="knowledge", knowledge_data=weights0, compatibility_tags=[TAG_WEIGHTS_STR, TAG_LINEAR_STR, TAG_INT8_STR], metadata={"desc": "INT8 Weights (N0)"} )
    k_id0 = vec.save_tensor(k_tensor0)
    print(f"\nSaved Knowledge (N0, int8): {k_id0}")

    coord_w1 = TensorCoordinate(layer=0, group=0, nest=1, x=1)
    weights1 = (np.random.rand(5, 3).astype(np.float32) - 0.5) * 0.1
    k_tensor1 = vec.create_tensor( coord=coord_w1, tensor_type="knowledge", knowledge_data=weights1, compatibility_tags=[TAG_WEIGHTS_STR, TAG_LINEAR_STR, TAG_FLOAT32_STR], metadata={"desc": "FP32 Weights (N1)"}, parents=[k_id0] )
    k_id1 = vec.save_tensor(k_tensor1)
    print(f"Saved Knowledge (N1, f32): {k_id1}")

    # --- Create Processor Tensor ---
    coord_p = TensorCoordinate(layer=1, group=0, nest=0, x=1)
    proc_tensor = vec.create_tensor(
        coord=coord_p, tensor_type="processor",
        ops_sequence=[OP_MATRIX_MULTIPLY, OP_PRINT], # Use constant
        required_knowledge_tags=[TAG_WEIGHTS_STR, TAG_LINEAR_STR], # Using strings
        param_mapping={TAG_WEIGHTS_STR: PARAM_NAME_WEIGHTS_STR}, # Map TAG string -> NAME string
        metadata={"desc": "Linear Processor + Print"} )
    processor_id = vec.save_tensor(proc_tensor)
    print(f"Saved Processor: {processor_id}")

    # --- Execute Compute (Requesting Nest 0) ---
    if processor_id and k_id0:
        print("\n--- Running Compute (Nest 0) ---")
        input_data = np.random.rand(1, 5).astype(np.float32)
        compute_context0 = { "input_data": input_data, "required_nest": 0 }
        result0 = vec.compute(processor_id, context=compute_context0)
        print("\nCompute Result (Nest 0):")
        if result0:
            print(f" Status: {result0.get('status')}"); data0 = result0.get('data')
            prov0 = result0.get('provenance', {}); # print(f" Provenance: {prov0}")
            loaded_k0 = prov0.get('loaded_knowledge_ids', []); print(f" Loaded Knowledge IDs: {loaded_k0}")
            # Assert needs adaptation if string tags were used in selection
            # assert k_id0 in loaded_k0, f"Nest 0 loaded {loaded_k0}, expected {k_id0}"
        else: print(" Compute failed.")
    else: print("\nSkipping Compute Nest 0 (Setup Failed).")

    # --- Execute Compute (Requesting Nest 1) ---
    if processor_id and k_id1:
        print("\n--- Running Compute (Nest 1) ---")
        # Reuse input_data
        compute_context1 = { "input_data": input_data, "required_nest": 1 }
        result1 = vec.compute(processor_id, context=compute_context1)
        print("\nCompute Result (Nest 1):")
        if result1:
            print(f" Status: {result1.get('status')}"); data1 = result1.get('data')
            prov1 = result1.get('provenance', {}); # print(f" Provenance: {prov1}")
            loaded_k1 = prov1.get('loaded_knowledge_ids', []); print(f" Loaded Knowledge IDs: {loaded_k1}")
            # Assert needs adaptation if string tags were used in selection
            # assert k_id1 in loaded_k1, f"Nest 1 loaded {loaded_k1}, expected {k_id1}"
        else: print(" Compute failed.")
    else: print("\nSkipping Compute Nest 1 (Setup Failed).")

    # Compare results
    if 'result0' in locals() and 'result1' in locals() and result0 and result1 and result0.get('data') is not None and result1.get('data') is not None:
         try: data0_f = result0['data'].astype(np.float32) if hasattr(result0.get('data'), 'astype') else None; data1_f = result1['data'].astype(np.float32) if hasattr(result1.get('data'), 'astype') else None
         except: data0_f=None; data1_f=None
         if data0_f is not None and data1_f is not None: print(f"\nResult Data Different (Nest 0 vs Nest 1): {not np.allclose(data0_f, data1_f)}")
         else: print("\nCould not compare result data.")

    # --- Spawn Example ---
    if k_id1:
        print("\n--- Spawning Knowledge (Mutate Nest 1) ---")
        mutated_k_id = vec.spawn_tensor(k_id1, strategy="mutate_knowledge", context={"mutation_scale": 0.5})
        print(f" Spawned mutated knowledge: {mutated_k_id}")
    else: print("\nSkipping Spawn Example (Parent Missing).")

    print("\n--- Example Finished ---")