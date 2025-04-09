# === Cell 0: Install Dependencies ===
# Installs necessary Python packages using pip.
# Run this cell first if you are in a new environment.

#!pip install numpy psutil torch transformers accelerate bitsandbytes ipfshttpclient qiskit qiskit-aer requests huggingface_hub -q

print("Dependencies installed/checked.")


#!rm -rf /content/data/


# === Cell 1: Configuration & General Imports ===
# Defines main configuration variables, performs necessary imports,
# handles authentication, and mounts Google Drive.

# --- Standard & External Library Imports ---
import numpy as np
import pickle
import hashlib
import time
import traceback
import os
import gc
import psutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union

try:
    import torch
    from torch import nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not found.")

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, PreTrainedTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: Transformers library not found.")

from google.colab import drive, files, userdata # Colab specific
from huggingface_hub import login             # Hugging Face login

print("Standard/External imports loaded.")

# --- Veector Project Imports ---
# Ensure Veector files (core.py, tensors.py, etc.) are accessible in the Colab environment
# (e.g., uploaded to /content/ or accessible via sys.path)
PROJECT_IMPORTS_OK = False
try:
    from core import Veector, CORE_VERSION
    from tensors import (
        TENSORS_VERSION, TensorCoordinate, create_tensor, MetadataTuple,
        validate_tensor_tuple, validate_tensor, DTYPE_MAPPING, get_tensor_hash,
        TAG_TYPE_PROCESSOR, TAG_TYPE_KNOWLEDGE, TAG_TYPE_CONVERTER, TAG_TYPE_STATE,
        TAG_MODEL_QWEN2, TAG_MODEL_LLAMA3, TAG_MODEL_DEEPSEEK,
        TAG_PREC_FLOAT32, TAG_PREC_FLOAT16, TAG_PREC_BFLOAT16,
        TAG_PREC_INT8, TAG_PREC_INT4,
        TAG_COMP_WEIGHTS, TAG_COMP_BIAS, TAG_COMP_EMBEDDING, TAG_COMP_ATTN_Q,
        TAG_COMP_ATTN_K, TAG_COMP_ATTN_V, TAG_COMP_ATTN_O, TAG_COMP_ATTN_QKV,
        TAG_COMP_FFN_GATE, TAG_COMP_FFN_UP, TAG_COMP_FFN_DOWN, TAG_COMP_LAYERNORM,
        TAG_COMP_LM_HEAD,
        TAG_FUNC_LINEAR, TAG_FUNC_ATTENTION, TAG_FUNC_FFN,
        TAG_FUNC_EMBED_LOOKUP, TAG_FUNC_CAST_DTYPE, TAG_FUNC_RESHAPE,
        TAG_SEMANTIC_HIDDEN_STATE, TAG_SEMANTIC_LOGITS, TAG_SEMANTIC_TOKEN_IDS,
        TAG_SEMANTIC_KV_CACHE,
        tag_layer,
        GROUP_IDX_QWEN_KNOWLEDGE, GROUP_IDX_QWEN_PROCESSOR,
        GROUP_IDX_DEEPSEEK_KNOWLEDGE
    )
    from veectordb import VeectorDB, VEECTORDB_VERSION
    from operations import OPERATIONS_VERSION # Import version, specific ops imported later if needed
    # OP Codes needed globally or frequently
    OP_ADD=[0,0,2]
    OP_MATRIX_MULTIPLY=[30,0,0]
    OP_LINEAR=OP_MATRIX_MULTIPLY
    OP_EMBEDDING_LOOKUP=[40,6,0]
    OP_LINEAR_HEAD=OP_LINEAR
    META_OP_CATEGORY=99
    OP_STORE=[99,0,0]
    OP_LOAD=[99,0,1]
    OP_QWEN2_RMSNORM = [300, 0, 0]
    OP_QWEN2_ATTENTION = [300, 1, 0]
    OP_QWEN2_MLP = [300, 2, 0]
    OP_GET_TUPLE_ELEM_0 = [99, 3, 0]
    OP_GET_TUPLE_ELEM_1 = [99, 3, 1]
    OP_GET_TUPLE_ELEM_2 = [99, 3, 2]

    print("Veector project components imported successfully.")
    print(f"Versions: Core={CORE_VERSION}, Tensors={TENSORS_VERSION}, Ops={OPERATIONS_VERSION}, DB={VEECTORDB_VERSION}")
    PROJECT_IMPORTS_OK = True

except ImportError as e:
    print(f"---!!! FATAL ERROR (ImportError) !!! ---")
    print(f"Specific error: {e}")
    print(f"Could not import required name from Veector files.")
    print(f"Ensure files are UP-TO-DATE and ACCESSIBLE.")
    print(f"-----------------------------------------")
    # Optionally define dummies if needed for notebook structure
except Exception as other_e:
    print(f"---!!! FATAL ERROR (Other Exception during Import) !!! ---")
    print(f"Specific error: {other_e}")
    traceback.print_exc()
    print(f"Check imported files for syntax errors.")
    print(f"----------------------------------------------------------")

if not PROJECT_IMPORTS_OK:
     raise ImportError("Failed to import necessary Veector components.")
if not TORCH_AVAILABLE or not TRANSFORMERS_AVAILABLE:
     raise ImportError("Failed to import Torch or Transformers.")

# --- Configuration Variables ---
# --- Модель и Пути ---
MODEL_NAME: str = "DeepSeek-R1-Distill-Qwen-1.5B" # Имя для файлов и логов
HF_MODEL_SOURCE: str = f"deepseek-ai/{MODEL_NAME}" # Полный идентификатор HF
DB_ROOT_DIR: str = "/content/data" # Корневая директория для данных
DB_PATH: Path = Path(DB_ROOT_DIR) / "db" # Путь к базе данных Veector

# --- Параметры Конвертации и Точности ---
# Используйте torch.float16 для совместимости или torch.bfloat16 если поддерживается
CONVERSION_DTYPE: torch.dtype = torch.float16
# Определяем соответствующий тег точности Veector
if CONVERSION_DTYPE == torch.float16:
    DEFAULT_PRECISION_TAG: int = TAG_PREC_FLOAT16
elif CONVERSION_DTYPE == torch.bfloat16:
    DEFAULT_PRECISION_TAG: int = TAG_PREC_BFLOAT16
elif CONVERSION_DTYPE == torch.float32:
    DEFAULT_PRECISION_TAG: int = TAG_PREC_FLOAT32
else:
    DEFAULT_PRECISION_TAG: int = TAG_PREC_FLOAT16 # Fallback
    print(f"Warning: Unsupported CONVERSION_DTYPE {CONVERSION_DTYPE}, falling back to float16 tag.")

# Квантовать ли Embedding и LM Head слои в INT8?
QUANTIZE_EMBED_LMHEAD: bool = True
QUANTIZED_PRECISION_TAG: int = TAG_PREC_INT8

# --- Параметры Групп и Модели ---
KNOWLEDGE_GROUP_IDX: int = GROUP_IDX_DEEPSEEK_KNOWLEDGE # Используем ID для DeepSeek
PROCESSOR_GROUP_IDX: int = GROUP_IDX_QWEN_PROCESSOR # Процессоры Qwen2
MODEL_TAG: int = TAG_MODEL_DEEPSEEK # Тег модели

# --- Параметры для Тестирования ---
PROMPT_FOR_TESTING: str = "Hello, how are you?"

# --- Создание директорий ---
try:
    DB_PATH.mkdir(parents=True, exist_ok=True)
    print(f"Veector DB directory ensured at: {DB_PATH.resolve()}")
except Exception as e:
    print(f"Error creating DB directory {DB_PATH}: {e}")
    raise

# --- Аутентификация и Google Drive ---
try:
    hf_token = userdata.get('HF_TOKEN')
    if not hf_token:
        raise ValueError("HF_TOKEN not found in Colab secrets. Please add it.")
    login(token=hf_token, add_to_git_credential=False)
    print("Hugging Face login successful.")
except Exception as e:
    print(f"Hugging Face login failed: {e}")
    # Decide if this is fatal or not
    # raise

try:
    drive.mount('/content/drive')
    print("Google Drive mounted successfully.")
except Exception as e:
    print(f"Google Drive mount failed: {e}")
    # Decide if this is fatal or not

print("\n--- Cell 1: Configuration & Imports Finished ---")


# === Cell 2: Knowledge Tensor Conversion ===
# Loads the HF model and converts its parameters into Veector knowledge tensors.
# Saves knowledge tensors, knowledge map, name ID map, and a dedicated knowledge index.
# This cell is self-contained, relying only on variables from Cell 1 (Configuration).

import time
import pickle
import numpy as np
import traceback
import os
import gc
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union

# --- Imports (Redundant but ensures independence) ---
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from core import Veector
    from tensors import (
        TensorCoordinate, create_tensor, validate_tensor,
        TAG_TYPE_KNOWLEDGE, TAG_COMP_WEIGHTS, TAG_COMP_BIAS,
        TAG_COMP_EMBEDDING, TAG_COMP_LM_HEAD, TAG_COMP_LAYERNORM, TAG_COMP_ATTN_Q,
        TAG_COMP_ATTN_K, TAG_COMP_ATTN_V, TAG_COMP_ATTN_O,
        TAG_COMP_FFN_GATE, TAG_COMP_FFN_UP, TAG_COMP_FFN_DOWN,
        tag_layer, DTYPE_MAPPING, TAG_PREC_INT8
    )
    from veectordb import VeectorDB # Needed to access INDEX_FILENAME if saving index here
except ImportError as e:
    print(f"FATAL ERROR in Cell 2: Missing imports: {e}")
    raise

# --- Configuration (Load from Cell 1 variables) ---
# These should be defined in the global scope by running Cell 1
if 'HF_MODEL_SOURCE' not in globals(): raise NameError("HF_MODEL_SOURCE not defined. Run Cell 1.")
if 'DB_PATH' not in globals(): raise NameError("DB_PATH not defined. Run Cell 1.")
if 'CONVERSION_DTYPE' not in globals(): raise NameError("CONVERSION_DTYPE not defined. Run Cell 1.")
if 'DEFAULT_PRECISION_TAG' not in globals(): raise NameError("DEFAULT_PRECISION_TAG not defined. Run Cell 1.")
if 'QUANTIZE_EMBED_LMHEAD' not in globals(): raise NameError("QUANTIZE_EMBED_LMHEAD not defined. Run Cell 1.")
if 'QUANTIZED_PRECISION_TAG' not in globals(): raise NameError("QUANTIZED_PRECISION_TAG not defined. Run Cell 1.")
if 'KNOWLEDGE_GROUP_IDX' not in globals(): raise NameError("KNOWLEDGE_GROUP_IDX not defined. Run Cell 1.")
if 'MODEL_TAG' not in globals(): raise NameError("MODEL_TAG not defined. Run Cell 1.")
if 'MODEL_NAME' not in globals(): raise NameError("MODEL_NAME not defined. Run Cell 1.")

print(f"--- Running Cell 2: Knowledge Conversion for {MODEL_NAME} ---")
print(f"    Target DB: {DB_PATH.resolve()}")
print(f"    Conversion Dtype: {CONVERSION_DTYPE}")
print(f"    Quantize Embed/LMHead: {QUANTIZE_EMBED_LMHEAD}")
start_cell2_time = time.time()

# --- Initialization ---
hf_model = None
vec_knowledge: Optional[Veector] = None
ORIGINAL_NAME_TO_ID_MAP: Dict[str, int] = {}
ID_TO_ORIGINAL_NAME_MAP: Dict[int, str] = {}
NEXT_NAME_ID: int = 0
knowledge_map: Dict[str, str] = {} # HF Name -> Veector Tensor ID
param_count: int = 0
conversion_errors: int = 0

# --- Helper function for Name IDs ---
def get_or_create_name_id(name: Optional[str]) -> int:
    """Assigns and returns a unique ID for a parameter name."""
    global NEXT_NAME_ID, ORIGINAL_NAME_TO_ID_MAP, ID_TO_ORIGINAL_NAME_MAP
    if not name: return -1
    if name in ORIGINAL_NAME_TO_ID_MAP: return ORIGINAL_NAME_TO_ID_MAP[name]
    current_id = NEXT_NAME_ID
    ORIGINAL_NAME_TO_ID_MAP[name] = current_id
    ID_TO_ORIGINAL_NAME_MAP[current_id] = name
    NEXT_NAME_ID += 1
    return current_id

try:
    # --- 1. Load Hugging Face Model ---
    print(f"\nLoading HF Model: {HF_MODEL_SOURCE}...")
    # Load in the target conversion dtype directly if possible
    # Note: Loading directly in float16 might cause issues if operations require float32
    # It might be safer to load in float32 and convert parameter by parameter.
    # Let's load in float32 for robustness during conversion.
    hf_model = AutoModelForCausalLM.from_pretrained(
        HF_MODEL_SOURCE,
        torch_dtype=torch.float32, # Load in float32 for processing
        trust_remote_code=True
    )
    hf_model.eval() # Set to evaluation mode
    model_config = hf_model.config # Get config from loaded model
    print(f"HF Model '{MODEL_NAME}' loaded successfully.")
    gc.collect()

    # --- 2. Initialize Veector Instance for this Cell ---
    print(f"\nInitializing Veector instance for knowledge conversion...")
    # Initialize with the main DB path, it will create an empty index if needed
    # We will save the knowledge index separately later.
    vec_knowledge = Veector(db_dir=DB_PATH, ipfs_enabled=False)
    print(f"Veector initialized. DB Index entries: {len(vec_knowledge.db.index)}")
    # Clear any existing index entries if we want a clean conversion
    # vec_knowledge.db.index = {}
    # vec_knowledge.db._index_dirty = True # Mark dirty if cleared
    # print("Cleared existing index for clean knowledge conversion.")


    # --- 3. Conversion Loop ---
    print(f"\nStarting parameter conversion...")
    total_params = sum(1 for _ in hf_model.named_parameters())
    print(f"Found {total_params} parameters to process.")

    for idx, (name, param) in enumerate(hf_model.named_parameters()):
        loop_start_time = time.time()
        print(f"\nProcessing Param {idx+1}/{total_params}: {name}")
        print(f"  Original Shape: {param.shape} | Dtype: {param.dtype}")

        param_data_fp32: Optional[np.ndarray] = None
        knowledge_data_to_pass: Optional[np.ndarray] = None
        tags: List[int] = []
        metadata_extra_to_pass: Optional[Dict] = None
        dtype_to_pass: Any = None
        final_tags: List[int] = []
        knowledge_coord: Optional[TensorCoordinate] = None
        name_id: int = -1
        create_result: Optional[List] = None
        knowledge_id: Optional[str] = None
        requires_transpose: bool = False

        try:
            # Step 3a: Get data, name ID, base tags, coordinates
            param_data_fp32 = param.data.cpu().numpy() # Already float32
            name_id = get_or_create_name_id(name)
            tags = [TAG_TYPE_KNOWLEDGE, MODEL_TAG] # Base tags
            layer_idx = -1
            group_idx = KNOWLEDGE_GROUP_IDX
            coord_x = 0
            current_nest = 1 # Default nest for knowledge
            is_weight = name.endswith(".weight")
            is_bias = name.endswith(".bias")

            if is_weight: tags.append(TAG_COMP_WEIGHTS)
            elif is_bias: tags.append(TAG_COMP_BIAS)

            # Determine component type, layer index, and X coordinate
            if "model.embed_tokens.weight" in name:
                 tags.append(TAG_COMP_EMBEDDING); coord_x = 0
            elif "lm_head.weight" in name:
                 tags.append(TAG_COMP_LM_HEAD); coord_x = 1; requires_transpose = True
            elif "model.norm.weight" in name:
                 layer_idx = model_config.num_hidden_layers; tags.append(TAG_COMP_LAYERNORM); coord_x = 0
            elif ".layers." in name:
                try:
                    layer_part = name.split('.layers.')[1]
                    layer_idx = int(layer_part.split('.')[0])
                    if layer_idx >= 0: tags.append(tag_layer(layer_idx))
                    else: raise ValueError(f"Invalid L idx: {layer_idx}")

                    component_tag_layer = None
                    if "self_attn" in name:
                        if "q_proj.weight" in name: component_tag_layer = TAG_COMP_ATTN_Q; coord_x = 10; requires_transpose = True
                        elif "q_proj.bias" in name: component_tag_layer = TAG_COMP_ATTN_Q; coord_x = 11
                        elif "k_proj.weight" in name: component_tag_layer = TAG_COMP_ATTN_K; coord_x = 20; requires_transpose = True
                        elif "k_proj.bias" in name: component_tag_layer = TAG_COMP_ATTN_K; coord_x = 21
                        elif "v_proj.weight" in name: component_tag_layer = TAG_COMP_ATTN_V; coord_x = 30; requires_transpose = True
                        elif "v_proj.bias" in name: component_tag_layer = TAG_COMP_ATTN_V; coord_x = 31
                        elif "o_proj.weight" in name: component_tag_layer = TAG_COMP_ATTN_O; coord_x = 40; requires_transpose = True
                    elif "mlp" in name:
                        if "gate_proj.weight" in name: component_tag_layer = TAG_COMP_FFN_GATE; coord_x = 50; requires_transpose = True
                        elif "up_proj.weight" in name: component_tag_layer = TAG_COMP_FFN_UP; coord_x = 60; requires_transpose = True
                        elif "down_proj.weight" in name: component_tag_layer = TAG_COMP_FFN_DOWN; coord_x = 70; requires_transpose = True
                    elif "input_layernorm.weight" in name: component_tag_layer = TAG_COMP_LAYERNORM; coord_x = 1
                    elif "post_attention_layernorm.weight" in name: component_tag_layer = TAG_COMP_LAYERNORM; coord_x = 2

                    if component_tag_layer: tags.append(component_tag_layer)
                    elif not is_weight and not is_bias: print(f"  WARN: Unrecognized comp in L{layer_idx}: {name}"); coord_x = 99
                except Exception as parse_e:
                    print(f"  Error parsing layer for {name}: {parse_e}"); conversion_errors += 1; continue
            else:
                print(f"  WARN: Param unmatched: {name}"); layer_idx = -1; coord_x = 999

            knowledge_coord = TensorCoordinate(layer=layer_idx, group=group_idx, nest=current_nest, x=coord_x)

            # Step 3b: Quantization / Type Casting / Transposition
            quantization_scale = None
            current_precision_tag = DEFAULT_PRECISION_TAG
            data_before_save = None
            target_np_dtype = np.float16 # Default target
            if CONVERSION_DTYPE == torch.float16: target_np_dtype = np.float16
            elif CONVERSION_DTYPE == torch.bfloat16: target_np_dtype = np.float32 # Numpy has no bfloat16, store as float32
            elif CONVERSION_DTYPE == torch.float32: target_np_dtype = np.float32

            # Special handling for Embedding and LM Head (Quantization)
            if QUANTIZE_EMBED_LMHEAD and (name == "model.embed_tokens.weight" or name == "lm_head.weight"):
                try:
                    print(f"  Quantizing {name} to INT8...")
                    abs_max = np.max(np.abs(param_data_fp32)); scale = 1.0
                    if abs_max >= 1e-9: scale = abs_max / 127.0
                    scale = max(scale, 1e-9) # Prevent division by zero
                    quantized_data = np.round(param_data_fp32 / scale).astype(np.int8)
                    data_before_save = quantized_data
                    dtype_to_pass = np.int8
                    quantization_scale = float(scale)
                    current_precision_tag = QUANTIZED_PRECISION_TAG # Use INT8 tag
                    metadata_extra_to_pass = {"quantization_scale": quantization_scale}
                    print(f"    Quantized Shape: {data_before_save.shape}, Scale: {quantization_scale:.4f}")
                    # Transpose LM Head AFTER quantization
                    if name == "lm_head.weight": # requires_transpose is True here
                        print("    Transposing quantized LM Head weights...")
                        data_before_save = data_before_save.T
                        print(f"    Transposed Shape: {data_before_save.shape}")
                except Exception as quant_e:
                     print(f"  ERROR quantizing {name}: {quant_e}"); conversion_errors += 1; continue
            else: # Standard parameters (cast and maybe transpose)
                try:
                    print(f"  Casting {name} to {target_np_dtype}...")
                    data_before_save = param_data_fp32.astype(target_np_dtype)
                    dtype_to_pass = data_before_save.dtype
                    current_precision_tag = DEFAULT_PRECISION_TAG # Use default precision tag
                    metadata_extra_to_pass = None
                    # Transpose if required
                    if requires_transpose:
                        print(f"    Transposing {name} weights...")
                        data_before_save = data_before_save.T
                        print(f"    Transposed Shape: {data_before_save.shape}")
                except Exception as cast_e:
                     print(f"  ERROR casting/transposing {name}: {cast_e}"); conversion_errors += 1; continue

            knowledge_data_to_pass = data_before_save
            final_shape_to_save = knowledge_data_to_pass.shape if knowledge_data_to_pass is not None else None

            # Step 3c: Finalize tags
            final_tags = list(tags) # Start with base tags
            # Remove default precision if a specific one was applied
            if current_precision_tag != DEFAULT_PRECISION_TAG and DEFAULT_PRECISION_TAG in final_tags:
                 final_tags.remove(DEFAULT_PRECISION_TAG)
            # Add the actual precision tag used
            if current_precision_tag:
                final_tags.append(current_precision_tag)
            final_tags = sorted(list(set(final_tags))) # Ensure uniqueness and order

            print(f"  Final Tags: {final_tags}"); print(f"  Coordinate: {knowledge_coord}")
            print(f"  Data to save: dtype={dtype_to_pass}, shape={final_shape_to_save}")
            if metadata_extra_to_pass: print(f"  Extra Metadata: {metadata_extra_to_pass}")

            # Step 3d: Create Veector Tensor structure (list)
            create_result = vec_knowledge.create_tensor(
                 coord=knowledge_coord,
                 tensor_type="knowledge",
                 knowledge_data=knowledge_data_to_pass,
                 tags=final_tags,
                 dtype=dtype_to_pass,
                 shape=final_shape_to_save,
                 name_id=name_id,
                 metadata_extra=metadata_extra_to_pass, # Pass quantization scale if any
                 status="active"
             )
            if not validate_tensor(create_result):
                 raise ValueError(f"Invalid tensor structure created for {name}")

            # Step 3e: Save Tensor
            knowledge_id = vec_knowledge.save_tensor(create_result) # Pass the list structure

            if knowledge_id:
                knowledge_map[name] = knowledge_id # Store HF name -> Veector ID mapping
                param_count += 1
                print(f"    Saved knowledge tensor with ID: {knowledge_id}")
            else:
                conversion_errors += 1
                print(f"  ERROR saving tensor for {name}")

        except Exception as create_save_e:
            print(f"  ERROR during create/save for {name}: {create_save_e}")
            traceback.print_exc()
            conversion_errors += 1
        finally:
            # Clean up intermediate numpy array for this parameter
            if param_data_fp32 is not None: del param_data_fp32
            # loop_end_time = time.time()
            # print(f"  Param {idx+1} time: {loop_end_time - loop_start_time:.2f}s") # Optional timing log

    # --- End of Conversion Loop ---

    print(f"\n--- Finished saving {param_count} knowledge tensors to {vec_knowledge.db.db_root_path if vec_knowledge.db else 'N/A'} ---")
    if conversion_errors > 0:
        print(f"!!! WARNING: {conversion_errors} errors occurred during knowledge conversion !!!")

    # --- 4. Save Maps and Knowledge Index ---
    # Save Name ID Map
    name_map_file = DB_PATH / f"{MODEL_NAME}_name_id_map.pkl"
    try:
        map_data_to_save = {
            "name_to_id": ORIGINAL_NAME_TO_ID_MAP,
            "id_to_name": ID_TO_ORIGINAL_NAME_MAP,
            "next_id": NEXT_NAME_ID
        }
        print(f"\nSaving Name <-> ID map ({len(ORIGINAL_NAME_TO_ID_MAP)} entries) to {name_map_file}...")
        with open(name_map_file, 'wb') as f: pickle.dump(map_data_to_save, f)
        print(f"Name ID map saved successfully.")
    except Exception as e:
        print(f"  Error saving name ID map: {e}")

    # Save Knowledge Map (HF Name -> Veector ID)
    knowledge_map_file = DB_PATH / f"{MODEL_NAME}_knowledge_map.pkl"
    try:
        print(f"\nSaving Knowledge map ({len(knowledge_map)} entries) to {knowledge_map_file}...")
        with open(knowledge_map_file, 'wb') as f: pickle.dump(knowledge_map, f)
        print(f"Knowledge map saved successfully.")
    except Exception as e:
        print(f"  Error saving knowledge map: {e}")

    # Save the index containing ONLY the knowledge tensors created in this cell
    knowledge_index_file = DB_PATH / f"{MODEL_NAME}_knowledge_index.pkl"
    try:
        print(f"\nSaving Knowledge index ({len(vec_knowledge.db.index)} entries) to {knowledge_index_file}...")
        # Use save_index_as to save the current index (only knowledge) to a specific file
        vec_knowledge.db.save_index_as(knowledge_index_file)
        print(f"Knowledge index saved successfully to {knowledge_index_file.name}.")
    except Exception as e:
        print(f"  Error saving knowledge index: {e}")
        traceback.print_exc()

    # --- 5. Cleanup ---
    print("\nCleaning up resources for Cell 2...")
    if vec_knowledge and hasattr(vec_knowledge, 'db') and vec_knowledge.db:
        vec_knowledge.db.close() # Close DB connection (won't save main index)
        print("Veector DB connection closed.")
    if hf_model is not None:
        del hf_model
        print("HF model deleted.")
    if 'vec_knowledge' in locals():
        del vec_knowledge
        print("Veector instance deleted.")

    if 'torch' in locals() and hasattr(torch, 'cuda') and torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("CUDA cache cleared.")
    gc.collect()
    print("Garbage collection run.")

except Exception as cell2_e:
    print(f"\n---!!! FATAL ERROR in Cell 2 Execution: {cell2_e} !!!---")
    traceback.print_exc()
    # Ensure cleanup happens even on error
    if 'vec_knowledge' in locals() and vec_knowledge and hasattr(vec_knowledge, 'db') and vec_knowledge.db:
        try: vec_knowledge.db.close()
        except: pass
    if 'hf_model' in locals() and hf_model is not None: del hf_model
    if 'vec_knowledge' in locals(): del vec_knowledge
    gc.collect()
    if 'torch' in locals() and hasattr(torch, 'cuda') and torch.cuda.is_available(): torch.cuda.empty_cache()
    raise # Re-raise the exception to stop notebook execution

finally:
    end_cell2_time = time.time()
    print(f"\n--- Cell 2 Finished in {end_cell2_time - start_cell2_time:.2f} seconds ---")



# !rm -rf ./data/db/g500/ && rm ./data/db/DeepSeek-R1-Distill-Qwen-1.5B_proc_map.pkl


# === Cell 3 (Updated v1.1 - Handle Tuple from MLP Op) ===
# Loads model config, knowledge map, name map, and knowledge index from Cell 2 outputs.
# Initializes Veector with the knowledge index.
# Defines and saves processor tensors (Embedding, Attn, FFN, Norm, LM Head).
# FFN processor sequence updated to handle tuple output from OP_QWEN2_MLP.
# Saves the processor map and the final combined index (tensor_index.pkl).
# This cell is self-contained, relying only on Cell 1 variables and Cell 2 output files.

import time
import pickle
import numpy as np
import traceback
import os
import gc
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union

# --- Imports (Redundant but ensures independence) ---
try:
    import torch
    from transformers import AutoConfig # Only need AutoConfig here
    from core import Veector
    from tensors import (
        TENSORS_VERSION, TensorCoordinate, create_tensor, MetadataTuple,
        validate_tensor_tuple, validate_tensor, DTYPE_MAPPING, get_tensor_hash,
        TAG_TYPE_PROCESSOR, TAG_FUNC_EMBED_LOOKUP, TAG_FUNC_ATTENTION,
        TAG_FUNC_FFN, TAG_FUNC_LINEAR, TAG_COMP_LAYERNORM, TAG_MODEL_DEEPSEEK,
        tag_layer, GROUP_IDX_QWEN_PROCESSOR, GROUP_IDX_QWEN_KNOWLEDGE,
        TAG_COMP_EMBEDDING, TAG_COMP_WEIGHTS, TAG_COMP_BIAS, TAG_COMP_ATTN_Q,
        TAG_COMP_ATTN_K, TAG_COMP_ATTN_V, TAG_COMP_ATTN_O, TAG_COMP_FFN_GATE,
        TAG_COMP_FFN_UP, TAG_COMP_FFN_DOWN, TAG_COMP_LM_HEAD,
        TAG_PREC_FLOAT32, TAG_PREC_FLOAT16, TAG_PREC_BFLOAT16, TAG_PREC_INT8
    )
    from veectordb import VeectorDB, VEECTORDB_VERSION
    from operations import OPERATIONS_VERSION # Import version, specific ops imported later if needed
    # OP Codes needed globally or frequently
    OP_ADD=[0,0,2]
    OP_MATRIX_MULTIPLY=[30,0,0]
    OP_LINEAR=OP_MATRIX_MULTIPLY
    OP_EMBEDDING_LOOKUP=[40,6,0]
    OP_LINEAR_HEAD=OP_LINEAR
    META_OP_CATEGORY=99
    OP_STORE=[99,0,0]
    OP_LOAD=[99,0,1]
    OP_QWEN2_RMSNORM = [300, 0, 0]
    OP_QWEN2_ATTENTION = [300, 1, 0]
    OP_QWEN2_MLP = [300, 2, 0]
    OP_GET_TUPLE_ELEM_0 = [99, 3, 0]
    OP_GET_TUPLE_ELEM_1 = [99, 3, 1]
    OP_GET_TUPLE_ELEM_2 = [99, 3, 2]
    OP_GET_TUPLE_ELEM_3 = [99, 3, 3] # Define needed OP code for tuple element 3

    print("External and Veector libraries imported successfully.")
    # print(f"Using Core: {CORE_VERSION}, Tensors: {TENSORS_VERSION}, Ops: {OPERATIONS_VERSION}, DB: {VEECTORDB_VERSION}") # Optional print
    # Add version checks if desired

except ImportError as e:
    print(f"FATAL ERROR: Failed to import components: {e}")
    raise
except Exception as e_other:
    print(f"FATAL ERROR during imports: {e_other}")
    raise

# --- Configuration (Load from Cell 1 variables) ---
if 'MODEL_NAME' not in globals(): raise NameError("MODEL_NAME not defined. Run Cell 1.")
if 'HF_MODEL_SOURCE' not in globals(): raise NameError("HF_MODEL_SOURCE not defined. Run Cell 1.")
if 'DB_PATH' not in globals(): raise NameError("DB_PATH not defined. Run Cell 1.")
if 'PROCESSOR_GROUP_IDX' not in globals(): raise NameError("PROCESSOR_GROUP_IDX not defined. Run Cell 1.")
if 'MODEL_TAG' not in globals(): raise NameError("MODEL_TAG not defined. Run Cell 1.")
if 'DEFAULT_PRECISION_TAG' not in globals(): raise NameError("DEFAULT_PRECISION_TAG not defined. Run Cell 1.")
if 'QUANTIZED_PRECISION_TAG' not in globals(): raise NameError("QUANTIZED_PRECISION_TAG not defined. Run Cell 1.")
prec_tag_weights = DEFAULT_PRECISION_TAG
prec_tag_quant = QUANTIZED_PRECISION_TAG

print(f"--- Running Cell 3 (v1.1): Processor Creation for {MODEL_NAME} ---")
print(f"    Target DB: {DB_PATH.resolve()}")
start_cell3_time = time.time()

# --- Initialization ---
model_config = None
knowledge_map: Optional[Dict[str, str]] = None
name_id_map_data: Optional[Dict] = None
vec_processor: Optional[Veector] = None
processor_errors: int = 0
processor_map: Dict[str, str] = {}

# --- File Paths for Inputs from Cell 2 ---
knowledge_map_filepath = DB_PATH / f"{MODEL_NAME}_knowledge_map.pkl"
name_map_filepath = DB_PATH / f"{MODEL_NAME}_name_id_map.pkl"
knowledge_index_filepath = DB_PATH / f"{MODEL_NAME}_knowledge_index.pkl"
main_index_filepath = DB_PATH / VeectorDB.INDEX_FILENAME

try:
    # --- 1. Load Model Config ---
    print(f"\nLoading Model Config from: {HF_MODEL_SOURCE}...")
    model_config = AutoConfig.from_pretrained(HF_MODEL_SOURCE, trust_remote_code=True)
    num_layers = model_config.num_hidden_layers
    num_attention_heads = model_config.num_attention_heads
    num_key_value_heads = getattr(model_config, 'num_key_value_heads', num_attention_heads)
    hidden_size = model_config.hidden_size
    head_dim = hidden_size // num_attention_heads
    rms_norm_eps = model_config.rms_norm_eps
    hidden_act_function_name = model_config.hidden_act # Needed for FFN processor
    rope_theta_value = getattr(model_config, 'rope_theta', 10000.0)
    print(f"Model Config loaded. HiddenAct='{hidden_act_function_name}', RopeTheta={rope_theta_value}")

    # --- 2. Load Maps from Cell 2 ---
    print(f"\nLoading maps from {DB_PATH}...")
    if not knowledge_map_filepath.is_file(): raise FileNotFoundError(f"Knowledge map file not found: {knowledge_map_filepath}")
    if not knowledge_index_filepath.is_file(): raise FileNotFoundError(f"Knowledge index file not found: {knowledge_index_filepath}")

    with open(knowledge_map_filepath, 'rb') as f: knowledge_map = pickle.load(f)
    print(f"Loaded knowledge map ({len(knowledge_map)} entries).")

    if name_map_filepath.is_file():
         with open(name_map_filepath, 'rb') as f: name_id_map_data = pickle.load(f)
         print(f"Loaded name ID map.")
    else: print(f"Warning: Name ID map file not found at {name_map_filepath}")

    # --- 3. Initialize Veector with Knowledge Index ---
    print(f"\nInitializing Veector instance for processor creation...")
    print(f"Loading initial index from: '{knowledge_index_filepath.name}'")
    vec_processor = Veector(db_dir=DB_PATH, initial_index_path=knowledge_index_filepath)
    print(f"Veector initialized. DB Index entries loaded: {len(vec_processor.db.index)}")
    if len(vec_processor.db.index) == 0: print(f"WARNING: Loaded knowledge index is empty!")
    vec_processor.db.index_path = main_index_filepath
    print(f"Default index save path set to: '{vec_processor.db.index_path.name}'")
    vec_processor.db._index_dirty = True # Mark dirty as we will add processors

    # --- 4. Helper Functions ---
    def find_knowledge_id(hf_param_name: str) -> Optional[str]:
        if knowledge_map is None: return None
        return knowledge_map.get(hf_param_name)

    def create_and_save_processor(name: str, coord: TensorCoordinate, tags: List[int], interface: Dict, ops_sequences: Dict):
        global processor_errors # Use global to modify counter
        # processor_map and vec_processor are accessible from outer scope
        proc_id: Optional[str] = None
        try:
            print(f"  Defining Processor: {name} at {coord}")
            tensor_structure = vec_processor.create_tensor(
                coord=coord, tensor_type="processor", tags=tags,
                interface=interface, ops_sequences=ops_sequences,
                status="active", name_id=-1
            )
            if not validate_tensor(tensor_structure): raise ValueError(f"Invalid list structure for {name}")

            proc_id = vec_processor.save_tensor(tensor_structure)

            if proc_id:
                map_key = ""
                if "Embedding" in name: map_key = "embedding"
                elif "Final Norm" in name: map_key = "final_norm"
                elif "LM Head" in name: map_key = "lm_head"
                elif "Attention Processor L" in name:
                  try: layer_idx = int(name.split("L")[-1]); map_key = f"attn_{layer_idx}"
                  except: pass
                elif "FFN Processor L" in name:
                  try: layer_idx = int(name.split("L")[-1]); map_key = f"ffn_{layer_idx}"
                  except: pass

                if map_key:
                    processor_map[map_key] = proc_id # Modify outer scope dict
                    print(f"    SUCCESS: Saved {name} with ID: {proc_id} (Key: {map_key})")
                else: print(f"    WARN: Saved {name} with ID: {proc_id}, but no map key.")
            else:
                processor_errors += 1; print(f"    ERROR saving processor {name}")
        except Exception as e:
            print(f"    ERROR during definition/saving of processor {name}: {e}")
            traceback.print_exc(); processor_errors += 1
        return proc_id

    # --- 5. Define and Save Processors ---
    print("\n--- Defining and Saving Veector Processor Tensors ---")

    # 5.A Embedding Processor
    print("\n--- Defining Embedding Processor ---")
    try:
        coord = TensorCoordinate(layer=-1, group=PROCESSOR_GROUP_IDX, nest=0, x=0)
        tags = [TAG_TYPE_PROCESSOR, TAG_FUNC_EMBED_LOOKUP, MODEL_TAG]
        param_name = "embedding_matrix"; hf_name = "model.embed_tokens.weight"
        kn_tags = [TAG_COMP_EMBEDDING, MODEL_TAG, TAG_COMP_WEIGHTS, prec_tag_quant]
        kid = find_knowledge_id(hf_name)
        if not kid: raise ValueError(f"Embedding knowledge ID not found for '{hf_name}'.")
        interface = {"inputs": [{"name":"token_ids", "dtype":"int64"}], "outputs": [{"name":"hidden_states", "dtype":"float16"}], "knowledge_needed": [{"param_name": param_name, "tags": kn_tags, "knowledge_id": kid}]}
        ops_sequences = {'default': [[OP_EMBEDDING_LOOKUP, {"embedding_matrix": param_name}]]}
        create_and_save_processor("Embedding Processor", coord, tags, interface, ops_sequences)
    except Exception as e: print(f"Error defining Embedding Processor: {e}"); traceback.print_exc(); processor_errors += 1

    # 5.B Transformer Layers
    print(f"\n--- Defining Transformer Layer Processors (0 to {num_layers-1}) ---")
    for layer_idx in range(num_layers):
        layer_tag = tag_layer(layer_idx)
        print(f"  Processing Layer {layer_idx}...")

        # --- Attention Processor ---
        try:
            coord_attn = TensorCoordinate(layer=layer_idx, group=PROCESSOR_GROUP_IDX, nest=0, x=0)
            tags_attn = [TAG_TYPE_PROCESSOR, TAG_FUNC_ATTENTION, layer_tag, MODEL_TAG]
            kn_defs_attn = [
                {"p":f"L{layer_idx}_input_norm_w", "t":[TAG_COMP_LAYERNORM, layer_tag, MODEL_TAG, TAG_COMP_WEIGHTS, prec_tag_weights], "f":f"model.layers.{layer_idx}.input_layernorm.weight"},
                {"p":f"L{layer_idx}_q_w",   "t":[TAG_COMP_ATTN_Q, layer_tag, MODEL_TAG, TAG_COMP_WEIGHTS, prec_tag_weights], "f":f"model.layers.{layer_idx}.self_attn.q_proj.weight"},
                {"p":f"L{layer_idx}_q_b",   "t":[TAG_COMP_ATTN_Q, layer_tag, MODEL_TAG, TAG_COMP_BIAS, prec_tag_weights],    "f":f"model.layers.{layer_idx}.self_attn.q_proj.bias", "opt": True},
                {"p":f"L{layer_idx}_k_w",   "t":[TAG_COMP_ATTN_K, layer_tag, MODEL_TAG, TAG_COMP_WEIGHTS, prec_tag_weights], "f":f"model.layers.{layer_idx}.self_attn.k_proj.weight"},
                {"p":f"L{layer_idx}_k_b",   "t":[TAG_COMP_ATTN_K, layer_tag, MODEL_TAG, TAG_COMP_BIAS, prec_tag_weights],    "f":f"model.layers.{layer_idx}.self_attn.k_proj.bias", "opt": True},
                {"p":f"L{layer_idx}_v_w",   "t":[TAG_COMP_ATTN_V, layer_tag, MODEL_TAG, TAG_COMP_WEIGHTS, prec_tag_weights], "f":f"model.layers.{layer_idx}.self_attn.v_proj.weight"},
                {"p":f"L{layer_idx}_v_b",   "t":[TAG_COMP_ATTN_V, layer_tag, MODEL_TAG, TAG_COMP_BIAS, prec_tag_weights],    "f":f"model.layers.{layer_idx}.self_attn.v_proj.bias", "opt": True},
                {"p":f"L{layer_idx}_o_w",   "t":[TAG_COMP_ATTN_O, layer_tag, MODEL_TAG, TAG_COMP_WEIGHTS, prec_tag_weights], "f":f"model.layers.{layer_idx}.self_attn.o_proj.weight"},
            ]
            knowledge_needs_attn = []
            missing_essential = False
            for kdef in kn_defs_attn:
                kid = find_knowledge_id(kdef["f"])
                is_opt = kdef.get("opt", False)
                if kid: knowledge_needs_attn.append({"param_name": kdef["p"], "tags": kdef["t"], "knowledge_id": kid, "optional": is_opt})
                elif not is_opt: missing_essential = True; print(f"ERROR: Missing essential knowledge for Attn L{layer_idx}: {kdef['p']} ({kdef['f']})")

            if not missing_essential:
                interface_attn = { "inputs": [ {"name": "hidden_state_in"}, {"name": "residual_input"}, {"name": "position_ids"}, {"name": "past_key", "optional": True}, {"name": "past_value", "optional": True}, {"name": "start_pos", "dtype": "int", "optional": True}, {"name": "total_seq_len", "dtype": "int", "optional": True} ], "outputs": [{"name": "attn_block_output"}], "knowledge_needed": knowledge_needs_attn }
                ops_sequences_attn = {'default': [ [OP_STORE, 'residual_attn'], [OP_QWEN2_RMSNORM, {"norm_weight": f"L{layer_idx}_input_norm_w", "eps": rms_norm_eps}], [OP_QWEN2_ATTENTION, {"q_weights": f"L{layer_idx}_q_w", "k_weights": f"L{layer_idx}_k_w", "v_weights": f"L{layer_idx}_v_w", "o_weights": f"L{layer_idx}_o_w", "q_bias": f"L{layer_idx}_q_b", "k_bias": f"L{layer_idx}_k_b", "v_bias": f"L{layer_idx}_v_b", "position_ids": "position_ids", "past_key": "past_key", "past_value": "past_value", "start_pos": "start_pos", "total_seq_len": "total_seq_len", "num_heads": num_attention_heads, "num_kv_heads": num_key_value_heads, "head_dim": head_dim, "hidden_size": hidden_size, "layer_idx": layer_idx, "rope_theta": rope_theta_value}], [OP_STORE, 'attn_tuple_output'], [OP_LOAD, 'attn_tuple_output'], [OP_GET_TUPLE_ELEM_1], [OP_STORE, 'k_cache_out'], [OP_LOAD, 'attn_tuple_output'], [OP_GET_TUPLE_ELEM_2], [OP_STORE, 'v_cache_out'], [OP_LOAD, 'attn_tuple_output'], [OP_GET_TUPLE_ELEM_0], [OP_ADD, {"input_a": "residual_attn", "input_b": "_"}] ]}
                create_and_save_processor(f"Attention Processor L{layer_idx}", coord_attn, tags_attn, interface_attn, ops_sequences_attn)
            else: processor_errors += 1
        except Exception as e: print(f"Error defining Attn L{layer_idx}: {e}"); traceback.print_exc(); processor_errors += 1

        # --- FFN Processor (Handles tuple output from MLP) ---
        try:
            coord_ffn = TensorCoordinate(layer=layer_idx, group=PROCESSOR_GROUP_IDX, nest=0, x=1)
            tags_ffn = [TAG_TYPE_PROCESSOR, TAG_FUNC_FFN, layer_tag, MODEL_TAG]
            kn_defs_ffn = [
                {"p": f"L{layer_idx}_post_attn_norm_w", "t": [TAG_COMP_LAYERNORM, layer_tag, MODEL_TAG, TAG_COMP_WEIGHTS, prec_tag_weights], "f": f"model.layers.{layer_idx}.post_attention_layernorm.weight"},
                {"p": f"L{layer_idx}_gate_w", "t": [TAG_COMP_FFN_GATE, layer_tag, MODEL_TAG, TAG_COMP_WEIGHTS, prec_tag_weights],  "f": f"model.layers.{layer_idx}.mlp.gate_proj.weight"},
                {"p": f"L{layer_idx}_up_w",   "t": [TAG_COMP_FFN_UP, layer_tag, MODEL_TAG, TAG_COMP_WEIGHTS, prec_tag_weights],    "f": f"model.layers.{layer_idx}.mlp.up_proj.weight"},
                {"p": f"L{layer_idx}_down_w", "t": [TAG_COMP_FFN_DOWN, layer_tag, MODEL_TAG, TAG_COMP_WEIGHTS, prec_tag_weights],  "f": f"model.layers.{layer_idx}.mlp.down_proj.weight"},
            ]
            knowledge_needs_ffn = []
            missing_essential = False
            for kdef in kn_defs_ffn:
                kid = find_knowledge_id(kdef["f"])
                is_opt = kdef.get("opt", False)
                if kid: knowledge_needs_ffn.append({"param_name": kdef["p"], "tags": kdef["t"], "knowledge_id": kid, "optional": is_opt})
                elif not is_opt: missing_essential = True; print(f"ERROR: Missing essential knowledge for FFN L{layer_idx}: {kdef['p']} ({kdef['f']})")

            if not missing_essential:
                interface_ffn = { "inputs": [{"name":"attn_block_output"}, {"name":"residual_input"}], "outputs": [{"name":"layer_output"}], "knowledge_needed": knowledge_needs_ffn }
                # --- UPDATED FFN Operation Sequence ---
                ops_sequences_ffn = {'default': [
                    # 1. Store input for the second residual connection
                    [OP_STORE, 'residual_ffn'],
                    # 2. Apply Post-Attention RMS Normalization
                    [OP_QWEN2_RMSNORM, {"norm_weight": f"L{layer_idx}_post_attn_norm_w", "eps": rms_norm_eps}],
                    # 3. Execute the MLP block operation (now returns a tuple)
                    [OP_QWEN2_MLP, {
                        "gate_weights": f"L{layer_idx}_gate_w",
                        "up_weights": f"L{layer_idx}_up_w",
                        "down_weights": f"L{layer_idx}_down_w",
                        "hidden_act": hidden_act_function_name # Use variable from config
                    }],
                    # 4. Store the results tuple from MLP
                    [OP_STORE, 'mlp_results_tuple'],
                    # 5. (Optional) Store intermediate MLP results for debugging
                    # These assume OP_GET_TUPLE_ELEM_X are defined up to index 3
                    [OP_LOAD, 'mlp_results_tuple'], [OP_GET_TUPLE_ELEM_1], [OP_STORE, f'L{layer_idx}_dbg_mlp_gate_out'],
                    [OP_LOAD, 'mlp_results_tuple'], [OP_GET_TUPLE_ELEM_2], [OP_STORE, f'L{layer_idx}_dbg_mlp_up_out'],
                    [OP_LOAD, 'mlp_results_tuple'], [OP_GET_TUPLE_ELEM_3], [OP_STORE, f'L{layer_idx}_dbg_mlp_activated'],
                    # 6. Extract the final MLP output (element 0) to be used in residual add
                    [OP_LOAD, 'mlp_results_tuple'],
                    [OP_GET_TUPLE_ELEM_0], # current_data is now the final MLP output tensor (before residual)
                    # 7. Add the second residual connection
                    [OP_ADD, {"input_a": "residual_ffn", "input_b": "_"}] # Adds stored input to MLP output
                ]}
                # --- END UPDATED FFN Operation Sequence ---
                create_and_save_processor(f"FFN Processor L{layer_idx}", coord_ffn, tags_ffn, interface_ffn, ops_sequences_ffn)
            else: processor_errors += 1
        except Exception as e: print(f"Error defining FFN L{layer_idx}: {e}"); traceback.print_exc(); processor_errors += 1
    # --- End Layer Loop ---

    # 5.C Final Norm Processor
    print("\n--- Defining Final Norm Processor ---")
    try:
        coord = TensorCoordinate(layer=-1, group=PROCESSOR_GROUP_IDX, nest=0, x=1)
        tags = [TAG_TYPE_PROCESSOR, TAG_COMP_LAYERNORM, MODEL_TAG]
        kn_tags = [TAG_COMP_LAYERNORM, MODEL_TAG, TAG_COMP_WEIGHTS, prec_tag_weights]
        hf_name = "model.norm.weight"; kid = find_knowledge_id(hf_name)
        if not kid: raise ValueError(f"Final Norm knowledge ID not found for '{hf_name}'.")
        knowledge_needs = [{"param_name": "norm_weight", "tags": kn_tags, "knowledge_id": kid}]
        interface = {"inputs": [{"name":"final_hidden_state"}], "outputs": [{"name":"final_normed_state"}], "knowledge_needed": knowledge_needs}
        ops_sequences = {'default': [[OP_QWEN2_RMSNORM, {"norm_weight": "norm_weight", "eps": rms_norm_eps}]]}
        create_and_save_processor("Final Norm Processor", coord, tags, interface, ops_sequences)
    except Exception as e: print(f"Error defining Final Norm Processor: {e}"); traceback.print_exc(); processor_errors += 1

    # 5.D LM Head Processor
    print("\n--- Defining LM Head Processor ---")
    try:
        coord = TensorCoordinate(layer=-1, group=PROCESSOR_GROUP_IDX, nest=0, x=2)
        tags = [TAG_TYPE_PROCESSOR, TAG_FUNC_LINEAR, MODEL_TAG]
        kn_tags = [TAG_COMP_LM_HEAD, MODEL_TAG, TAG_COMP_WEIGHTS, prec_tag_quant] # Expect quantized
        hf_name = "lm_head.weight"; kid = find_knowledge_id(hf_name)
        if not kid: raise ValueError(f"LM Head knowledge ID not found for '{hf_name}'.")
        knowledge_needs = [{"param_name": "lm_head_weights", "tags": kn_tags, "knowledge_id": kid}]
        interface = {"inputs": [{"name":"final_normed_state"}], "outputs": [{"name":"logits"}], "knowledge_needed": knowledge_needs}
        ops_sequences = {'default': [[OP_LINEAR_HEAD, {"weights": "lm_head_weights"}]]}
        create_and_save_processor("LM Head Processor", coord, tags, interface, ops_sequences)
    except Exception as e: print(f"Error defining LM Head Processor: {e}"); traceback.print_exc(); processor_errors += 1

    # --- 6. Finalization ---
    print(f"\n--- Finalizing Cell 3 ({processor_errors} errors during processor creation) ---")

    # Save Processor Map
    processor_map_filepath = DB_PATH / f"{MODEL_NAME}_proc_map.pkl"
    try:
        expected_proc_count = 3 + 2 * num_layers # Embed, Norm, Head + 2*Layers
        if processor_errors == 0 and len(processor_map) == expected_proc_count:
            print(f"Saving processor map ({len(processor_map)} entries) to {processor_map_filepath}...")
            with open(processor_map_filepath, 'wb') as f: pickle.dump(processor_map, f)
            print(f"Processor map saved successfully.")
        elif processor_errors > 0:
            print(f"Processor map NOT saved due to {processor_errors} errors during creation.")
        else:
             print(f"WARN: Processor map has incorrect entry count ({len(processor_map)} vs {expected_proc_count}). NOT SAVED.")
    except Exception as e: print(f"Error saving processor map: {e}")

    # --- 7. Cleanup ---
    print("\nCleaning up resources for Cell 3...")
    if vec_processor and hasattr(vec_processor, 'db') and vec_processor.db:
        print(f"Closing Veector DB connection (saving main index to '{main_index_filepath.name}')...")
        print(f"Index size before final save in Cell 3: {len(vec_processor.db.index)}")
        vec_processor.db.close() # Saves the main index (knowledge + processors)
        print("Veector DB connection closed.")
    if 'vec_processor' in locals():
        del vec_processor
        print("Veector instance deleted.")

    gc.collect()
    if 'torch' in locals() and hasattr(torch, 'cuda') and torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("CUDA cache cleared.")
    print("Garbage collection run.")

except Exception as cell3_e:
    print(f"\n---!!! FATAL ERROR in Cell 3 Execution: {cell3_e} !!!---")
    traceback.print_exc()
    # Ensure cleanup happens even on error
    if 'vec_processor' in locals() and vec_processor and hasattr(vec_processor, 'db') and vec_processor.db:
        try: vec_processor.db.close()
        except: pass
    if 'vec_processor' in locals(): del vec_processor
    gc.collect()
    if 'torch' in locals() and hasattr(torch, 'cuda') and torch.cuda.is_available(): torch.cuda.empty_cache()
    raise # Re-raise the exception

finally:
    end_cell3_time = time.time()
    if processor_errors == 0:
        print(f"\n--- Cell 3 Finished Successfully in {end_cell3_time - start_cell3_time:.2f} seconds ---")
    else:
        print(f"\n--- Cell 3 Finished with {processor_errors} ERRORS in {end_cell3_time - start_cell3_time:.2f} seconds ---")



# === Cell 4: Reference HF Run ===
# Version: 1.1 (Fixed nonlocal error)
# Loads the original Hugging Face model and tokenizer.
# Runs a forward pass with a test prompt to capture intermediate outputs using hooks.
# Saves these outputs to a .pkl file for later comparison.
# Relies only on configuration variables from Cell 1.

import time
import pickle
import numpy as np
import traceback
import os
import gc
from pathlib import Path
from functools import partial
from typing import Dict, List, Any, Optional, Tuple, Union

# --- Imports (Redundant but ensures independence) ---
try:
    import torch
    from torch import nn
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, PreTrainedTokenizer
    print("Torch and Transformers imported successfully.")
except ImportError as e:
    print(f"FATAL ERROR in Cell 4: Missing imports: {e}")
    raise

# --- Configuration (Load from Cell 1 variables) ---
# These should be defined in the global scope by running Cell 1
if 'MODEL_NAME' not in globals(): raise NameError("MODEL_NAME not defined. Run Cell 1.")
if 'HF_MODEL_SOURCE' not in globals(): raise NameError("HF_MODEL_SOURCE not defined. Run Cell 1.")
if 'DB_PATH' not in globals(): raise NameError("DB_PATH not defined. Run Cell 1.")
if 'PROMPT_FOR_TESTING' not in globals(): raise NameError("PROMPT_FOR_TESTING not defined. Run Cell 1.")

print(f"--- Running Cell 4: Reference HF Run for {MODEL_NAME} ---")
print(f"    Using Prompt: '{PROMPT_FOR_TESTING}'")
start_cell4_time = time.time()

# --- Output File ---
# Use DB_PATH defined in Cell 1
output_dir_ref = DB_PATH
output_filename_ref = f"{MODEL_NAME}_hf_reference_outputs_fp32.pkl"
output_filepath_ref = output_dir_ref / output_filename_ref

# --- Initialization ---
tokenizer_ref: Optional[PreTrainedTokenizer] = None
model_ref_fp32 = None
hf_outputs_ref: Dict[str, np.ndarray] = {} # Dictionary to store captured outputs
hook_handles_ref: List[Any] = []
input_ids_torch_ref: Optional[torch.Tensor] = None

# --- Hook Function (Исправлено: убран nonlocal) ---
def get_hook_ref(name: str):
    """Creates a hook to capture the layer's output."""
    def hook_fn(module: nn.Module, input_args: Tuple[Any, ...], output: Any):
        """Captures the layer output and stores it in hf_outputs_ref."""
        # nonlocal hf_outputs_ref # <<< ЭТА СТРОКА УДАЛЕНА >>>
        actual_output: Optional[torch.Tensor] = None
        if isinstance(output, torch.Tensor): actual_output = output
        elif isinstance(output, tuple) and len(output) > 0 and isinstance(output[0], torch.Tensor): actual_output = output[0]
        elif isinstance(output, dict) and 'last_hidden_state' in output and isinstance(output['last_hidden_state'], torch.Tensor): actual_output = output['last_hidden_state']
        elif isinstance(output, tuple) and len(output) > 0 and name.endswith("_attn_out"):
             if isinstance(output[0], torch.Tensor): actual_output = output[0]

        if actual_output is not None:
            # hf_outputs_ref is accessible and modifiable from the outer scope
            hf_outputs_ref[name] = actual_output.detach().cpu().numpy().astype(np.float32)
        else:
            print(f"  [HOOK_REF] WARN: Could not capture tensor output for {name}. Output type: {type(output)}")
    return hook_fn

try:
    # --- 1. Load Tokenizer ---
    print(f"\nLoading Tokenizer from: {HF_MODEL_SOURCE}")
    tokenizer_ref = AutoTokenizer.from_pretrained(HF_MODEL_SOURCE, trust_remote_code=True, use_fast=False)
    print(f"Tokenizer class: {tokenizer_ref.__class__.__name__}")

    # Add special tokens explicitly before getting IDs
    user_token = "<|User|>"
    assistant_token = "<|Assistant|>"
    num_added = tokenizer_ref.add_special_tokens({'additional_special_tokens': [user_token, assistant_token]})
    print(f"Added {num_added} special tokens explicitly.")

    bos_token_id_ref = tokenizer_ref.bos_token_id
    eos_token_id_ref = tokenizer_ref.eos_token_id
    user_token_id_ref = tokenizer_ref.convert_tokens_to_ids(user_token)
    assistant_token_id_ref = tokenizer_ref.convert_tokens_to_ids(assistant_token)

    if isinstance(user_token_id_ref, str) or user_token_id_ref == tokenizer_ref.unk_token_id: raise ValueError("User token ID not found.")
    if isinstance(assistant_token_id_ref, str) or assistant_token_id_ref == tokenizer_ref.unk_token_id: raise ValueError("Assistant token ID not found.")
    if tokenizer_ref.pad_token_id is None: tokenizer_ref.pad_token_id = eos_token_id_ref if eos_token_id_ref is not None else tokenizer_ref.vocab_size
    print(f"Tokens: BOS={bos_token_id_ref}, EOS={eos_token_id_ref}, PAD={tokenizer_ref.pad_token_id}, User={user_token_id_ref}, Assistant={assistant_token_id_ref}")

    # --- 2. Prepare Input ---
    print("\nPreparing Input IDs...")
    user_text_ids_ref = tokenizer_ref.encode(PROMPT_FOR_TESTING, add_special_tokens=False)
    input_ids_list_ref = []
    if bos_token_id_ref is not None: input_ids_list_ref.append(bos_token_id_ref)
    input_ids_list_ref.append(user_token_id_ref)
    input_ids_list_ref.extend(user_text_ids_ref)
    input_ids_list_ref.append(assistant_token_id_ref)
    prompt_input_ids_np_ref = np.array([input_ids_list_ref], dtype=np.int64)
    input_ids_torch_ref = torch.tensor(prompt_input_ids_np_ref)
    print(f"Input IDs shape: {input_ids_torch_ref.shape}")
    print(f"Decoded Input: '{tokenizer_ref.decode(input_ids_list_ref)}'")

    # --- 3. Load Model ---
    print(f"\nLoading HF Model {HF_MODEL_SOURCE} with float32...")
    model_ref_fp32 = AutoModelForCausalLM.from_pretrained(HF_MODEL_SOURCE, torch_dtype=torch.float32, trust_remote_code=True)
    model_ref_fp32.eval()
    device_ref = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device_ref}")
    model_ref_fp32.to(device_ref)
    input_ids_torch_ref = input_ids_torch_ref.to(device_ref)
    print(f"HF Model loaded to device: {model_ref_fp32.device}")

    # --- 4. Register Hooks ---
    print("\nRegistering hooks...")
    model_config_ref = model_ref_fp32.config
    num_layers_ref = model_config_ref.num_hidden_layers
    hook_handles_ref.append(model_ref_fp32.model.embed_tokens.register_forward_hook(get_hook_ref("embed_tokens")))
    for i in range(num_layers_ref):
        layer = model_ref_fp32.model.layers[i]
        hook_handles_ref.append(layer.input_layernorm.register_forward_hook(get_hook_ref(f"L{i}_input_norm_out")))
        hook_handles_ref.append(layer.self_attn.register_forward_hook(get_hook_ref(f"L{i}_attn_out")))
        hook_handles_ref.append(layer.post_attention_layernorm.register_forward_hook(get_hook_ref(f"L{i}_post_attn_norm_out")))
        hook_handles_ref.append(layer.mlp.register_forward_hook(get_hook_ref(f"L{i}_mlp_out")))
        hook_handles_ref.append(layer.register_forward_hook(get_hook_ref(f"L{i}_layer_output")))
    hook_handles_ref.append(model_ref_fp32.model.norm.register_forward_hook(get_hook_ref("final_norm")))
    hook_handles_ref.append(model_ref_fp32.lm_head.register_forward_hook(get_hook_ref("lm_head")))
    print(f"Registered {len(hook_handles_ref)} hooks.")

    # --- 5. Run Forward Pass ---
    print("\nRunning HF model forward pass (float32)...")
    with torch.no_grad():
        hf_model_output = model_ref_fp32(input_ids=input_ids_torch_ref, use_cache=False)
    print("HF forward pass complete.")

    # --- 6. Save Outputs ---
    if hf_outputs_ref:
        print(f"\nSaving Captured Float32 Outputs to {output_filepath_ref}...")
        output_filepath_ref.parent.mkdir(parents=True, exist_ok=True)
        with open(output_filepath_ref, 'wb') as f:
            pickle.dump(hf_outputs_ref, f, pickle.HIGHEST_PROTOCOL)
        print(f"Successfully saved {len(hf_outputs_ref)} reference outputs.")
    else:
        print("\n--- No outputs captured from HF model, skipping save. ---")

except Exception as cell4_e:
    print(f"\n---!!! FATAL ERROR in Cell 4 Execution: {cell4_e} !!!---")
    traceback.print_exc()
finally:
    # --- 7. Cleanup ---
    print("\nCleaning up resources for Cell 4...")
    if hook_handles_ref:
        print(f"Removing {len(hook_handles_ref)} hooks...")
        for handle in hook_handles_ref: handle.remove()
        print("Removed hooks.")
    if model_ref_fp32 is not None:
        del model_ref_fp32
        print("HF model deleted.")
    if 'tokenizer_ref' in locals(): del tokenizer_ref
    if 'input_ids_torch_ref' in locals(): del input_ids_torch_ref

    if 'torch' in locals() and hasattr(torch, 'cuda') and torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("CUDA cache cleared.")
    gc.collect()
    print("Garbage collection run.")

    end_cell4_time = time.time()
    print(f"\n--- Cell 4 Finished in {end_cell4_time - start_cell4_time:.2f} seconds ---")



# === Cell 4.2: Reference HF Run ===
# Version: 1.3 (Added hooks for MLP intermediate outputs)
# Loads the original Hugging Face model and tokenizer.
# Runs a forward pass with a test prompt to capture intermediate outputs using hooks.
# Saves these outputs to a .pkl file for later comparison.
# Relies only on configuration variables from Cell 1.

import time
import pickle
import numpy as np
import traceback
import os
import gc
from pathlib import Path
from functools import partial
from typing import Dict, List, Any, Optional, Tuple, Union

# --- Imports (Redundant but ensures independence) ---
try:
    import torch
    from torch import nn
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, PreTrainedTokenizer
    print("Torch and Transformers imported successfully.")
except ImportError as e:
    print(f"FATAL ERROR in Cell 4: Missing imports: {e}")
    raise SystemExit(f"Missing essential libraries: {e}")

# --- Configuration (Load from Cell 1 variables) ---
# These should be defined in the global scope by running Cell 1 first
if 'MODEL_NAME' not in globals(): raise NameError("MODEL_NAME not defined. Run Cell 1.")
if 'HF_MODEL_SOURCE' not in globals(): raise NameError("HF_MODEL_SOURCE not defined. Run Cell 1.")
if 'DB_PATH' not in globals(): raise NameError("DB_PATH not defined. Run Cell 1.")
if 'PROMPT_FOR_TESTING' not in globals(): raise NameError("PROMPT_FOR_TESTING not defined. Run Cell 1.")

print(f"--- Running Cell 4: Reference HF Run for {MODEL_NAME} ---")
print(f"    Using Prompt: '{PROMPT_FOR_TESTING}'")
start_cell4_time = time.time()

# --- Output File ---
# Define the path where the reference outputs will be saved
output_dir_ref = DB_PATH # Use DB_PATH defined in Cell 1
output_filename_ref = f"{MODEL_NAME}_hf_reference_outputs_fp32.pkl" # File will contain more outputs now
output_filepath_ref = output_dir_ref / output_filename_ref

# --- Initialization ---
tokenizer_ref: Optional[PreTrainedTokenizer] = None
model_ref_fp32 = None
hf_outputs_ref: Dict[str, np.ndarray] = {} # Dictionary to store captured outputs
hook_handles_ref: List[Any] = []
input_ids_torch_ref: Optional[torch.Tensor] = None

# --- Hook Functions ---
# Hook to capture OUTPUT of a module
def get_output_hook_ref(name: str):
    """Creates a hook to capture the module's output."""
    def hook_fn(module: nn.Module, input_args: Tuple[Any, ...], output: Any):
        """Captures the output and stores it in hf_outputs_ref."""
        actual_output: Optional[torch.Tensor] = None
        if isinstance(output, torch.Tensor): actual_output = output
        elif isinstance(output, tuple) and len(output) > 0 and isinstance(output[0], torch.Tensor): actual_output = output[0]
        elif isinstance(output, dict) and 'last_hidden_state' in output and isinstance(output['last_hidden_state'], torch.Tensor): actual_output = output['last_hidden_state']
        elif isinstance(output, tuple) and len(output) > 0 and name.endswith("_attn_out"):
             if isinstance(output[0], torch.Tensor): actual_output = output[0]

        if actual_output is not None:
            hf_outputs_ref[name] = actual_output.detach().cpu().numpy().astype(np.float32)
        else:
            pass # Ignore non-tensor outputs silently
    return hook_fn

# Hook to capture INPUT of a module (using pre-hook)
def get_input_hook_ref(name: str):
    """Creates a pre-hook to capture the module's first input argument."""
    def pre_hook_fn(module: nn.Module, input_args: Tuple[Any, ...]):
        """Captures the first input tensor and stores it."""
        if input_args and isinstance(input_args[0], torch.Tensor):
            hf_outputs_ref[name] = input_args[0].detach().cpu().numpy().astype(np.float32)
        else:
            print(f"  [HOOK_IN_REF] WARN: Could not capture input tensor for {name}. Input type: {type(input_args[0]) if input_args else 'None'}")
    return pre_hook_fn

try:
    # --- 1. Load Tokenizer ---
    print(f"\nLoading Tokenizer from: {HF_MODEL_SOURCE}")
    tokenizer_ref = AutoTokenizer.from_pretrained(HF_MODEL_SOURCE, trust_remote_code=True, use_fast=False)
    print(f"Tokenizer class: {tokenizer_ref.__class__.__name__}")

    user_token = "<|User|>"; assistant_token = "<|Assistant|>"
    num_added = tokenizer_ref.add_special_tokens({'additional_special_tokens': [user_token, assistant_token]})
    print(f"Added {num_added} special tokens explicitly.")

    bos_token_id_ref = tokenizer_ref.bos_token_id
    eos_token_id_ref = tokenizer_ref.eos_token_id
    user_token_id_ref = tokenizer_ref.convert_tokens_to_ids(user_token)
    assistant_token_id_ref = tokenizer_ref.convert_tokens_to_ids(assistant_token)

    if isinstance(user_token_id_ref, str) or user_token_id_ref == tokenizer_ref.unk_token_id: raise ValueError("User token ID not found.")
    if isinstance(assistant_token_id_ref, str) or assistant_token_id_ref == tokenizer_ref.unk_token_id: raise ValueError("Assistant token ID not found.")
    if tokenizer_ref.pad_token_id is None: tokenizer_ref.pad_token_id = eos_token_id_ref if eos_token_id_ref is not None else tokenizer_ref.vocab_size
    print(f"Tokens: BOS={bos_token_id_ref}, EOS={eos_token_id_ref}, PAD={tokenizer_ref.pad_token_id}, User={user_token_id_ref}, Assistant={assistant_token_id_ref}")

    # --- 2. Prepare Input (Using ONNX-style prompt - NO BOS) ---
    print("\nPreparing Input IDs (ONNX-style, no BOS)...")
    user_text_ids_ref = tokenizer_ref.encode(PROMPT_FOR_TESTING, add_special_tokens=False)
    input_ids_list_ref = []
    # if bos_token_id_ref is not None: input_ids_list_ref.append(bos_token_id_ref) # BOS removed
    input_ids_list_ref.append(user_token_id_ref)
    input_ids_list_ref.extend(user_text_ids_ref)
    input_ids_list_ref.append(assistant_token_id_ref)

    prompt_input_ids_np_ref = np.array([input_ids_list_ref], dtype=np.int64)
    input_ids_torch_ref = torch.tensor(prompt_input_ids_np_ref)
    print(f"Input IDs shape: {input_ids_torch_ref.shape}")
    # Print decoded input for verification
    print(f"Decoded Input for Reference Run: '{tokenizer_ref.decode(input_ids_list_ref)}'")

    # --- 3. Load Model ---
    print(f"\nLoading HF Model {HF_MODEL_SOURCE} with float32...")
    model_ref_fp32 = AutoModelForCausalLM.from_pretrained(HF_MODEL_SOURCE, torch_dtype=torch.float32, trust_remote_code=True)
    model_ref_fp32.eval()
    device_ref = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device_ref}")
    model_ref_fp32.to(device_ref)
    input_ids_torch_ref = input_ids_torch_ref.to(device_ref)
    print(f"HF Model loaded to device: {model_ref_fp32.device}")

    # --- 4. Register Hooks (Updated for MLP intermediates) ---
    print("\nRegistering hooks...")
    model_config_ref = model_ref_fp32.config
    num_layers_ref = model_config_ref.num_hidden_layers
    hook_handles_ref.clear() # Clear previous handles

    # Hook on Embedding output
    hook_handles_ref.append(model_ref_fp32.model.embed_tokens.register_forward_hook(get_output_hook_ref("embed_tokens")))

    # Hooks for each layer
    for i in range(num_layers_ref):
        layer = model_ref_fp32.model.layers[i]
        # Attention block related hooks
        hook_handles_ref.append(layer.input_layernorm.register_forward_hook(get_output_hook_ref(f"L{i}_input_norm_out")))
        hook_handles_ref.append(layer.self_attn.register_forward_hook(get_output_hook_ref(f"L{i}_attn_out"))) # Attention output before residual
        hook_handles_ref.append(layer.post_attention_layernorm.register_forward_pre_hook(get_input_hook_ref(f"L{i}_attn_block_output"))) # Input to post_attn_norm = Output after first residual

        # MLP block related hooks
        hook_handles_ref.append(layer.post_attention_layernorm.register_forward_hook(get_output_hook_ref(f"L{i}_post_attn_norm_out"))) # Output of norm before MLP
        # --- MLP Intermediate Hooks ---
        hook_handles_ref.append(layer.mlp.gate_proj.register_forward_hook(get_output_hook_ref(f"L{i}_mlp_gate_out"))) # Output of gate projection
        hook_handles_ref.append(layer.mlp.up_proj.register_forward_hook(get_output_hook_ref(f"L{i}_mlp_up_out"))) # Output of up projection
        hook_handles_ref.append(layer.mlp.down_proj.register_forward_pre_hook(get_input_hook_ref(f"L{i}_mlp_act_mul_up"))) # Input to down_proj = result of act(gate)*up
        hook_handles_ref.append(layer.mlp.down_proj.register_forward_hook(get_output_hook_ref(f"L{i}_mlp_down_out"))) # Output of down_proj (before residual)
        # --- End MLP Intermediate Hooks ---
        hook_handles_ref.append(layer.mlp.register_forward_hook(get_output_hook_ref(f"L{i}_mlp_out"))) # Output of the whole MLP module (usually same as down_out)
        hook_handles_ref.append(layer.register_forward_hook(get_output_hook_ref(f"L{i}_layer_output"))) # Output of the entire layer (after second residual add)

    # Hook on Final Norm output
    hook_handles_ref.append(model_ref_fp32.model.norm.register_forward_hook(get_output_hook_ref("final_norm")))
    # Hook on LM Head output
    hook_handles_ref.append(model_ref_fp32.lm_head.register_forward_hook(get_output_hook_ref("lm_head")))

    print(f"Registered {len(hook_handles_ref)} hooks.")

    # --- 5. Run Forward Pass ---
    print("\nRunning HF model forward pass (float32)...")
    with torch.no_grad():
        hf_model_output = model_ref_fp32(input_ids=input_ids_torch_ref, use_cache=False)
    print("HF forward pass complete.")

    # --- 6. Save Outputs ---
    if hf_outputs_ref:
        print(f"\nSaving Captured Float32 Outputs (including MLP intermediates) to {output_filepath_ref}...")
        output_filepath_ref.parent.mkdir(parents=True, exist_ok=True)
        with open(output_filepath_ref, 'wb') as f:
            pickle.dump(hf_outputs_ref, f, pickle.HIGHEST_PROTOCOL)
        print(f"Successfully saved {len(hf_outputs_ref)} reference outputs.")
        # Print some keys to confirm new outputs are present
        saved_keys = list(hf_outputs_ref.keys())
        print(f"  Example saved keys: {saved_keys[:5]}...{saved_keys[-5:]}")
        # Check for one of the new MLP keys
        if "L0_mlp_gate_out" in saved_keys:
             print("  Confirmed MLP intermediate keys (e.g., 'L0_mlp_gate_out') are present.")
        else:
             print("  WARNING: MLP intermediate keys NOT found in saved outputs!")
    else:
        print("\n--- No outputs captured from HF model, skipping save. ---")

except Exception as cell4_e:
    print(f"\n---!!! FATAL ERROR in Cell 4 Execution: {cell4_e} !!!---")
    traceback.print_exc()
finally:
    # --- 7. Cleanup ---
    print("\nCleaning up resources for Cell 4...")
    if hook_handles_ref:
        print(f"Removing {len(hook_handles_ref)} hooks...")
        for handle in hook_handles_ref: handle.remove()
        print("Removed hooks.")
        hook_handles_ref.clear()
    if model_ref_fp32 is not None:
        del model_ref_fp32
        print("HF model deleted.")
    if 'tokenizer_ref' in locals(): del tokenizer_ref
    if 'input_ids_torch_ref' in locals(): del input_ids_torch_ref
    if 'hf_outputs_ref' in locals(): hf_outputs_ref.clear()

    if 'torch' in locals() and hasattr(torch, 'cuda') and torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("CUDA cache cleared.")
    gc.collect()
    print("Garbage collection run.")

    end_cell4_time = time.time()
    print(f"\n--- Cell 4 Finished in {end_cell4_time - start_cell4_time:.2f} seconds ---")



# === Cell 4.6 (Attention L0 Test Cell) ===
# Version: 1.3 (Fixed KV cache passing for Attn L0 test)
# Loads prerequisites. Initializes Veector. Initializes empty KV cache.
# Runs ONLY Embedding and Layer 0 Attention processor, passing initial cache state.
# Compares the output of the Attention block with the reference value.
# Relies on Cell 1 variables and output files from Cell 3 and Cell 4.

import time
import pickle
import numpy as np
import traceback
import os
import gc
import psutil
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union

# --- Imports ---
try:
    import torch
    from transformers import AutoTokenizer, AutoConfig, PreTrainedTokenizer
    from core import Veector
    from tensors import TensorCoordinate, GROUP_IDX_QWEN_KNOWLEDGE
    from operations import softmax
    from veectordb import VeectorDB
    print("External and Veector libraries imported successfully.")
except ImportError as e:
    print(f"FATAL ERROR in Cell 5: Missing imports: {e}")
    raise

# --- Configuration ---
if 'MODEL_NAME' not in globals(): raise NameError("MODEL_NAME not defined. Run Cell 1.")
if 'HF_MODEL_SOURCE' not in globals(): raise NameError("HF_MODEL_SOURCE not defined. Run Cell 1.")
if 'DB_PATH' not in globals(): raise NameError("DB_PATH not defined. Run Cell 1.")
if 'PROMPT_FOR_TESTING' not in globals(): raise NameError("PROMPT_FOR_TESTING not defined. Run Cell 1.")
if 'KNOWLEDGE_GROUP_IDX' not in globals(): KNOWLEDGE_GROUP_IDX = GROUP_IDX_QWEN_KNOWLEDGE # Fallback

# --- Test Parameters ---
NEST_LEVEL: int = 1
ATOL: float = 1e-2
RTOL: float = 1e-2
# MAX_SEQ_LEN needs to be defined for cache initialization
MAX_SEQ_LEN: Optional[int] = None # Will be loaded from config

print(f"--- Running Cell 5 (Attention L0 Test v1.3) for {MODEL_NAME} ---")
print(f"    DB Path: {DB_PATH.resolve()}")
print(f"    Prompt: '{PROMPT_FOR_TESTING}'")
print(f"    Nest Level: {NEST_LEVEL}")
print(f"    Comparison Tolerances: atol={ATOL}, rtol={RTOL}")
start_cell5_time = time.time()

# --- File Paths ---
proc_map_filepath = DB_PATH / f"{MODEL_NAME}_proc_map.pkl"
ref_output_filepath = DB_PATH / f"{MODEL_NAME}_hf_reference_outputs_fp32.pkl"
main_index_filepath = DB_PATH / VeectorDB.INDEX_FILENAME

# --- Initialization ---
tokenizer_test: Optional[PreTrainedTokenizer] = None
model_config_test = None
processor_map_test: Optional[Dict[str, str]] = None
hf_outputs_test: Optional[Dict[str, np.ndarray]] = None
vec_test: Optional[Veector] = None
error_occurred_test = False
difference_found_test = False
# --- НОВОЕ: Переменные для кеша ---
kv_cache_list_test: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None
num_layers_test: int = 0
num_kv_heads_test: int = 0
head_dim_test: int = 0
max_seq_len_test: int = 2048 # Default fallback

# --- Helper Functions (Без изменений) ---
def log_memory_usage_comp(stage: str):
    try: process = psutil.Process(os.getpid()); mem_info = process.memory_info(); vmem = psutil.virtual_memory(); print(f"  [MEM_LOG_COMP] {stage}: RSS={mem_info.rss / (1024**2):.2f} MB, RAM Used={vmem.percent:.1f}%")
    except Exception as e: print(f"  [MEM_LOG_COMP] Error getting memory usage: {e}")
def sample_top_p_comp(logits: np.ndarray, temperature: float, top_p: float) -> int:
    if np.any(np.isnan(logits)): print("ERROR: NaN detected in logits before sampling! Returning argmax."); return int(np.argmax(logits))
    if temperature < 1e-9: return int(np.argmax(logits))
    logits_f32 = logits.astype(np.float32); scaled_logits = logits_f32 / temperature; probabilities = softmax(scaled_logits)
    if np.any(np.isnan(probabilities)): print("ERROR: NaN detected in probabilities after softmax! Returning argmax."); return int(np.argmax(logits_f32))
    if 0.0 < top_p < 1.0:
        sorted_indices = np.argsort(probabilities)[::-1]; sorted_probabilities = probabilities[sorted_indices]; cumulative_probabilities = np.cumsum(sorted_probabilities); cutoff_index = np.searchsorted(cumulative_probabilities, top_p); cutoff_index = min(cutoff_index, len(sorted_probabilities) - 1); cutoff_prob = sorted_probabilities[cutoff_index]; probabilities[probabilities < cutoff_prob] = 0.0
    prob_sum = np.sum(probabilities)
    if prob_sum > 1e-9: final_probabilities = probabilities / prob_sum
    else: print("Warning: All probabilities became zero after top-p. Using argmax."); return int(np.argmax(logits_f32))
    if np.any(np.isnan(final_probabilities)): print("ERROR: NaN detected in final_probabilities before choice! Using argmax."); return int(np.argmax(logits_f32))
    vocab_size = len(final_probabilities); token_ids = np.arange(vocab_size)
    try: final_probabilities /= final_probabilities.sum(); predicted_token_id = np.random.choice(token_ids, p=final_probabilities)
    except ValueError as e: print(f"ERROR in np.random.choice (Top-P): {e}. Prob sum: {np.sum(final_probabilities)}. Using argmax."); predicted_token_id = np.argmax(logits_f32)
    return int(predicted_token_id)
def log_tensor_stats_comp(name: str, tensor: Optional[np.ndarray], log_values: bool = False):
    if tensor is None: print(f"  [STATS_COMP] {name}: None"); return
    has_nan = np.any(np.isnan(tensor)); shape_str = str(tensor.shape); dtype_str = str(tensor.dtype)
    print(f"  [STATS_COMP] {name}: shape={shape_str}, dtype={dtype_str}, NaN={has_nan}")
    if (has_nan or log_values) and tensor.size > 0 :
        try: sample_slice = tensor.flatten()[:5].tolist(); print(f"               Sample: {sample_slice}")
        except Exception as e: print(f"               Error getting sample: {e}")
def compare_and_log_test(key: str, vec_out: Optional[np.ndarray]) -> bool:
    global difference_found_test
    if difference_found_test or hf_outputs_test is None: return difference_found_test
    print(f"  Comparing: {key}")
    hf_out = hf_outputs_test.get(key)
    if hf_out is None or vec_out is None: print(f"    ERROR: Output missing for comparison (HF: {'OK' if hf_out is not None else 'MISSING'}, Veector: {'OK' if vec_out is not None else 'MISSING'})"); difference_found_test = True; return True
    hf_out_sliced = hf_out; vec_out_sliced = vec_out
    print(f"    HF Shape (fp32): {hf_out_sliced.shape}, dtype: {hf_out_sliced.dtype}"); print(f"    Veector Shape (target): {vec_out_sliced.shape}, dtype: {vec_out_sliced.dtype}")
    if hf_out_sliced.shape != vec_out_sliced.shape: print(f"    ERROR: Shape mismatch for {key}!"); difference_found_test = True; return True
    try:
        hf_out_f32 = hf_out_sliced.astype(np.float32); vec_out_f32 = vec_out_sliced.astype(np.float32)
        are_close = np.allclose(hf_out_f32, vec_out_f32, atol=ATOL, rtol=RTOL)
        print(f"    Result: {'CLOSE' if are_close else '!!! DIFFERENT !!!'}")
        if not are_close:
            diff = np.abs(hf_out_f32 - vec_out_f32); max_diff = np.max(diff); mean_diff = np.mean(diff); max_diff_idx = np.unravel_index(np.argmax(diff), diff.shape)
            print(f"      Max Abs Difference:  {max_diff:.6f} at index {max_diff_idx}"); print(f"      Mean Abs Difference: {mean_diff:.6f}")
            print(f"      HF Sample @ max diff:      {hf_out_f32[max_diff_idx]:.6f}"); print(f"      Veector Sample @ max diff: {vec_out_f32[max_diff_idx]:.6f}")
            difference_found_test = True; return True
    except Exception as cmp_e: print(f"    ERROR during comparison for {key}: {cmp_e}"); difference_found_test = True; return True
    return False

# --- Main Execution Block ---
try:
    log_memory_usage_comp("Start of Cell 5")
    # --- 1. Load Prerequisites ---
    print(f"\nLoading prerequisites...")
    # Load Tokenizer
    print(f"  Loading Tokenizer from: {HF_MODEL_SOURCE}")
    tokenizer_test = AutoTokenizer.from_pretrained(HF_MODEL_SOURCE, trust_remote_code=True, use_fast=False)
    user_token = "<|User|>"; assistant_token = "<|Assistant|>"
    tokenizer_test.add_special_tokens({'additional_special_tokens': [user_token, assistant_token]})
    bos_token_id_test = tokenizer_test.bos_token_id; eos_token_id_test = tokenizer_test.eos_token_id
    user_token_id_test = tokenizer_test.convert_tokens_to_ids(user_token)
    assistant_token_id_test = tokenizer_test.convert_tokens_to_ids(assistant_token)
    if isinstance(user_token_id_test, str) or user_token_id_test == tokenizer_test.unk_token_id: raise ValueError("User token ID not found.")
    if isinstance(assistant_token_id_test, str) or assistant_token_id_test == tokenizer_test.unk_token_id: raise ValueError("Assistant token ID not found.")
    if tokenizer_test.pad_token_id is None: tokenizer_test.pad_token_id = eos_token_id_test if eos_token_id_test is not None else tokenizer_test.vocab_size
    print(f"  Tokenizer loaded.")

    # Load Config
    print(f"  Loading Config from: {HF_MODEL_SOURCE}")
    model_config_test = AutoConfig.from_pretrained(HF_MODEL_SOURCE, trust_remote_code=True)
    # --- ИЗВЛЕКАЕМ ПАРАМЕТРЫ ДЛЯ КЕША ---
    num_layers_test = model_config_test.num_hidden_layers
    num_kv_heads_test = getattr(model_config_test, 'num_key_value_heads', model_config_test.num_attention_heads)
    head_dim_test = model_config_test.hidden_size // model_config_test.num_attention_heads
    if MAX_SEQ_LEN is None: max_seq_len_test = getattr(model_config_test, 'max_position_embeddings', 2048)
    else: max_seq_len_test = MAX_SEQ_LEN
    print(f"  Config loaded. L={num_layers_test}, KVH={num_kv_heads_test}, HDim={head_dim_test}, MaxSeqLen={max_seq_len_test}")

    # Load Processor Map
    if not proc_map_filepath.is_file(): raise FileNotFoundError(f"Processor map file not found: {proc_map_filepath}")
    print(f"  Loading Processor map from: {proc_map_filepath}")
    with open(proc_map_filepath, 'rb') as f: processor_map_test = pickle.load(f)
    print(f"  Processor map loaded ({len(processor_map_test)} entries).")

    # Load Reference Outputs
    if not ref_output_filepath.is_file(): raise FileNotFoundError(f"Reference output file not found: {ref_output_filepath}")
    print(f"  Loading Reference HF outputs from: {ref_output_filepath}")
    with open(ref_output_filepath, 'rb') as f: hf_outputs_test = pickle.load(f)
    if not isinstance(hf_outputs_test, dict): raise TypeError("Reference data is not a dict.")
    print(f"  Reference outputs loaded ({len(hf_outputs_test)} entries).")
    if "L0_attn_block_output" not in hf_outputs_test: raise KeyError("Ref file missing 'L0_attn_block_output'. Re-run Cell 4.")

    # --- 2. Initialize Veector ---
    print(f"\nInitializing Veector instance...")
    vec_test = Veector(db_dir=DB_PATH)
    print(f"Veector initialized. DB Index entries: {len(vec_test.db.index)}")
    if len(vec_test.db.index) == 0: raise RuntimeError("Loaded main index is empty!")

    # --- 3. Check Processors ---
    print("\nChecking processor IDs...")
    if "embedding" not in processor_map_test or "attn_0" not in processor_map_test: raise ValueError("Required processor IDs not found.")
    embedding_processor_id_test = processor_map_test["embedding"]
    attn_0_processor_id_test = processor_map_test["attn_0"]
    print("Required processor IDs found.")

    # --- 4. Prepare Input Data (ONNX Style - No BOS) ---
    print("\nPreparing Input IDs (ONNX-style, no BOS)...")
    prompt_input_ids_np_test: Optional[np.ndarray] = None
    try:
        user_text_ids_test = tokenizer_test.encode(PROMPT_FOR_TESTING, add_special_tokens=False)
        input_ids_list_test = []
        input_ids_list_test.append(user_token_id_test)
        input_ids_list_test.extend(user_text_ids_test)
        input_ids_list_test.append(assistant_token_id_test)
        prompt_input_ids_np_test = np.array([input_ids_list_test], dtype=np.int64)
        print(f"Input IDs shape: {prompt_input_ids_np_test.shape}")
        print(f"Decoded Input: '{tokenizer_test.decode(input_ids_list_test)}'")
    except Exception as e: raise RuntimeError(f"Error constructing prompt tokens: {e}")

    # --- НОВОЕ: Инициализация KV кеша для теста ---
    print(f"\nInitializing KV Cache for testing Attention L0...")
    kv_cache_list_test = []
    cache_dtype = np.float16
    batch_size_test = prompt_input_ids_np_test.shape[0]
    # Используем параметры, загруженные из конфига
    cache_shape = (batch_size_test, num_kv_heads_test, max_seq_len_test, head_dim_test)
    print(f"  Shape per layer: K={cache_shape}, V={cache_shape}, dtype={cache_dtype}")
    # Создаем кеш только для слоя 0, но можем создать и для всех на всякий случай
    # for i in range(num_layers_test):
    k_cache_layer_init = np.zeros(cache_shape, dtype=cache_dtype)
    v_cache_layer_init = np.zeros(cache_shape, dtype=cache_dtype)
    kv_cache_list_test.append((k_cache_layer_init, v_cache_layer_init))
    # Добавим заглушки для остальных слоев, если бы они были нужны
    # for _ in range(1, num_layers_test): kv_cache_list_test.append((None, None))
    print("KV Cache initialized for test.")
    log_memory_usage_comp("After Test KV Cache Init")
    # --- КОНЕЦ НОВОГО ---

    # --- 5. Execute Veector Steps (Embedding + Attn L0 ONLY) ---
    print(f"\n--- Running Veector Execution for L0 Attention Test ---")
    current_hidden_states_test = None
    attn_block_output_test = None
    total_seq_len_test = prompt_input_ids_np_test.shape[1]
    position_ids_test = np.arange(0, total_seq_len_test, dtype=np.int64).reshape(1, total_seq_len_test)

    # Step 5.A: Embedding
    print(f"  Running Embedding...")
    compute_context_embed = {"input_data": prompt_input_ids_np_test, "required_nest": NEST_LEVEL, "target_knowledge_group": KNOWLEDGE_GROUP_IDX}
    embed_result = vec_test.compute(embedding_processor_id_test, context=compute_context_embed)
    if not (embed_result and embed_result.get("status") == "completed"): raise RuntimeError(f"Embedding failed: {embed_result.get('provenance', {}).get('error', 'Unknown error')}")
    current_hidden_states_test = embed_result.get("data")
    if current_hidden_states_test is None: raise RuntimeError(f"Embedding returned None data.")
    print("    Embedding OK.")
    if compare_and_log_test("embed_tokens", current_hidden_states_test): raise RuntimeError("Difference found in Embedding, stopping test.")

    # Step 5.B: Layer 0 Attention Block
    print(f"\n  Running Attention Processor L0...")
    residual_input_test = current_hidden_states_test
    # --- ИЗМЕНЕНО: Передаем инициализированный кеш ---
    initial_k_cache, initial_v_cache = kv_cache_list_test[0] # Берем кеш для слоя 0
    attn_context_test = {
        "input_data": current_hidden_states_test,
        "residual_input": residual_input_test,
        "required_nest": NEST_LEVEL,
        "target_knowledge_group": KNOWLEDGE_GROUP_IDX,
        "position_ids": position_ids_test,
        "total_seq_len": total_seq_len_test,
        "past_key": initial_k_cache, # Передаем пустой массив кеша
        "past_value": initial_v_cache, # Передаем пустой массив кеша
        "start_pos": 0
    }
    # --- КОНЕЦ ИЗМЕНЕНИЯ ---
    attn_result_test = vec_test.compute(attn_0_processor_id_test, context=attn_context_test)
    if not (attn_result_test and attn_result_test.get("status") == "completed"): raise RuntimeError(f"Attn L0 failed: {attn_result_test.get('provenance', {}).get('error', 'Unknown error')}")
    attn_block_output_test = attn_result_test.get("data")
    if attn_block_output_test is None: raise RuntimeError(f"Attn L0 returned None data.")
    print("    Attention L0 OK.")

    # --- 6. Compare Attention Block Output ---
    print("\n--- Comparing L0 Attention Block Output ---")
    compare_and_log_test("L0_attn_block_output", attn_block_output_test)

# --- Error Handling ---
except Exception as cell5_e:
    print(f"\n---!!! ERROR during Attention Test execution: {cell5_e} !!!---")
    traceback.print_exc()
    error_occurred_test = True
finally:
    # --- 7. Cleanup ---
    print("\nCleaning up resources for Cell 5...")
    if vec_test and hasattr(vec_test, 'db') and vec_test.db:
        try: vec_test.db.close(); print("Veector DB connection closed.")
        except Exception as db_close_e: print(f"Error closing DB connection: {db_close_e}")
    if 'vec_test' in locals(): del vec_test
    if 'tokenizer_test' in locals(): del tokenizer_test
    if 'model_config_test' in locals(): del model_config_test
    if 'processor_map_test' in locals(): del processor_map_test
    if 'hf_outputs_test' in locals(): del hf_outputs_test
    if 'kv_cache_list_test' in locals(): del kv_cache_list_test # Очищаем кеш

    gc.collect()
    if 'torch' in locals() and hasattr(torch, 'cuda') and torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("CUDA cache cleared.")
    print("Garbage collection run.")

    # --- Final Verdict ---
    if not error_occurred_test:
        if difference_found_test: print("\n--- RESULT: Difference found in L0 Attention Block output. ---")
        else: print("\n--- RESULT: L0 Attention Block output is CLOSE to reference! ---")
    else: print("\n--- RESULT: Test not completed due to runtime errors. ---")

    end_cell5_time = time.time()
    print(f"\n--- Cell 5 (Attention Test) Finished in {end_cell5_time - start_cell5_time:.2f} seconds ---")



# === Cell 4.8 (FFN L0 Test Cell - Intermediate Comparison) ===
# Version: 1.2
# Loads prerequisites including reference file with MLP intermediate outputs.
# Initializes Veector.
# Loads the REFERENCE Attention block output from the HF run.
# Runs ONLY the Layer 0 FFN processor (which now stores intermediates in context).
# Extracts and Compares intermediate MLP outputs (gate, up, activated, down)
# as well as the final layer output against the reference values.
# Relies on Cell 1 variables and output files from Cell 3 (v1.1+) and Cell 4 (v1.3+).

import time
import pickle
import numpy as np
import traceback
import os
import gc
import psutil
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union

# --- Imports ---
try:
    import torch
    from transformers import AutoTokenizer, AutoConfig, PreTrainedTokenizer
    from core import Veector
    from tensors import TensorCoordinate, GROUP_IDX_QWEN_KNOWLEDGE
    from operations import softmax
    from veectordb import VeectorDB
    print("External and Veector libraries imported successfully.")
except ImportError as e:
    print(f"FATAL ERROR in FFN Test Cell: Missing imports: {e}")
    raise

# --- Configuration ---
if 'MODEL_NAME' not in globals(): raise NameError("MODEL_NAME not defined. Run Cell 1.")
if 'HF_MODEL_SOURCE' not in globals(): raise NameError("HF_MODEL_SOURCE not defined. Run Cell 1.")
if 'DB_PATH' not in globals(): raise NameError("DB_PATH not defined. Run Cell 1.")
if 'KNOWLEDGE_GROUP_IDX' not in globals(): KNOWLEDGE_GROUP_IDX = GROUP_IDX_QWEN_KNOWLEDGE # Fallback

# --- Test Parameters ---
NEST_LEVEL: int = 1 # Precision level for the FFN processor
ATOL: float = 1e-2 # Use increased tolerance
RTOL: float = 1e-2 # Use increased tolerance

print(f"--- Running Cell 5.1 (FFN L0 Intermediate Test v1.1) for {MODEL_NAME} ---")
print(f"    DB Path: {DB_PATH.resolve()}")
print(f"    Nest Level: {NEST_LEVEL}")
print(f"    Comparison Tolerances: atol={ATOL}, rtol={RTOL}")
start_cell_ffn_test_time = time.time()

# --- File Paths ---
proc_map_filepath = DB_PATH / f"{MODEL_NAME}_proc_map.pkl"
# Expecting reference file generated by Cell 4 v1.3+
ref_output_filepath = DB_PATH / f"{MODEL_NAME}_hf_reference_outputs_fp32.pkl"
main_index_filepath = DB_PATH / VeectorDB.INDEX_FILENAME

# --- Initialization ---
processor_map_test_ffn: Optional[Dict[str, str]] = None
hf_outputs_test_ffn: Optional[Dict[str, np.ndarray]] = None
vec_test_ffn: Optional[Veector] = None
error_occurred_test_ffn = False
difference_found_test_ffn = False # Reset flag

# --- Helper Functions ---
def log_memory_usage_ffn(stage: str):
    try: process = psutil.Process(os.getpid()); mem_info = process.memory_info(); vmem = psutil.virtual_memory(); print(f"  [MEM_LOG_FFN] {stage}: RSS={mem_info.rss / (1024**2):.2f} MB, RAM Used={vmem.percent:.1f}%")
    except Exception as e: print(f"  [MEM_LOG_FFN] Error getting memory usage: {e}")
def log_tensor_stats_ffn(name: str, tensor: Optional[np.ndarray], log_values: bool = False):
    if tensor is None: print(f"  [STATS_FFN] {name}: None"); return
    has_nan = np.any(np.isnan(tensor)); shape_str = str(tensor.shape); dtype_str = str(tensor.dtype)
    print(f"  [STATS_FFN] {name}: shape={shape_str}, dtype={dtype_str}, NaN={has_nan}")
    if (has_nan or log_values) and tensor.size > 0 :
        try: sample_slice = tensor.flatten()[:5].tolist(); print(f"               Sample: {sample_slice}")
        except Exception as e: print(f"               Error getting sample: {e}")

def compare_intermediates(step_ctx: Dict, ref_outputs: Dict, layer_idx: int = 0) -> bool:
    """Compares intermediate MLP values stored in step_ctx with reference."""
    global difference_found_test_ffn # Allow modification
    if difference_found_test_ffn: return True # Skip if already different

    print("\n--- Comparing MLP Intermediate Outputs ---")
    diff_found_mlp = False

    # Define keys for comparison
    # Key in step_ctx (Veector) : Key in ref_outputs (HF)
    comparison_keys = {
        f'L{layer_idx}_dbg_mlp_gate_out': f'L{layer_idx}_mlp_gate_out',
        f'L{layer_idx}_dbg_mlp_up_out': f'L{layer_idx}_mlp_up_out',
        f'L{layer_idx}_dbg_mlp_activated': f'L{layer_idx}_mlp_act_mul_up',
        # Also compare the output of MLP before residual add
        # This should be stored in the tuple element 0 -> context 'mlp_final_output_tensor' (or similar)
        # Let's assume Cell 3 stored tuple element 0 as 'mlp_pre_residual_out' for clarity
        # We need to add [OP_LOAD, 'mlp_results_tuple'], [OP_GET_TUPLE_ELEM_0], [OP_STORE, 'mlp_pre_residual_out'] in Cell 3
        'mlp_pre_residual_out': f'L{layer_idx}_mlp_down_out',
    }

    for vec_key, ref_key in comparison_keys.items():
        print(f"  Comparing: Veector '{vec_key}' vs Reference '{ref_key}'")
        vec_val = step_ctx.get(vec_key)
        ref_val = ref_outputs.get(ref_key)

        if ref_val is None or vec_val is None:
            print(f"    ERROR: Output missing for comparison (Ref '{ref_key}': {'OK' if ref_val is not None else 'MISSING'}, Vec '{vec_key}': {'OK' if vec_val is not None else 'MISSING'})")
            diff_found_mlp = True; continue # Continue checking other keys

        print(f"    HF Shape (fp32): {ref_val.shape}, dtype: {ref_val.dtype}")
        print(f"    Veector Shape (target): {vec_val.shape}, dtype: {vec_val.dtype}")

        if ref_val.shape != vec_val.shape:
            print(f"    ERROR: Shape mismatch!")
            diff_found_mlp = True; continue

        try:
            ref_f32 = ref_val.astype(np.float32)
            vec_f32 = vec_val.astype(np.float32)
            are_close = np.allclose(ref_f32, vec_f32, atol=ATOL, rtol=RTOL)
            print(f"    Result: {'CLOSE' if are_close else '!!! DIFFERENT !!!'}")
            if not are_close:
                diff = np.abs(ref_f32 - vec_f32); max_diff = np.max(diff); mean_diff = np.mean(diff); max_diff_idx = np.unravel_index(np.argmax(diff), diff.shape)
                print(f"      Max Abs Difference:  {max_diff:.6f} at index {max_diff_idx}"); print(f"      Mean Abs Difference: {mean_diff:.6f}")
                print(f"      HF Sample ('{ref_key}') @ max diff:      {ref_f32[max_diff_idx]:.6f}"); print(f"      Veector Sample ('{vec_key}') @ max diff: {vec_f32[max_diff_idx]:.6f}")
                diff_found_mlp = True # Found a difference in intermediates
        except Exception as cmp_e:
            print(f"    ERROR during comparison for {vec_key}/{ref_key}: {cmp_e}")
            diff_found_mlp = True

    if diff_found_mlp:
        difference_found_test_ffn = True # Set global flag if any intermediate differs
    return diff_found_mlp

# --- Main Execution Block ---
try:
    log_memory_usage_ffn("Start of FFN Test Cell")
    # --- 1. Load Prerequisites ---
    print(f"\nLoading prerequisites...")
    # Load Processor Map
    if not proc_map_filepath.is_file(): raise FileNotFoundError(f"Processor map file not found: {proc_map_filepath}")
    print(f"  Loading Processor map from: {proc_map_filepath}")
    with open(proc_map_filepath, 'rb') as f: processor_map_test_ffn = pickle.load(f)
    print(f"  Processor map loaded ({len(processor_map_test_ffn)} entries).")

    # Load Reference Outputs
    if not ref_output_filepath.is_file(): raise FileNotFoundError(f"Reference output file not found: {ref_output_filepath}")
    print(f"  Loading Reference HF outputs from: {ref_output_filepath}")
    with open(ref_output_filepath, 'rb') as f: hf_outputs_test_ffn = pickle.load(f)
    if not isinstance(hf_outputs_test_ffn, dict): raise TypeError("Reference data is not a dict.")
    print(f"  Reference outputs loaded ({len(hf_outputs_test_ffn)} entries).")
    # Check if the required keys exist
    required_ref_keys = [
        "L0_attn_block_output", "L0_layer_output",
        "L0_mlp_gate_out", "L0_mlp_up_out",
        "L0_mlp_act_mul_up", "L0_mlp_down_out"
    ]
    for key in required_ref_keys:
        if key not in hf_outputs_test_ffn:
            raise KeyError(f"Reference file missing required key '{key}'. Re-run Cell 4 (v1.3+).")

    # --- 2. Initialize Veector ---
    print(f"\nInitializing Veector instance...")
    vec_test_ffn = Veector(db_dir=DB_PATH)
    print(f"Veector initialized. DB Index entries: {len(vec_test_ffn.db.index)}")
    if len(vec_test_ffn.db.index) == 0: raise RuntimeError("Loaded main index is empty!")

    # --- 3. Check FFN Processor ID ---
    print("\nChecking processor ID...")
    if "ffn_0" not in processor_map_test_ffn: raise ValueError("Processor ID 'ffn_0' not found in map.")
    ffn_0_processor_id_test = processor_map_test_ffn["ffn_0"]
    print("Processor ID 'ffn_0' found.")

    # --- 4. Get Input Data (Reference Attention Output) ---
    print("\nLoading reference 'L0_attn_block_output' as input for FFN...")
    ffn_input_data_np = hf_outputs_test_ffn.get("L0_attn_block_output")
    if ffn_input_data_np is None: raise ValueError("Failed to get 'L0_attn_block_output' from reference data.")
    # Ensure input is in the target precision for the Veector processor
    ffn_input_data_np = ffn_input_data_np.astype(np.float16 if NEST_LEVEL == 1 else np.float32)
    log_tensor_stats_ffn("Input to FFN (Ref Attn Output)", ffn_input_data_np)

    # --- 5. Execute Veector FFN Processor L0 ---
    print(f"\n--- Running Veector FFN Processor L0 ---")
    ffn_output_test = None
    ffn_step_context = None # To store context with intermediates
    ffn_context_test = {
        "input_data": ffn_input_data_np,
        "residual_input": ffn_input_data_np, # Name used by OP_ADD
        "required_nest": NEST_LEVEL,
        "target_knowledge_group": KNOWLEDGE_GROUP_IDX
    }
    ffn_result_test = vec_test_ffn.compute(ffn_0_processor_id_test, context=ffn_context_test)

    # Check execution status
    if not (ffn_result_test and ffn_result_test.get("status") == "completed"):
        raise RuntimeError(f"FFN L0 failed: {ffn_result_test.get('provenance', {}).get('error', 'Unknown error')}")

    # Get final output and step context
    ffn_output_test = ffn_result_test.get("data") # This is the final output of the layer (FFN + Residual)
    ffn_step_context = ffn_result_test.get("step_context") # Context contains stored intermediates

    if ffn_output_test is None: raise RuntimeError(f"FFN L0 returned None data.")
    if ffn_step_context is None: raise RuntimeError(f"FFN L0 did not return step_context.")
    print("    FFN L0 Execution OK.")
    log_tensor_stats_ffn("Final Output of FFN L0 (Veector)", ffn_output_test)

    # --- 6. Compare Intermediate MLP Outputs ---
    # This function compares values stored in ffn_step_context with hf_outputs_test_ffn
    # It will set difference_found_test_ffn if any intermediate differs
    compare_intermediates(step_ctx=ffn_step_context, ref_outputs=hf_outputs_test_ffn, layer_idx=0)

    # --- 7. Compare Final Layer Output ---
    print("\n--- Comparing Final L0 Layer Output (Veector) with Reference L0 Layer Output ---")
    # Compare the final output of the Veector FFN processor (ffn_output_test)
    # with the final output of the reference Layer 0 ("L0_layer_output")
    # This comparison might be redundant if intermediates already differed, but good for confirmation
    compare_and_log_ffn_test(key_ref="L0_layer_output", key_vec="Final FFN L0 Output", vec_out=ffn_output_test)

# --- Error Handling ---
except Exception as cell5_ffn_e:
    print(f"\n---!!! ERROR during FFN Test execution: {cell5_ffn_e} !!!---")
    traceback.print_exc()
    error_occurred_test_ffn = True
finally:
    # --- 8. Cleanup ---
    print("\nCleaning up resources for FFN Test Cell...")
    if vec_test_ffn and hasattr(vec_test_ffn, 'db') and vec_test_ffn.db:
        try: vec_test_ffn.db.close(); print("Veector DB connection closed.")
        except Exception as db_close_e: print(f"Error closing DB connection: {db_close_e}")
    if 'vec_test_ffn' in locals(): del vec_test_ffn
    if 'processor_map_test_ffn' in locals(): del processor_map_test_ffn
    if 'hf_outputs_test_ffn' in locals(): del hf_outputs_test_ffn

    gc.collect()
    if 'torch' in locals() and hasattr(torch, 'cuda') and torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("CUDA cache cleared.")
    print("Garbage collection run.")

    # --- Final Verdict ---
    if not error_occurred_test_ffn:
        if difference_found_test_ffn: print("\n--- RESULT: Difference found in L0 FFN Block (check intermediate comparisons). ---")
        else: print("\n--- RESULT: L0 FFN Block output and intermediates are CLOSE to reference! ---")
    else: print("\n--- RESULT: Test not completed due to runtime errors. ---")

    end_cell_ffn_test_time = time.time()
    print(f"\n--- Cell 5.1 (FFN Intermediate Test) Finished in {end_cell_ffn_test_time - start_cell_ffn_test_time:.2f} seconds ---")



# === Cell 5: Veector Inference & Comparison ===
# Version: 1.1 (Fixed nonlocal error in compare_and_log_comp)
# Loads tokenizer, config, processor map, and reference HF outputs.
# Initializes Veector using the main combined index (tensor_index.pkl).
# Runs autoregressive inference using Veector processors.
# Compares intermediate Veector outputs with the reference HF outputs step-by-step.
# Relies on Cell 1 variables and output files from Cell 3 and Cell 4.

import time
import pickle
import numpy as np
import traceback
import os
import gc
import psutil
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union

# --- Imports (Redundant but ensures independence) ---
try:
    import torch
    from transformers import AutoTokenizer, AutoConfig, PreTrainedTokenizer
    from core import Veector
    from tensors import TensorCoordinate, GROUP_IDX_QWEN_KNOWLEDGE # Import necessary constants
    from operations import softmax # Import necessary ops
    from veectordb import VeectorDB # Import for potential debug/info
    print("External and Veector libraries imported successfully.")
except ImportError as e:
    print(f"FATAL ERROR in Cell 5: Missing imports: {e}")
    raise

# --- Configuration (Load from Cell 1 variables) ---
if 'MODEL_NAME' not in globals(): raise NameError("MODEL_NAME not defined. Run Cell 1.")
if 'HF_MODEL_SOURCE' not in globals(): raise NameError("HF_MODEL_SOURCE not defined. Run Cell 1.")
if 'DB_PATH' not in globals(): raise NameError("DB_PATH not defined. Run Cell 1.")
if 'PROMPT_FOR_TESTING' not in globals(): raise NameError("PROMPT_FOR_TESTING not defined. Run Cell 1.")
# Add other necessary config vars from Cell 1 if needed (e.g., KNOWLEDGE_GROUP_IDX)
if 'KNOWLEDGE_GROUP_IDX' not in globals(): KNOWLEDGE_GROUP_IDX = GROUP_IDX_QWEN_KNOWLEDGE # Fallback

# --- Inference Parameters ---
# These can be overridden or passed as arguments if needed
NEST_LEVEL: int = 1 # Target precision (0=int8, 1=fp16, 2=fp32) - should match processor definition
TEMPERATURE: float = 0.1 # Low temp for more deterministic comparison
TOP_P: float = 0.9
MAX_NEW_TOKENS: int = 10 # Limit for comparison run
MAX_SEQ_LEN: Optional[int] = None # Set specific limit or None for auto from config
USE_KV_CACHE: bool = True
COMPARE_OUTPUTS: bool = False # Enable/disable comparison
ATOL: float = 1e-2 # Absolute tolerance for comparison
RTOL: float = 1e-2 # Relative tolerance for comparison

print(f"--- Running Cell 5: Veector Inference & Comparison for {MODEL_NAME} ---")
print(f"    DB Path: {DB_PATH.resolve()}")
print(f"    Prompt: '{PROMPT_FOR_TESTING}'")
print(f"    Nest Level: {NEST_LEVEL}")
print(f"    Use KV Cache: {USE_KV_CACHE}")
print(f"    Compare Outputs: {COMPARE_OUTPUTS}")
start_cell5_time = time.time()

# --- File Paths ---
proc_map_filepath = DB_PATH / f"{MODEL_NAME}_proc_map.pkl"
ref_output_filepath = DB_PATH / f"{MODEL_NAME}_hf_reference_outputs_fp32.pkl"
main_index_filepath = DB_PATH / VeectorDB.INDEX_FILENAME # Main index

# --- Initialization ---
tokenizer_comp: Optional[PreTrainedTokenizer] = None
model_config_comp = None
processor_map_comp: Optional[Dict[str, str]] = None
hf_outputs_comp: Optional[Dict[str, np.ndarray]] = None
vec_comp: Optional[Veector] = None
num_layers_comp: int = 0
num_kv_heads_comp: int = 0
head_dim_comp: int = 0
eos_token_id_comp: Optional[int] = None
bos_token_id_comp: Optional[int] = None
user_token_id_comp: Optional[int] = None
assistant_token_id_comp: Optional[int] = None
max_seq_len_comp: int = 2048 # Default fallback
error_occurred_comp = False      # Define error flag at cell level
difference_found_comp = False    # Define difference flag at cell level

# --- Helper Functions ---
def log_memory_usage_comp(stage: str):
    try: process = psutil.Process(os.getpid()); mem_info = process.memory_info(); vmem = psutil.virtual_memory(); print(f"  [MEM_LOG_COMP] {stage}: RSS={mem_info.rss / (1024**2):.2f} MB, RAM Used={vmem.percent:.1f}%")
    except Exception as e: print(f"  [MEM_LOG_COMP] Error getting memory usage: {e}")

def sample_top_p_comp(logits: np.ndarray, temperature: float, top_p: float) -> int:
    # (Identical sampling function as used before)
    if np.any(np.isnan(logits)): print("ERROR: NaN detected in logits before sampling! Returning argmax."); return int(np.argmax(logits))
    if temperature < 1e-9: return int(np.argmax(logits))
    logits_f32 = logits.astype(np.float32); scaled_logits = logits_f32 / temperature; probabilities = softmax(scaled_logits)
    if np.any(np.isnan(probabilities)): print("ERROR: NaN detected in probabilities after softmax! Returning argmax."); return int(np.argmax(logits_f32))
    if 0.0 < top_p < 1.0:
        sorted_indices = np.argsort(probabilities)[::-1]; sorted_probabilities = probabilities[sorted_indices]; cumulative_probabilities = np.cumsum(sorted_probabilities); cutoff_index = np.searchsorted(cumulative_probabilities, top_p); cutoff_index = min(cutoff_index, len(sorted_probabilities) - 1); cutoff_prob = sorted_probabilities[cutoff_index]; probabilities[probabilities < cutoff_prob] = 0.0
    prob_sum = np.sum(probabilities)
    if prob_sum > 1e-9: final_probabilities = probabilities / prob_sum
    else: print("Warning: All probabilities became zero after top-p. Using argmax."); return int(np.argmax(logits_f32))
    if np.any(np.isnan(final_probabilities)): print("ERROR: NaN detected in final_probabilities before choice! Using argmax."); return int(np.argmax(logits_f32))
    vocab_size = len(final_probabilities); token_ids = np.arange(vocab_size)
    try: final_probabilities /= final_probabilities.sum(); predicted_token_id = np.random.choice(token_ids, p=final_probabilities)
    except ValueError as e: print(f"ERROR in np.random.choice (Top-P): {e}. Prob sum: {np.sum(final_probabilities)}. Using argmax."); predicted_token_id = np.argmax(logits_f32)
    return int(predicted_token_id)

def log_tensor_stats_comp(name: str, tensor: Optional[np.ndarray], log_values: bool = False):
    if tensor is None: print(f"  [STATS_COMP] {name}: None"); return
    has_nan = np.any(np.isnan(tensor)); shape_str = str(tensor.shape); dtype_str = str(tensor.dtype)
    print(f"  [STATS_COMP] {name}: shape={shape_str}, dtype={dtype_str}, NaN={has_nan}")
    if (has_nan or log_values) and tensor.size > 0 :
        try: sample_slice = tensor.flatten()[:5].tolist(); print(f"               Sample: {sample_slice}")
        except Exception as e: print(f"               Error getting sample: {e}")

# --- Comparison Function (Исправлено: nonlocal -> global) ---
def compare_and_log_comp(key: str, vec_out: Optional[np.ndarray]) -> bool:
    """Compares Veector output with HF reference and logs the result. Returns True if difference found."""
    # Use global to modify the flag defined in the cell's main scope
    global difference_found_comp
    # No need to modify COMPARE_OUTPUTS or hf_outputs_comp from here

    if difference_found_comp or not COMPARE_OUTPUTS or hf_outputs_comp is None:
        return difference_found_comp # Return current state if already different or comparison disabled

    print(f"  Comparing: {key}")
    hf_out = hf_outputs_comp.get(key)
    if hf_out is None or vec_out is None:
        print(f"    ERROR: Output missing for comparison (HF: {'OK' if hf_out is not None else 'MISSING'}, Veector: {'OK' if vec_out is not None else 'MISSING'})")
        difference_found_comp = True
        return True

    # Handle potential sequence length mismatch if comparing step > 0
    current_len_vec = vec_out.shape[1] if vec_out.ndim > 1 else 1
    current_len_hf = hf_out.shape[1] if hf_out.ndim > 1 else 1
    compare_len = current_len_vec # Compare the length generated by Veector in this step

    # Slice HF output only if it's longer than Veector's current output (usually only on step 0)
    if current_len_hf > compare_len:
         hf_out_sliced = hf_out[:, :compare_len, ...]
    else:
         hf_out_sliced = hf_out # Assume lengths match
    vec_out_sliced = vec_out # Veector output is already the correct length for this step

    print(f"    HF Shape (fp32): {hf_out_sliced.shape}, dtype: {hf_out_sliced.dtype}")
    print(f"    Veector Shape (target): {vec_out_sliced.shape}, dtype: {vec_out_sliced.dtype}")

    if hf_out_sliced.shape != vec_out_sliced.shape:
        print(f"    ERROR: Shape mismatch for {key} after slicing!")
        difference_found_comp = True
        return True

    try:
        hf_out_f32 = hf_out_sliced.astype(np.float32) # Ensure HF is float32
        vec_out_f32 = vec_out_sliced.astype(np.float32) # Cast Veector output to float32 for comparison
        are_close = np.allclose(hf_out_f32, vec_out_f32, atol=ATOL, rtol=RTOL)
        print(f"    Result: {'CLOSE' if are_close else '!!! DIFFERENT !!!'}")
        if not are_close:
            diff = np.abs(hf_out_f32 - vec_out_f32)
            max_diff = np.max(diff); mean_diff = np.mean(diff)
            max_diff_idx = np.unravel_index(np.argmax(diff), diff.shape)
            print(f"      Max Abs Difference:  {max_diff:.6f} at index {max_diff_idx}")
            print(f"      Mean Abs Difference: {mean_diff:.6f}")
            print(f"      HF Sample @ max diff:      {hf_out_f32[max_diff_idx]:.6f}")
            print(f"      Veector Sample @ max diff: {vec_out_f32[max_diff_idx]:.6f}")
            difference_found_comp = True # Set the global flag
            return True # Difference found
    except Exception as cmp_e:
        print(f"    ERROR during comparison for {key}: {cmp_e}")
        difference_found_comp = True # Set the global flag
        return True # Difference found due to error
    return False # No difference found for this key

try:
    log_memory_usage_comp("Start of Cell 5")
    # --- 1. Load Prerequisites ---
    print(f"\nLoading prerequisites...")
    # Load Tokenizer
    print(f"  Loading Tokenizer from: {HF_MODEL_SOURCE}")
    tokenizer_comp = AutoTokenizer.from_pretrained(HF_MODEL_SOURCE, trust_remote_code=True, use_fast=False)
    user_token = "<|User|>"; assistant_token = "<|Assistant|>"
    tokenizer_comp.add_special_tokens({'additional_special_tokens': [user_token, assistant_token]})
    bos_token_id_comp = tokenizer_comp.bos_token_id; eos_token_id_comp = tokenizer_comp.eos_token_id
    user_token_id_comp = tokenizer_comp.convert_tokens_to_ids(user_token)
    assistant_token_id_comp = tokenizer_comp.convert_tokens_to_ids(assistant_token)
    if isinstance(user_token_id_comp, str) or user_token_id_comp == tokenizer_comp.unk_token_id: raise ValueError("User token ID not found.")
    if isinstance(assistant_token_id_comp, str) or assistant_token_id_comp == tokenizer_comp.unk_token_id: raise ValueError("Assistant token ID not found.")
    if tokenizer_comp.pad_token_id is None: tokenizer_comp.pad_token_id = eos_token_id_comp if eos_token_id_comp is not None else tokenizer_comp.vocab_size
    print(f"  Tokenizer loaded. User={user_token_id_comp}, Assistant={assistant_token_id_comp}")

    # Load Config
    print(f"  Loading Config from: {HF_MODEL_SOURCE}")
    model_config_comp = AutoConfig.from_pretrained(HF_MODEL_SOURCE, trust_remote_code=True)
    num_layers_comp = model_config_comp.num_hidden_layers
    num_kv_heads_comp = getattr(model_config_comp, 'num_key_value_heads', model_config_comp.num_attention_heads)
    head_dim_comp = model_config_comp.hidden_size // model_config_comp.num_attention_heads
    if MAX_SEQ_LEN is None: max_seq_len_comp = getattr(model_config_comp, 'max_position_embeddings', 2048)
    else: max_seq_len_comp = MAX_SEQ_LEN
    print(f"  Config loaded. L={num_layers_comp}, KVH={num_kv_heads_comp}, HDim={head_dim_comp}, MaxSeqLen={max_seq_len_comp}")

    # Load Processor Map
    if not proc_map_filepath.is_file(): raise FileNotFoundError(f"Processor map file not found: {proc_map_filepath}")
    print(f"  Loading Processor map from: {proc_map_filepath}")
    with open(proc_map_filepath, 'rb') as f: processor_map_comp = pickle.load(f)
    print(f"  Processor map loaded ({len(processor_map_comp)} entries).")

    # Load Reference Outputs (if comparing)
    if COMPARE_OUTPUTS:
        if not ref_output_filepath.is_file():
            print(f"  Warning: Reference output file not found: {ref_output_filepath}. Comparison disabled.")
            COMPARE_OUTPUTS = False
        else:
            print(f"  Loading Reference HF outputs from: {ref_output_filepath}")
            try:
                with open(ref_output_filepath, 'rb') as f: hf_outputs_comp = pickle.load(f)
                if not isinstance(hf_outputs_comp, dict): raise TypeError("Reference data is not a dict.")
                print(f"  Reference outputs loaded ({len(hf_outputs_comp)} entries).")
            except Exception as e:
                print(f"  Warning: Failed to load reference outputs: {e}. Comparison disabled.")
                COMPARE_OUTPUTS = False; hf_outputs_comp = None

    # --- 2. Initialize Veector ---
    print(f"\nInitializing Veector instance...")
    # Initialize using the main DB path, which should load tensor_index.pkl
    vec_comp = Veector(db_dir=DB_PATH)
    print(f"Veector initialized. DB Index entries: {len(vec_comp.db.index)}")
    if len(vec_comp.db.index) == 0:
        raise RuntimeError("Loaded main index (tensor_index.pkl) is empty! Ensure Cell 3 ran correctly.")

    # --- 3. Check Processors from Map ---
    print("\nChecking processor IDs...")
    required_proc_keys = ["embedding", "final_norm", "lm_head"] + [f"attn_{i}" for i in range(num_layers_comp)] + [f"ffn_{i}" for i in range(num_layers_comp)]
    missing_procs = [key for key in required_proc_keys if key not in processor_map_comp]
    if missing_procs: raise ValueError(f"Required processors missing from map: {missing_procs}")
    embedding_processor_id_comp = processor_map_comp["embedding"]
    final_norm_id_comp = processor_map_comp["final_norm"]
    lm_head_id_comp = processor_map_comp["lm_head"]
    print("All required processor IDs found in map.")

    # --- 4. Prepare Input Data ---
    # print("\nPreparing Input IDs...")
    # prompt_input_ids_np_comp: Optional[np.ndarray] = None
    # try:
    #     user_text_ids_comp = tokenizer_comp.encode(PROMPT_FOR_TESTING, add_special_tokens=False)
    #     input_ids_list_comp = []
    #     if bos_token_id_comp is not None: input_ids_list_comp.append(bos_token_id_comp)
    #     input_ids_list_comp.append(user_token_id_comp)
    #     input_ids_list_comp.extend(user_text_ids_comp)
    #     input_ids_list_comp.append(assistant_token_id_comp)
    #     prompt_input_ids_np_comp = np.array([input_ids_list_comp], dtype=np.int64)
    #     print(f"Input IDs shape: {prompt_input_ids_np_comp.shape}")
    #     print(f"Decoded Input: '{tokenizer_comp.decode(input_ids_list_comp)}'")
    # except Exception as e: raise RuntimeError(f"Error constructing prompt tokens: {e}")
        # --- Prepare Input IDs (Формат как в ONNX, без BOS) ---
    print("\nPreparing Input IDs (ONNX-style, no BOS)...")
    prompt_input_ids_np_comp: Optional[np.ndarray] = None
    try:
        user_text_ids_comp = tokenizer_comp.encode(PROMPT_FOR_TESTING, add_special_tokens=False)
        input_ids_list_comp = []
        # --- УБИРАЕМ ДОБАВЛЕНИЕ BOS ---
        # if bos_token_id_comp is not None:
        #     input_ids_list_comp.append(bos_token_id_comp)
        # --- КОНЕЦ ИЗМЕНЕНИЯ ---

        if user_token_id_comp is None or assistant_token_id_comp is None: raise ValueError("User or Assistant token ID is None.")

        input_ids_list_comp.append(user_token_id_comp)
        input_ids_list_comp.extend(user_text_ids_comp)
        input_ids_list_comp.append(assistant_token_id_comp)

        # Пока не добавляем <think>\n, используем чистый ONNX формат
        # if think_token_id is not None: input_ids_list_comp.append(think_token_id)
        # if newline_token_id is not None: input_ids_list_comp.append(newline_token_id)

        prompt_input_ids_np_comp = np.array([input_ids_list_comp], dtype=np.int64)

        print(f"Input IDs shape: {prompt_input_ids_np_comp.shape}")
        print(f"Decoded Input: '{tokenizer_comp.decode(input_ids_list_comp)}'")
    except Exception as e:
        raise RuntimeError(f"Error constructing prompt tokens: {e}")

    # --- 5. Initialize KV Cache (if used) ---
    kv_cache_list_comp: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None
    if USE_KV_CACHE:
        kv_cache_list_comp = []
        cache_dtype = np.float16 # Use float16 for cache
        batch_size_comp = prompt_input_ids_np_comp.shape[0]
        print(f"\nInitializing KV Cache for {num_layers_comp} layers...")
        cache_shape = (batch_size_comp, num_kv_heads_comp, max_seq_len_comp, head_dim_comp)
        print(f"  Shape per layer: K={cache_shape}, V={cache_shape}, dtype={cache_dtype}")
        for i in range(num_layers_comp):
            k_cache_layer = np.zeros(cache_shape, dtype=cache_dtype)
            v_cache_layer = np.zeros(cache_shape, dtype=cache_dtype)
            kv_cache_list_comp.append((k_cache_layer, v_cache_layer))
        print("KV Cache initialized.")
        log_memory_usage_comp("After KV Cache Init")
    else:
        print("\nKV Cache is disabled.")

    # --- 6. Autoregressive Generation & Comparison ---
    print(f"\n--- Starting Autoregressive Generation & Comparison ---")
    generated_ids_comp: List[int] = []
    current_input_ids_for_step_comp: np.ndarray = prompt_input_ids_np_comp
    prompt_len_comp = current_input_ids_for_step_comp.shape[1]
    total_seq_len_comp = prompt_len_comp
    full_response_ids_comp = list(prompt_input_ids_np_comp[0])
    # error_occurred_comp and difference_found_comp defined earlier

    # --- Main Inference Loop ---
    veector_step0_outputs = {} # Store step 0 outputs for comparison
    for step in range(MAX_NEW_TOKENS):
        step_start_time = time.time()
        current_seq_length_comp = current_input_ids_for_step_comp.shape[1]
        start_pos_comp = total_seq_len_comp - current_seq_length_comp
        position_ids_comp = np.arange(start_pos_comp, total_seq_len_comp, dtype=np.int64).reshape(1, current_seq_length_comp)

        if total_seq_len_comp > max_seq_len_comp:
            print(f"\nERROR: total_seq_len ({total_seq_len_comp}) exceeds max_seq_len ({max_seq_len_comp}).")
            break

        print(f"\n--- Step {step + 1}/{MAX_NEW_TOKENS} (Pos: {start_pos_comp}..{total_seq_len_comp-1}) ---")
        log_tensor_stats_comp("Input IDs", current_input_ids_for_step_comp)

        # 1. Embedding
        print(f"  Running Embedding...")
        compute_context_embed = {"input_data": current_input_ids_for_step_comp, "required_nest": NEST_LEVEL, "target_knowledge_group": KNOWLEDGE_GROUP_IDX}
        embed_result = vec_comp.compute(embedding_processor_id_comp, context=compute_context_embed)
        if not (embed_result and embed_result.get("status") == "completed"): raise RuntimeError(f"Embedding failed: {embed_result.get('provenance', {}).get('error', 'Unknown error')}")
        current_hidden_states_comp = embed_result.get("data")
        if current_hidden_states_comp is None: raise RuntimeError(f"Embedding returned None data.")
        print("    Embedding OK.")
        if step == 0: veector_step0_outputs["embed_tokens"] = current_hidden_states_comp
        # Compare only on step 0
        if step == 0 and COMPARE_OUTPUTS and compare_and_log_comp("embed_tokens", current_hidden_states_comp): break

        # 2. Transformer Layers
        residual_input_comp = current_hidden_states_comp
        for layer_idx in range(num_layers_comp):
            # print(f"  Running Layer {layer_idx}...") # Reduce verbosity
            if current_hidden_states_comp is None: raise RuntimeError(f"Input for Layer {layer_idx} is None.")
            attn_proc_id_comp = processor_map_comp[f"attn_{layer_idx}"]
            ffn_proc_id_comp = processor_map_comp[f"ffn_{layer_idx}"]

            # Attention
            attn_context = {"input_data": current_hidden_states_comp, "residual_input": residual_input_comp, "required_nest": NEST_LEVEL, "target_knowledge_group": KNOWLEDGE_GROUP_IDX, "position_ids": position_ids_comp, "total_seq_len": total_seq_len_comp}
            if USE_KV_CACHE and kv_cache_list_comp: attn_context["past_key"], attn_context["past_value"] = kv_cache_list_comp[layer_idx]; attn_context["start_pos"] = start_pos_comp
            attn_result = vec_comp.compute(attn_proc_id_comp, context=attn_context)
            if not (attn_result and attn_result.get("status") == "completed"): raise RuntimeError(f"Attn L{layer_idx} failed: {attn_result.get('provenance', {}).get('error', 'Unknown error')}")
            attn_block_output_comp = attn_result.get("data")
            if attn_block_output_comp is None: raise RuntimeError(f"Attn L{layer_idx} returned None data.")
            if USE_KV_CACHE and kv_cache_list_comp:
                result_step_context_attn = attn_result.get("step_context", {})
                new_k, new_v = result_step_context_attn.get('k_cache_out'), result_step_context_attn.get('v_cache_out')
                if new_k is not None and new_v is not None: kv_cache_list_comp[layer_idx] = (new_k, new_v)

            # FFN
            ffn_input_comp = attn_block_output_comp
            residual_input_ffn_comp = ffn_input_comp
            ffn_context = {"input_data": ffn_input_comp, "residual_input": residual_input_ffn_comp, "required_nest": NEST_LEVEL, "target_knowledge_group": KNOWLEDGE_GROUP_IDX}
            ffn_result = vec_comp.compute(ffn_proc_id_comp, context=ffn_context)
            if not (ffn_result and ffn_result.get("status") == "completed"): raise RuntimeError(f"FFN L{layer_idx} failed: {ffn_result.get('provenance', {}).get('error', 'Unknown error')}")
            layer_output_comp = ffn_result.get("data")
            if layer_output_comp is None: raise RuntimeError(f"FFN L{layer_idx} returned None data.")

            current_hidden_states_comp = layer_output_comp
            residual_input_comp = layer_output_comp # Input for next layer's residual

            # Compare output of the whole layer only on step 0
            if step == 0:
                 veector_step0_outputs[f"L{layer_idx}_layer_output"] = layer_output_comp
                 if COMPARE_OUTPUTS and compare_and_log_comp(f"L{layer_idx}_layer_output", layer_output_comp): break # Break inner layer loop
        if difference_found_comp: break # Break outer generation loop

        # 3. Final Norm
        print("  Running Final Norm...")
        norm_context = {"input_data": current_hidden_states_comp, "required_nest": NEST_LEVEL, "target_knowledge_group": KNOWLEDGE_GROUP_IDX}
        norm_result = vec_comp.compute(final_norm_id_comp, context=norm_context)
        if not (norm_result and norm_result.get("status") == "completed"): raise RuntimeError(f"Final Norm failed: {norm_result.get('provenance', {}).get('error', 'Unknown error')}")
        final_normed_states_comp = norm_result.get("data")
        if final_normed_states_comp is None: raise RuntimeError(f"Final Norm returned None data.")
        print("    Final Norm OK.")
        if step == 0: veector_step0_outputs["final_norm"] = final_normed_states_comp
        if step == 0 and COMPARE_OUTPUTS and compare_and_log_comp("final_norm", final_normed_states_comp): break

        # 4. LM Head
        print("  Running LM Head...")
        last_token_hidden_state_comp = final_normed_states_comp[:, -1:, :] # Select last token's state
        lm_head_context = {"input_data": last_token_hidden_state_comp, "required_nest": NEST_LEVEL, "target_knowledge_group": KNOWLEDGE_GROUP_IDX}
        logits_result = vec_comp.compute(lm_head_id_comp, context=lm_head_context)
        if not (logits_result and logits_result.get("status") == "completed"): raise RuntimeError(f"LM Head failed: {logits_result.get('provenance', {}).get('error', 'Unknown error')}")
        final_logits_comp = logits_result.get("data")
        if final_logits_comp is None: raise RuntimeError(f"LM Head returned None data.")
        print("    LM Head OK.")
        if step == 0: veector_step0_outputs["lm_head"] = final_logits_comp # Logits for the last token of the prompt
        if step == 0 and COMPARE_OUTPUTS:
            hf_lm_head_out = hf_outputs_comp.get("lm_head") if hf_outputs_comp else None
            if hf_lm_head_out is not None:
                hf_last_token_logits = hf_lm_head_out[:, -1:, :]
                vec_logits_to_compare = final_logits_comp
                if compare_and_log_comp("lm_head", vec_logits_to_compare): break
            else: print("    WARN: Reference 'lm_head' output not found for comparison.")

        # 5. Sampling
        print("  Sampling next token...")
        last_token_logits_comp = final_logits_comp[0, -1, :]
        predicted_token_id_comp = sample_top_p_comp(logits=last_token_logits_comp, temperature=TEMPERATURE, top_p=TOP_P)
        print(f"  --> Generated token ID = {predicted_token_id_comp}, Decoded = '{tokenizer_comp.decode([predicted_token_id_comp])}'")

        # 6. Check EOS
        if eos_token_id_comp is not None and predicted_token_id_comp == eos_token_id_comp: print(f"\nEOS token generated. Stopping."); break

        # 7. Prepare for Next Iteration
        generated_ids_comp.append(predicted_token_id_comp)
        full_response_ids_comp.append(predicted_token_id_comp)
        current_input_ids_for_step_comp = np.array([[predicted_token_id_comp]], dtype=np.int64)
        total_seq_len_comp += 1

        current_token_str = tokenizer_comp.decode([predicted_token_id_comp])
        print(current_token_str, end='', flush=True)

        if vec_comp: vec_comp.clear_cache(clear_knowledge=False, clear_compute=True)
        log_memory_usage_comp(f"End of Step {step+1}")

        if total_seq_len_comp >= max_seq_len_comp: print(f"\nMax sequence length reached."); break
    # --- End of Generation Loop ---
    print() # Newline after generation

    # --- 7. Final Output ---
    print("\n--- Final Generated Sequence (Decoded) ---")
    generated_text_comp = tokenizer_comp.decode(generated_ids_comp, skip_special_tokens=True)
    print(f"Generated Text Only: '{generated_text_comp}'")
    full_response_comp = tokenizer_comp.decode(full_response_ids_comp, skip_special_tokens=False)
    print(f"\nFull Response (incl. prompt): '{full_response_comp}'")

except Exception as cell5_e:
    print(f"\n---!!! ERROR during inference/comparison execution: {cell5_e} !!!---")
    traceback.print_exc()
    error_occurred_comp = True
finally:
    # --- 8. Cleanup ---
    print("\nCleaning up resources for Cell 5...")
    if vec_comp and hasattr(vec_comp, 'db') and vec_comp.db:
        try: vec_comp.db.close(); print("Veector DB connection closed.")
        except Exception as db_close_e: print(f"Error closing DB connection: {db_close_e}")
    if 'vec_comp' in locals(): del vec_comp
    if 'tokenizer_comp' in locals(): del tokenizer_comp
    if 'model_config_comp' in locals(): del model_config_comp
    if 'processor_map_comp' in locals(): del processor_map_comp
    if 'hf_outputs_comp' in locals(): del hf_outputs_comp
    if 'kv_cache_list_comp' in locals(): del kv_cache_list_comp

    gc.collect()
    if 'torch' in locals() and hasattr(torch, 'cuda') and torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("CUDA cache cleared.")
    print("Garbage collection run.")

    # --- Final Verdict ---
    if COMPARE_OUTPUTS and not error_occurred_comp:
        if difference_found_comp: print("\n--- RESULT: Differences found during comparison. Stopped at first mismatch. ---")
        else: print("\n--- RESULT: All compared outputs are CLOSE! ---")
    elif error_occurred_comp: print("\n--- RESULT: Comparison not completed due to runtime errors. ---")
    else: print("\n--- RESULT: Comparison was disabled or reference file not found. ---")

    end_cell5_time = time.time()
    print(f"\n--- Cell 5 Finished in {end_cell5_time - start_cell5_time:.2f} seconds ---")


# === Cell 6: Archive & Download ===
# Creates a zip archive of the specified data directory and initiates download.
# Relies on Cell 1 variables (DB_PATH, MODEL_NAME).

import shutil
from google.colab import files # For downloading in Colab
from pathlib import Path

# --- Configuration (Load from Cell 1 variables) ---
if 'DB_PATH' not in globals(): raise NameError("DB_PATH not defined. Run Cell 1.")
if 'MODEL_NAME' not in globals(): raise NameError("MODEL_NAME not defined. Run Cell 1.")

# Define the directory to archive (usually the parent of DB_PATH)
# e.g., if DB_PATH is /content/data/db, archive /content/data
DIR_TO_ARCHIVE = DB_PATH.parent
ARCHIVE_BASENAME = f"veector_data_{MODEL_NAME}" # Base name for the archive file
ARCHIVE_FORMAT = "zip"
ARCHIVE_FILENAME = f"{ARCHIVE_BASENAME}.{ARCHIVE_FORMAT}"

print(f"--- Running Cell 6: Archiving ---")
print(f"    Source directory: {DIR_TO_ARCHIVE}")
print(f"    Archive basename: {ARCHIVE_BASENAME}")
print(f"    Format: {ARCHIVE_FORMAT}")

try:
    # Create the archive
    archive_path = shutil.make_archive(
        base_name=ARCHIVE_BASENAME,
        format=ARCHIVE_FORMAT,
        root_dir=DIR_TO_ARCHIVE.parent, # Start archiving from the parent of DIR_TO_ARCHIVE
        base_dir=DIR_TO_ARCHIVE.name   # The directory name within root_dir to archive
    )
    print(f"Archive created successfully: {archive_path}")

    # Initiate download in Colab
    print(f"\nInitiating download for {ARCHIVE_FILENAME}...")
    files.download(archive_path)
    print("Download initiated. Check your browser.")

except Exception as e:
    print(f"---!!! ERROR during archiving or download: {e} !!!---")
    traceback.print_exc()

print(f"\n--- Cell 6 Finished ---")



# === Cell 7: Upload to Google Drive ===
# Copies the created archive file to a specified Google Drive path.
# Relies on Cell 1 (for Drive mount) and Cell 6 (for archive creation).

import shutil
from google.colab import drive
from pathlib import Path

# --- Configuration (Load from Cell 1/6 variables) ---
# ARCHIVE_FILENAME should be defined based on Cell 6 execution
# Or define it manually if running independently after archive exists
if 'ARCHIVE_FILENAME' not in globals():
    # Try to reconstruct from Cell 1 variables if Cell 6 wasn't run in this session
    if 'MODEL_NAME' in globals():
        ARCHIVE_BASENAME = f"veector_data_{MODEL_NAME}"
        ARCHIVE_FORMAT = "zip"
        ARCHIVE_FILENAME = f"{ARCHIVE_BASENAME}.{ARCHIVE_FORMAT}"
        print(f"WARN: ARCHIVE_FILENAME not found, reconstructed as {ARCHIVE_FILENAME}")
    else:
        raise NameError("ARCHIVE_FILENAME or MODEL_NAME not defined. Run previous cells.")

# Define your target path on Google Drive
# Ensure the 'models' directory (or your desired path) exists in 'My Drive'
GDRIVE_DESTINATION_PATH = "/content/drive/My Drive/veector_models/" # Example path

print(f"--- Running Cell 7: Upload to Google Drive ---")
print(f"    Archive file: {ARCHIVE_FILENAME}")
print(f"    Destination: {GDRIVE_DESTINATION_PATH}")

# Ensure archive file exists locally
local_archive_path = Path(f"./{ARCHIVE_FILENAME}")
if not local_archive_path.is_file():
    raise FileNotFoundError(f"Archive file {local_archive_path} not found. Run Cell 6 first.")

try:
    # Ensure Google Drive is mounted (might need re-authentication)
    drive.mount('/content/drive', force_remount=True) # Force remount might be needed

    # Create destination directory on Drive if it doesn't exist
    dest_path_obj = Path(GDRIVE_DESTINATION_PATH)
    dest_path_obj.mkdir(parents=True, exist_ok=True)
    print(f"Ensured Google Drive directory exists: {dest_path_obj}")

    # Copy the file
    print(f"Copying {local_archive_path} to {dest_path_obj}...")
    shutil.copy(str(local_archive_path), str(dest_path_obj))
    print(f"🟢 [LOG] ✅ Archive uploaded successfully to Google Drive: {dest_path_obj / ARCHIVE_FILENAME}")

except Exception as e:
    print(f"---!!! ERROR during Google Drive upload: {e} !!!---")
    traceback.print_exc()

print(f"\n--- Cell 7 Finished ---")



