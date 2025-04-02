# === Cell 0: Install Dependencies ===
#!pip install numpy psutil torch transformers accelerate bitsandbytes ipfshttpclient qiskit qiskit-aer requests huggingface_hub -q
print("Dependencies installed/checked.")

#upload tensors.py
#upload operations.py
#upload veectordb.py
#upload memory.py
#upload core.py

# === Cell 1: Imports for core.py ===

# --- Standard Imports ---
import numpy as np
import queue # May not be used directly in the final core, but was in original user code
import threading # May not be used directly in the final core, but was in original user code
import time
import random
import psutil
import os
import gc
import pickle
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union
from google.colab import drive, files, userdata
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- Optional Imports (Attempt to import, proceed if fail) ---
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not found. GPU features may be limited.")

try:
    import ipfshttpclient
    IPFS_AVAILABLE = True
except ImportError:
    IPFS_AVAILABLE = False
    # print("Warning: ipfshttpclient not found. IPFS features disabled.")

try:
    from qiskit import QuantumCircuit
    from qiskit.providers.aer import Aer
    from qiskit import execute
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False
    # print("Warning: Qiskit not found. Quantum operations disabled.")


# --- Veector Project Imports ---
# These MUST succeed. Ensure the .py files are uploaded to Colab or pasted in previous cells.
try:
    from core import Veector # <--- ВОТ ОН!
    from tensors import TensorCoordinate, create_tensor # и другие нужные функции/классы из tensors
    # from operations import ... # Если нужны операции напрямую
    # Определения TAG_* констант (можно вынести в отдельный файл или определить здесь)
    TAG_WEIGHTS = 101; TAG_LINEAR = 102; TAG_FLOAT32 = 32; TAG_INT8 = 8 # и т.д.
    PARAM_NAME_WEIGHTS = "weights"
    print("Successfully imported Veector components.")
except ImportError as e:
    print(f"FATAL: Failed to import Veector components needed for this script: {e}")
    raise RuntimeError("Cannot proceed without Veector components") from e

try:
    from veectordb import VeectorDB
    from tensors import (
        TensorCoordinate, create_tensor, validate_tensor, get_tensor_coord,
        get_tensor_op_channels, get_tensor_default_op, get_tensor_filters,
        get_tensor_exit_gates, get_tensor_metadata, get_tensor_parents,
        get_tensor_status, get_tensor_hash, get_tensor_type,
        get_processor_ops_sequence, get_processor_required_knowledge_tags, # Needed by core logic
        get_processor_param_mapping, get_knowledge_compatibility_tags, # Needed by core logic
        get_tensor_input_channels, get_tensor_output_channels # Needed by core logic
    )
    from memory import Memory # Optional memory module

    # --- Explicit imports from operations.py ---
    # Import ALL functions referenced in the core_ops dictionary
    # Use 'pass' in except block if operations.py might be missing some temporarily
    try:
        from operations import (
            mod, floor, ceil, arcsin, arccos, arctan, xor, nand, nor,
            random_uniform, random_normal, median, mean, std_dev, relu,
            sigmoid, softmax, leaky_relu, gelu, exponential_smoothing,
            normalize, interpolate, layer_normalization, batch_norm, dropout,
            matrix_multiply, matrix_determinant, matrix_eigenvalues,
            convolution, transpose, inverse, trace, multi_head_attention
        )
    except ImportError as op_e:
         print(f"Warning: Could not import specific functions from operations.py: {op_e}. Lambdas using them might fail.")
         # Define minimal dummies for missing functions if needed, e.g.:
         # def matrix_multiply(d, **kw): print("WARN: DUMMY matrix_multiply"); return d
         pass # Allow execution to continue, but ops might fail later

except ImportError as core_e:
    print(f"FATAL ERROR: Could not import core Veector components: {core_e}. Cannot proceed.")
    # Define dummy Veector class to prevent NameErrors later if script continues
    class Veector:
        def __init__(self, *args, **kwargs):
            print("FATAL: Veector core components missing.")
            raise RuntimeError("Core components not imported.")


print("Core imports completed (check for warnings/errors).")

# Ячейка для проверки импортов
print("Checking imports...")
try:
    import veectordb
    from veectordb import VeectorDB
    print("-> veectordb OK")
except Exception as e:
    print(f"-> ERROR importing veectordb: {e}")

try:
    import tensors
    from tensors import TensorCoordinate # Проверяем импорт хотя бы одного элемента
    print("-> tensors OK")
except Exception as e:
    print(f"-> ERROR importing tensors: {e}")

try:
    import operations
    from operations import matrix_multiply # Проверяем импорт хотя бы одной операции
    print("-> operations OK")
except Exception as e:
    print(f"-> ERROR importing operations: {e}")

try:
    import memory
    from memory import Memory
    print("-> memory OK")
except Exception as e:
    print(f"-> ERROR importing memory: {e}")
print("Import check finished.")

# --- Configuration ---

# Очистка директории для чистоты эксперимента
#!rm -rf data/
output_dir = "data"
os.makedirs(output_dir, exist_ok=True)

# Аутентификация с Hugging Face
hf_token = userdata.get('HF_TOKEN')
if not hf_token:
    raise ValueError("Добавь HF_TOKEN в секреты Colab!")
login(hf_token)
print("Аутентификация прошла успешно")

# Подключение Google Drive
drive.mount('/content/drive')
print("Google Drive подключён")

HF_MODEL_NAME = "DeepSeek-R1-Distill-Qwen-1.5B"
# Use a unique timestamped directory for this conversion run
DB_DIR_NAME = f"{HF_MODEL_NAME.replace('/', '_')}_veector_db_{time.strftime('%Y%m%d%H%M%S')}"
DB_PATH = Path("./data") / DB_DIR_NAME # Store in ./data subdirectory

# Set data type (bfloat16 might not be fully supported everywhere, float16 is safer)
TORCH_DTYPE = torch.float16 # Use float16 for wider compatibility

print(f"Model to convert: {HF_MODEL_NAME}")
print(f"Target Veector DB: {DB_PATH}")
print(f"Target dtype: {TORCH_DTYPE}")

# === Cell 2: Tag Ontology Definition (Corrected) ===

# Using tuples for potential hierarchy: (Category, SubCategory, Detail)

# Categories
TAG_CAT_TYPE = 0
TAG_CAT_COMPONENT = 1
TAG_CAT_PRECISION = 2
TAG_CAT_MODEL_FAMILY = 3
TAG_CAT_LAYER_IDX = 4 # Use actual layer index directly here
TAG_CAT_FUNCTION = 5 # For processors
TAG_CAT_DATA_SEMANTIC = 6 # Semantic meaning of data
TAG_CAT_USER = 1000 # Range for user-defined tags

# Type Tags (Category 0)
TAG_TYPE_PROCESSOR = (TAG_CAT_TYPE, 1)
TAG_TYPE_KNOWLEDGE = (TAG_CAT_TYPE, 2)
TAG_TYPE_CONVERTER = (TAG_CAT_TYPE, 3)
TAG_TYPE_STATE = (TAG_CAT_TYPE, 4)

# Component Tags (Category 1) - Example for Qwen2 style models
TAG_COMP_WEIGHTS = (TAG_CAT_COMPONENT, 1)
TAG_COMP_BIAS = (TAG_CAT_COMPONENT, 2)
TAG_COMP_EMBEDDING = (TAG_CAT_COMPONENT, 10)
# --- Added Q, K, V constants ---
TAG_COMP_ATTN_Q = (TAG_CAT_COMPONENT, 21) # Separate Q
TAG_COMP_ATTN_K = (TAG_CAT_COMPONENT, 22) # Separate K
TAG_COMP_ATTN_V = (TAG_CAT_COMPONENT, 23) # Separate V
# --- End Added ---
TAG_COMP_ATTN_O = (TAG_CAT_COMPONENT, 24) # Output projection
TAG_COMP_ATTN_QKV = (TAG_CAT_COMPONENT, 25) # Keep for models that DO use fused QKV
TAG_COMP_FFN_GATE = (TAG_CAT_COMPONENT, 30) # gate_proj
TAG_COMP_FFN_UP = (TAG_CAT_COMPONENT, 31)   # up_proj
TAG_COMP_FFN_DOWN = (TAG_CAT_COMPONENT, 32) # down_proj
TAG_COMP_LAYERNORM = (TAG_CAT_COMPONENT, 40)
TAG_COMP_LM_HEAD = (TAG_CAT_COMPONENT, 50)

# Precision Tags (Category 2)
TAG_PREC_FLOAT32 = (TAG_CAT_PRECISION, 32)
TAG_PREC_FLOAT16 = (TAG_CAT_PRECISION, 16)
TAG_PREC_BFLOAT16 = (TAG_CAT_PRECISION, 17)
TAG_PREC_INT8 = (TAG_CAT_PRECISION, 8)
TAG_PREC_INT4 = (TAG_CAT_PRECISION, 4)

# Model Family Tags (Category 3)
TAG_MODEL_QWEN2 = (TAG_CAT_MODEL_FAMILY, 1)
TAG_MODEL_DEEPSEEK = (TAG_CAT_MODEL_FAMILY, 3)

# Layer Index Tags (Category 4) - Use function to generate
def tag_layer(idx: int):
    # Make sure layer index tag is distinct enough
    return (TAG_CAT_LAYER_IDX, idx)

# Function Tags (Category 5) - For Processors
TAG_FUNC_LINEAR = (TAG_CAT_FUNCTION, 1)
TAG_FUNC_ATTENTION = (TAG_CAT_FUNCTION, 2)
TAG_FUNC_FFN = (TAG_CAT_FUNCTION, 3)
TAG_FUNC_EMBED_LOOKUP = (TAG_CAT_FUNCTION, 4)
TAG_FUNC_CAST_DTYPE = (TAG_CAT_FUNCTION, 90)
TAG_FUNC_RESHAPE = (TAG_CAT_FUNCTION, 91)

# Data Semantic Type Tags (Category 6)
TAG_SEMANTIC_HIDDEN_STATE = (TAG_CAT_DATA_SEMANTIC, 1)
TAG_SEMANTIC_LOGITS = (TAG_CAT_DATA_SEMANTIC, 2)
TAG_SEMANTIC_TOKEN_IDS = (TAG_CAT_DATA_SEMANTIC, 3)
TAG_SEMANTIC_KV_CACHE = (TAG_CAT_DATA_SEMANTIC, 4)

print("Tag ontology defined (with Q, K, V tags).")

# === Cell 3: Initialize Veector ===

try:
    # Point to the DB path defined in Cell 1
    vec = Veector(db_dir=DB_PATH, ipfs_enabled=False) # Keep IPFS disabled for local conversion
    print(f"Veector core initialized using DB at: {DB_PATH.resolve()}")
except Exception as e:
    print(f"FATAL: Veector initialization failed: {e}")
    # Stop execution if Veector can't init
    raise RuntimeError("Veector Core failed to initialize") from e

# === Cell 4: Load Hugging Face Model ===

model = None
tokenizer = None
try:
    model = AutoModelForCausalLM.from_pretrained(f"deepseek-ai/{HF_MODEL_NAME}", torch_dtype=TORCH_DTYPE, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(f"deepseek-ai/{HF_MODEL_NAME}", trust_remote_code=True)
    model.eval() # Set to evaluation mode
    print(f"Successfully loaded HF model: {HF_MODEL_NAME}")
    print(f"Model config: {model.config}")
except Exception as e:
    print(f"FATAL: Failed to load HF model '{HF_MODEL_NAME}': {e}")
    # Stop execution
    raise RuntimeError(f"Hugging Face model loading failed") from e

# Clean up GPU memory if possible after loading
if TORCH_AVAILABLE and torch.cuda.is_available():
    torch.cuda.empty_cache()
gc.collect()
print("Model loaded and memory potentially cleaned.")

# === Cell 5: Convert Parameters to Knowledge Tensors ===

param_count = 0
knowledge_map = {} # Store original_name -> knowledge_id for processor mapping
conversion_errors = 0

# Define precision tag based on loaded dtype
precision_tag = TAG_PREC_FLOAT16 # Defaulting to float16
if TORCH_DTYPE == torch.bfloat16:
    precision_tag = TAG_PREC_BFLOAT16
elif TORCH_DTYPE == torch.float32:
    precision_tag = TAG_PREC_FLOAT32

print(f"\n--- Creating Knowledge Tensors (Precision: {precision_tag}) ---")

if model: # Ensure model was loaded
    for name, param in model.named_parameters():
        print(f"Processing: {name} | Shape: {param.shape} | Dtype: {param.dtype}")
        # Ensure data is on CPU for saving via pickle
        param_data = param.data.cpu().clone()

        # --- Determine Tags and Coordinates (Careful Parsing Required!) ---
        tags = [TAG_TYPE_KNOWLEDGE, precision_tag] # Start with type and precision
        layer_idx = -1 # Default for non-layer specific parts
        group_idx = 0 # Default group (e.g., 0 for 'other')
        component_tag = None
        coord_x = 0 # Default x, used to distinguish components within layer/group

        # Determine parameter type (Weight or Bias)
        if name.endswith(".weight"):
            tags.append(TAG_COMP_WEIGHTS)
        elif name.endswith(".bias"):
            tags.append(TAG_COMP_BIAS)

        # Identify Component and Layer based on Qwen2 naming convention
        # This requires precise matching with the output of print(model) or HF model card
        if "model.embed_tokens.weight" in name:
            group_idx = 1 # Embedding group
            component_tag = TAG_COMP_EMBEDDING
            tags.append(TAG_COMP_WEIGHTS)
            coord_x = 0 # Specific coordinate for embedding weights
        elif "model.norm.weight" in name: # Final LayerNorm weight
            group_idx = 4 # Norm group
            layer_idx = model.config.num_hidden_layers # Conceptually after last layer
            component_tag = TAG_COMP_LAYERNORM
            tags.append(TAG_COMP_WEIGHTS)
            coord_x = 0 # Weight for final norm
        elif "model.norm.bias" in name: # Final LayerNorm bias
              group_idx = 4; layer_idx = model.config.num_hidden_layers; component_tag = TAG_COMP_LAYERNORM; tags.append(TAG_COMP_BIAS); coord_x = 1 # Bias for final norm
        elif "lm_head.weight" in name:
            group_idx = 5 # Head group
            component_tag = TAG_COMP_LM_HEAD
            tags.append(TAG_COMP_WEIGHTS)
            coord_x = 0
        elif ".layers." in name:
            try:
                layer_part = name.split('.layers.')[1]
                layer_idx = int(layer_part.split('.')[0])
                tags.append(tag_layer(layer_idx)) # Add specific layer tag

                # Identify component within the layer
                if "self_attn" in name:
                    group_idx = 2 # Attention group
                    # --- ADDED Q, K, V HANDLING ---
                    if "q_proj.weight" in name:
                        component_tag = TAG_COMP_ATTN_Q
                        tags.append(TAG_COMP_WEIGHTS)
                        coord_x = 1 # Assign unique X for Q
                    elif "q_proj.bias" in name:
                        component_tag = TAG_COMP_ATTN_Q
                        tags.append(TAG_COMP_BIAS)
                        coord_x = 11 # Assign unique X for Q Bias
                    elif "k_proj.weight" in name:
                        component_tag = TAG_COMP_ATTN_K
                        tags.append(TAG_COMP_WEIGHTS)
                        coord_x = 2 # Assign unique X for K
                    elif "k_proj.bias" in name:
                        component_tag = TAG_COMP_ATTN_K
                        tags.append(TAG_COMP_BIAS)
                        coord_x = 12 # Assign unique X for K Bias
                    elif "v_proj.weight" in name:
                        component_tag = TAG_COMP_ATTN_V
                        tags.append(TAG_COMP_WEIGHTS)
                        coord_x = 3 # Assign unique X for V
                    elif "v_proj.bias" in name:
                        component_tag = TAG_COMP_ATTN_V
                        tags.append(TAG_COMP_BIAS)
                        coord_x = 13 # Assign unique X for V Bias
                    # --- End ADDED Q, K, V ---
                    elif "o_proj.weight" in name:
                        component_tag = TAG_COMP_ATTN_O
                        tags.append(TAG_COMP_WEIGHTS)
                        coord_x = 4 # Assign unique X for O
                    elif "o_proj.bias" in name: # Check if o_proj has bias
                        component_tag = TAG_COMP_ATTN_O
                        tags.append(TAG_COMP_BIAS)
                        coord_x = 14 # Assign unique X for O Bias
                    else:
                          print(f"  Warning: Unknown self_attn component: {name}")

                elif "mlp" in name:
                    group_idx = 3 # FFN group
                    if "gate_proj.weight" in name:
                        component_tag = TAG_COMP_FFN_GATE
                        tags.append(TAG_COMP_WEIGHTS)
                        coord_x = 5 # Assign unique X
                    elif "up_proj.weight" in name:
                        component_tag = TAG_COMP_FFN_UP
                        tags.append(TAG_COMP_WEIGHTS)
                        coord_x = 6 # Assign unique X
                    elif "down_proj.weight" in name:
                        component_tag = TAG_COMP_FFN_DOWN
                        tags.append(TAG_COMP_WEIGHTS)
                        coord_x = 7 # Assign unique X
                    # Add bias checks for MLP if they exist in Qwen2
                    # elif "gate_proj.bias" in name: ...
                    else:
                          print(f"  Warning: Unknown mlp component: {name}")

                elif "input_layernorm.weight" in name:
                    component_tag = TAG_COMP_LAYERNORM; group_idx=4; tags.append(TAG_COMP_WEIGHTS); coord_x=10
                elif "input_layernorm.bias" in name: # Qwen2 uses RMSNorm, might not have bias? Check! If not, remove this elif.
                    component_tag = TAG_COMP_LAYERNORM; group_idx=4; tags.append(TAG_COMP_BIAS); coord_x=11
                elif "post_attention_layernorm.weight" in name:
                    component_tag = TAG_COMP_LAYERNORM; group_idx=4; tags.append(TAG_COMP_WEIGHTS); coord_x=12
                elif "post_attention_layernorm.bias" in name: # Qwen2 uses RMSNorm, might not have bias? Check! If not, remove this elif.
                    component_tag = TAG_COMP_LAYERNORM; group_idx=4; tags.append(TAG_COMP_BIAS); coord_x=13
                else:
                      print(f"  Warning: Unknown component in layer {layer_idx}: {name}")

            except Exception as parse_e:
                print(f"  Error parsing layer/component for {name}: {parse_e}")
                continue # Skip this parameter if parsing fails

        # Add component and precision tags if identified
        if component_tag:
            tags.append(component_tag)
        if precision_tag:
            tags.append(precision_tag)

        # Ensure all tags are hashable (use tuples)
        final_tags = [tuple(t) if isinstance(t, list) else t for t in tags]

        # --- Create Coordinate ---
        knowledge_coord = TensorCoordinate(layer=layer_idx, group=group_idx, nest=1, x=coord_x) # nest=1 for bf16/fp16

        # --- Create and Save Knowledge Tensor ---
        knowledge_tensor = vec.create_tensor(
            coord=knowledge_coord,
            tensor_type="knowledge",
            knowledge_data=param_data,
            compatibility_tags=final_tags, # Use specific field
            tags=final_tags, # Also store in general tags? Optional.
            metadata={"original_name": name}
        )
        knowledge_id = vec.save_tensor(knowledge_tensor)

        if knowledge_id:
            print(f"  Saved: {name} -> {knowledge_id} (Coord: {knowledge_coord}, Tags: {final_tags})")
            knowledge_map[name] = knowledge_id # Store mapping
            param_count += 1
        else:
            print(f"  ERROR saving knowledge tensor for: {name}")
            conversion_errors += 1

    print(f"\n--- Finished saving {param_count} knowledge tensors ---")
    if conversion_errors > 0:
         print(f"!!! WARNING: Encountered {conversion_errors} errors during knowledge tensor saving !!!")

    # Optional: Save the mapping for later use
    map_file = DB_PATH / f"{HF_MODEL_NAME}_param_map.pkl"
    try:
        with open(map_file, 'wb') as f:
            pickle.dump(knowledge_map, f)
        print(f"Parameter name to Knowledge ID map saved to {map_file}")
    except Exception as e:
        print(f"Error saving parameter map: {e}")

else:
    print("Model was not loaded successfully. Skipping parameter conversion.")

# Clean up model from memory
del model
if TORCH_AVAILABLE and torch.cuda.is_available():
    torch.cuda.empty_cache()
gc.collect()
print("Cleaned up HF model from memory.")

# === Cell 6: Define Processor Tensors (Example/Placeholder) ===

# This part is complex and requires detailed knowledge of the architecture.
# We will define ONE example processor for the embedding layer.
# Defining processors for Attention and FFN requires careful op_sequence design.

print("\n--- Defining Processor Tensors (Example: Embedding) ---")

processor_errors = 0
embedding_processor_id = None

# --- Embedding Processor ---
embed_proc_coord = TensorCoordinate(layer=-1, group=1, nest=1, x=0) # Layer -1, Group 1 (Embed), Nest 1 (bf16/fp16)
embed_knowledge_tag = TAG_COMP_EMBEDDING # Tag used to find the embedding matrix

# Find the knowledge ID for the embedding matrix using the map saved earlier
# OR query the DB using tags
embedding_knowledge_id = None
# Method 1: Using saved map
map_file = DB_PATH / f"{HF_MODEL_NAME}_param_map.pkl"
if map_file.exists():
    with open(map_file, 'rb') as f:
        knowledge_map = pickle.load(f)
    # Find the key corresponding to embedding weights
    for name, kid in knowledge_map.items():
         if "embed_tokens.weight" in name:
              embedding_knowledge_id = kid
              break
else:
     # Method 2: Query DB (Less reliable if multiple embeddings exist)
     found = vec.db.find_active_tensors(tensor_type="knowledge", tags=[TAG_COMP_EMBEDDING, TAG_COMP_WEIGHTS])
     if found: embedding_knowledge_id = list(found.keys())[0]

if embedding_knowledge_id:
    print(f"Found Embedding Knowledge ID: {embedding_knowledge_id}")
    # Define OP code for embedding lookup (assuming it exists in core_ops)
    OP_EMBEDDING_LOOKUP = [40, 6, 0] # Placeholder operation code - DEFINE IT

    embed_proc = vec.create_tensor(
        coord=embed_proc_coord,
        tensor_type="processor",
        tags=[TAG_TYPE_PROCESSOR, TAG_FUNC_EMBED_LOOKUP], # Function tag
        ops_sequence=[OP_EMBEDDING_LOOKUP], # Single operation
        required_knowledge_tags=[TAG_COMP_EMBEDDING], # Tag required
        # Map the *specific knowledge ID* found to the parameter name the op expects
        param_mapping={embedding_knowledge_id: "embedding_matrix"},
        interface={
            "inputs": [{"name":"token_ids", "tags":[TAG_SEMANTIC_TOKEN_IDS], "dtype":"int64"}],
            "outputs": [{"name":"hidden_states", "tags":[TAG_SEMANTIC_HIDDEN_STATE], "dtype":str(TORCH_DTYPE).split('.')[-1]}],
            "knowledge_needed": [{"param_name": "embedding_matrix", "tags":[TAG_COMP_EMBEDDING]}]
        },
        metadata={"description": f"{HF_MODEL_NAME} Embedding Lookup Processor"}
    )
    embedding_processor_id = vec.save_tensor(embed_proc)
    if embedding_processor_id:
        print(f"Saved Embedding Processor: {embedding_processor_id}")
    else:
        print(f"ERROR saving Embedding Processor")
        processor_errors += 1
else:
    print("ERROR: Could not find Embedding Knowledge ID. Skipping Embedding Processor.")
    processor_errors += 1

# --- Placeholder for Attention Processor (Layer 0) ---
print("\n--- Defining Processor Tensor (Layer 0 Attention - Placeholder) ---")
# TODO: Define ops_sequence, required_tags, param_mapping for Attention
# proc_coord_attn0 = TensorCoordinate(layer=0, group=2, nest=1, x=1)
# Find relevant knowledge IDs (QKV, O, Norms for layer 0) from knowledge_map
# attn0_proc = vec.create_tensor(...)
# attn0_proc_id = vec.save_tensor(attn0_proc)
# print(f"Saved Attention Processor L0: {attn0_proc_id}")

# --- Placeholder for FFN Processor (Layer 0) ---
print("\n--- Defining Processor Tensor (Layer 0 FFN - Placeholder) ---")
# TODO: Define ops_sequence, required_tags, param_mapping for FFN
# proc_coord_ffn0 = TensorCoordinate(layer=0, group=3, nest=1, x=3)
# Find relevant knowledge IDs (Gate, Up, Down, Norms for layer 0) from knowledge_map
# ffn0_proc = vec.create_tensor(...)
# ffn0_proc_id = vec.save_tensor(ffn0_proc)
# print(f"Saved FFN Processor L0: {ffn0_proc_id}")

# --- Placeholder for LM Head Processor ---
print("\n--- Defining Processor Tensor (LM Head - Placeholder) ---")
# TODO: Define ops_sequence, required_tags, param_mapping for LM Head

if processor_errors > 0:
    print(f"!!! WARNING: Encountered {processor_errors} errors during processor tensor creation !!!")
else:
    print("Example processor tensor definition completed (Embed only).")

print("\n--- Processor Definition Finished (Placeholders need implementation) ---")

# === Cell 7: Inference Test (Placeholder) ===

print("\n--- Inference Test Placeholder ---")

if 'embedding_processor_id' in locals() and embedding_processor_id:
    print(f"Using Embedding Processor: {embedding_processor_id}")

    # 1. Prepare Input
    text = "Hello Veector!"
    # Needs tokenizer loaded in Cell 4
    if tokenizer:
        input_ids = tokenizer.encode(text, return_tensors="np") # Get numpy array
        print(f"Input Text: '{text}'")
        print(f"Input IDs: {input_ids}")

        # 2. Run Embedding Processor
        compute_context = {
            "input_data": input_ids,
            "required_nest": 1 # Request bf16/fp16 knowledge
        }
        try:
            embedding_result = vec.compute(embedding_processor_id, context=compute_context)

            if embedding_result and embedding_result.get("status") == "completed":
                 hidden_states = embedding_result.get("data")
                 print(f"Embedding Output Shape: {getattr(hidden_states, 'shape', 'N/A')}")
                 # TODO: Continue inference by calling Layer 0 Attention Processor, etc.
                 # next_processor_id = find_processor_for_layer(0, TAG_FUNC_ATTENTION)
                 # attn_context = {"input_data": hidden_states, "required_nest": 1, ...}
                 # attn_result = vec.compute(next_processor_id, context=attn_context)
                 # ... and so on for all layers ...
                 print("Inference successful up to embedding layer.")
            else:
                 print("Embedding computation failed or did not complete.")
                 print(f"Result: {embedding_result}")

        except Exception as e:
            print(f"Error during inference test: {e}")
            import traceback
            traceback.print_exc()

    else:
        print("Tokenizer not loaded, cannot run inference test.")
else:
    print("Embedding processor was not created. Cannot run inference test.")


print("\n--- Conversion and Basic Test Finished ---")

#---------------------------------
# Архивация и скачивание
import shutil
shutil.make_archive("model_DeepSeek-r1-distill-1.5b", "zip", "data")
zip_name = "model_DeepSeek-r1-distill-1.5b.zip"

#------------------------------------
# Выгрузка на Google Drive
drive.mount('/content/drive', force_remount=True)
destination_path = f"/content/drive/My Drive/models/"
shutil.copy(zip_name, destination_path)
print(f"🟢 [LOG] ✅ Архив загружен на Google Drive: {destination_path}")