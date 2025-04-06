# FILE: qwen_inference.py
# Version: 0.2.42 (Ruchnoe formirovanie prompta v stile GGUF shablona)

import argparse
import time
import pickle
import numpy as np
import traceback
import os
import gc
import psutil
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union

# --- Version ---
# IZMENENO: Obnovlena versija
QWEN_INFERENCE_VERSION = "0.2.42" # Ruchnoj prompt (stil' GGUF)

# --- Inference & Sampling Parameters ---
TEMPERATURE = 0.6
TOP_P = 0.95
MAX_NEW_TOKENS = 20
MAX_SEQ_LEN = 2048

# --- Neobhodimye importy iz proekta ---
PROJECT_IMPORTS_OK = False
CORE_VERSION_REQ = "0.7.3"
TENSORS_VERSION_REQ = "0.7.6"
VEECTORDB_VERSION_REQ = "0.9.7"
OPERATIONS_VERSION_REQ = "0.8.9"

try:
    from core import Veector, CORE_VERSION
    print(f"  Imported Core (v{CORE_VERSION})")
    core_v_parts = list(map(int, CORE_VERSION.split('.')))
    req_core_v_parts = list(map(int, CORE_VERSION_REQ.split('.')))
    if core_v_parts < req_core_v_parts:
         raise ImportError(f"qwen_inference.py requires core v{CORE_VERSION_REQ}+, found v{CORE_VERSION}")

    from tensors import TensorCoordinate, TENSORS_VERSION, GROUP_IDX_QWEN_KNOWLEDGE
    print(f"  Imported Tensors (v{TENSORS_VERSION})")
    tensors_v_parts = list(map(int, TENSORS_VERSION.split('.')))
    req_tensors_v_parts = list(map(int, TENSORS_VERSION_REQ.split('.')))
    if tensors_v_parts < req_tensors_v_parts:
         raise ImportError(f"qwen_inference.py requires tensors v{TENSORS_VERSION_REQ}+, found v{TENSORS_VERSION}")

    from operations import OPERATIONS_VERSION, softmax
    print(f"  Imported operations (v{OPERATIONS_VERSION})")
    ops_v_parts = list(map(int, OPERATIONS_VERSION.split('.')))
    req_ops_v_parts_strict = [0, 8, 9]
    if ops_v_parts < req_ops_v_parts_strict:
        print(f"WARN: operations.py version is {OPERATIONS_VERSION}, but v0.8.9+ with SDPA fix is recommended.")
    elif ops_v_parts < list(map(int, OPERATIONS_VERSION_REQ.split('.'))):
         raise ImportError(f"qwen_inference.py requires operations v{OPERATIONS_VERSION_REQ}+, found v{OPERATIONS_VERSION}")


    from transformers import AutoTokenizer, PreTrainedTokenizer, AutoConfig
    print("Project components imported successfully.")
    PROJECT_IMPORTS_OK = True

except ImportError as e:
    print(f"---!!! FATAL ERROR (ImportError in qwen_inference.py) !!! ---")
    print(f"Specific error: {e}")
    print(f"Ensure files (core v{CORE_VERSION_REQ}+, tensors v{TENSORS_VERSION_REQ}+, operations v{OPERATIONS_VERSION_REQ}+) are OK.")
    print(f"-----------------------------------------")
    PROJECT_IMPORTS_OK = False
except NameError as ne:
    print(f"---!!! FATAL ERROR (NameError in qwen_inference.py) !!! ---")
    print(f"Specific error: {ne}")
    print(f"-----------------------------------------")
    PROJECT_IMPORTS_OK = False
except Exception as import_e:
    print(f"---!!! FATAL ERROR (Other Exception during Import in qwen_inference.py) !!! ---")
    print(f"Specific error: {import_e}")
    traceback.print_exc()
    print(f"----------------------------------------------------------")
    PROJECT_IMPORTS_OK = False

# --- Vspomogatel'naja funkcija logirovanija pamjati ---
def log_memory_usage(stage: str):
    """Logiruet tekushhee ispol'zovanie RAM."""
    try:
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        vmem = psutil.virtual_memory()
        print(f"  [MEM_LOG] {stage}: RSS={mem_info.rss / (1024**2):.2f} MB, RAM Used={vmem.percent:.1f}%")
    except Exception as e:
        print(f"  [MEM_LOG] Error getting memory usage: {e}")

# --- Funkcii Semplirovanija ---
def sample_top_p(logits: np.ndarray, temperature: float, top_p: float) -> int:
    """Primenjaet temperature scaling i top-p sampling."""
    if np.any(np.isnan(logits)):
        print("ERROR: NaN detected in logits before sampling! Returning argmax.")
        return int(np.argmax(logits))

    if temperature < 1e-9:
        return int(np.argmax(logits))

    logits_f32 = logits.astype(np.float32)
    scaled_logits = logits_f32 / temperature
    probabilities = softmax(scaled_logits)

    if np.any(np.isnan(probabilities)):
        print("ERROR: NaN detected in probabilities after softmax! Returning argmax.")
        return int(np.argmax(logits_f32))

    if 0.0 < top_p < 1.0:
        sorted_indices = np.argsort(probabilities)[::-1]
        sorted_probabilities = probabilities[sorted_indices]
        cumulative_probabilities = np.cumsum(sorted_probabilities)
        cutoff_index = np.searchsorted(cumulative_probabilities, top_p)
        cutoff_index = min(cutoff_index, len(sorted_probabilities) - 1)
        cutoff_prob = sorted_probabilities[cutoff_index]
        probabilities[probabilities < cutoff_prob] = 0.0

    prob_sum = np.sum(probabilities)
    if prob_sum > 1e-9:
        final_probabilities = probabilities / prob_sum
    else:
        print("Warning: All probabilities became zero after top-p. Using argmax.")
        return int(np.argmax(logits_f32))

    if np.any(np.isnan(final_probabilities)):
        print("ERROR: NaN detected in final_probabilities before choice! Using argmax.")
        return int(np.argmax(logits_f32))

    vocab_size = len(final_probabilities)
    token_ids = np.arange(vocab_size)
    try:
        final_probabilities /= final_probabilities.sum()
        predicted_token_id = np.random.choice(token_ids, p=final_probabilities)
    except ValueError as e:
        print(f"ERROR in np.random.choice (Top-P): {e}. Prob sum: {np.sum(final_probabilities)}. Using argmax.")
        predicted_token_id = np.argmax(logits_f32)
    return int(predicted_token_id)

# --- Vspomogatel'naja funkcija dlja logirovanija tenzora ---
def log_tensor_stats(name: str, tensor: Optional[np.ndarray], log_values: bool = False):
    """Logiruet formu, tip, nalichie NaN i primernye znachenija tenzora."""
    if tensor is None:
        print(f"  [STATS] {name}: None")
        return
    has_nan = np.any(np.isnan(tensor))
    shape_str = str(tensor.shape)
    dtype_str = str(tensor.dtype)
    print(f"  [STATS] {name}: shape={shape_str}, dtype={dtype_str}, NaN={has_nan}")
    if has_nan or log_values:
        try:
            sample_slice = tensor.flatten()[:5]
            print(f"          Sample: {sample_slice}")
        except Exception as e:
            print(f"          Error getting sample: {e}")

# --- Osnovnaja funkcija inferensa ---
def run_qwen_inference(
    text: str,
    db_path: Union[str, Path],
    model_name_hf: str, # Teper' eto mozhet byt' lokal'nyj put' ili identifikator HF
    nest_level: int = 1,
    temperature: float = TEMPERATURE,
    top_p: float = TOP_P,
    max_new_tokens: int = MAX_NEW_TOKENS,
    max_seq_len: int = MAX_SEQ_LEN
    ):
    """Zapuskaet avtoregressivnyj inferens s ispol'zovaniem Veector i KV-kjesha."""
    print(f"\n--- Running Inference Script v{QWEN_INFERENCE_VERSION} ---")
    if not PROJECT_IMPORTS_OK:
        print("Cannot run: failed project imports.")
        return

    log_memory_usage("Start of inference function")

    # --- Opredeljaem put' k lokal'nym fajlam tokenizatora ---
    # Ispol'zuem put', ukazannyj pol'zovatelem (s ispravlennoj opečatkoj "Distill")
    script_dir = Path(__file__).parent.resolve() if "__file__" in locals() else Path.cwd()
    # Predpolagaem, chto papka s tokenizatorom nazyvaetsja pravil'no
    local_tokenizer_path = script_dir / "../data/db/DeepSeek-R1-Distill-Qwen-1.5B"
    print(f"Expected local tokenizer path: {local_tokenizer_path.resolve()}")

    # --- Zagruzka konfiga modeli dlja parametrov ---
    hf_config = None
    num_layers = 0; num_heads = 0; num_kv_heads = 0; hidden_size = 0; head_dim = 0
    config_source = model_name_hf # Put' ili ID dlja zagruzki konfiga
    # Esli model_name_hf eto lokal'nyj put', probuem zagruzit' konfig ottuda
    if Path(model_name_hf).is_dir():
        config_source = model_name_hf
        print(f"Attempting to load config from local path: {config_source}")
    else:
        # Inache probuem zagruzit' po identifikatoru HF
        config_identifier = f"deepseek-ai/{model_name_hf.split('/')[-1]}"
        config_source = config_identifier
        print(f"Attempting to load config from HF identifier: {config_source}")

    try:
        hf_config = AutoConfig.from_pretrained(config_source, trust_remote_code=True)
        num_layers = hf_config.num_hidden_layers
        num_heads = hf_config.num_attention_heads
        num_kv_heads = getattr(hf_config, 'num_key_value_heads', num_heads)
        hidden_size = hf_config.hidden_size
        if num_heads > 0: head_dim = hidden_size // num_heads
        else: raise ValueError("num_attention_heads is zero in model config.")
        print(f"Model Config Loaded: L={num_layers}, H={num_heads}, KVH={num_kv_heads}, HDim={head_dim}, Hidden={hidden_size}")
        if hidden_size % num_heads != 0: print(f"WARN: hidden_size {hidden_size} not perfectly divisible by num_attention_heads {num_heads}")
    except Exception as e:
        print(f"WARN: Failed to load model config from '{config_source}': {e}.")
        print("WARN: Using hardcoded model parameters as fallback!")
        num_layers = 28
        num_heads = 12
        num_kv_heads = 2
        hidden_size = 1536
        head_dim = 128
        print(f"Using Fallback Config: L={num_layers}, H={num_heads}, KVH={num_kv_heads}, HDim={head_dim}, Hidden={hidden_size}")

    # --- Logirovanie parametrov ---
    print(f"Original Input Text: '{text}'")
    db_path = Path(db_path)
    print(f"DB Path: {db_path.resolve()}")
    print(f"Model Source Used: {model_name_hf}")
    print(f"Target Nest Level: {nest_level}")
    print(f"Sampling Params: Temp={temperature}, TopP={top_p}")
    print(f"Max New Tokens: {max_new_tokens}")
    print(f"Max Seq Len (for Cache): {max_seq_len}")
    if not db_path.is_dir(): print(f"ERROR: DB directory not found: {db_path}"); return

    # --- Zagruzka Tokenizatora (Lokal'no) ---
    tokenizer: Optional[PreTrainedTokenizer] = None
    eos_token_id: Optional[int] = None
    bos_token_id: Optional[int] = None
    user_token: Optional[str] = "<｜User｜>"
    assistant_token: Optional[str] = "<｜Assistant｜>"
    # think_token: Optional[str] = "<think>" # Poka ne ispol'zuem
    # nl_token: Optional[str] = "\n" # Poka ne ispol'zuem

    user_token_id: Optional[int] = None
    assistant_token_id: Optional[int] = None
    # think_token_id: Optional[int] = None
    # nl_id: Optional[int] = None

    try:
        print(f"Loading tokenizer from local path: '{local_tokenizer_path}'...")
        if not local_tokenizer_path.is_dir():
             # Poprobuem put' s opečatkoj, esli osnovnoj ne najden
             local_tokenizer_path_typo = Path("../data/db/DeepSeek-R1-Deistill-Qwen-1.5B")
             if local_tokenizer_path_typo.is_dir():
                  print(f"WARN: Directory '{local_tokenizer_path.name}' not found, using '{local_tokenizer_path_typo.name}' instead.")
                  local_tokenizer_path = local_tokenizer_path_typo
             else:
                  raise FileNotFoundError(f"Local tokenizer directory not found at {local_tokenizer_path.resolve()} or {local_tokenizer_path_typo.resolve()}")

        tokenizer = AutoTokenizer.from_pretrained(
            local_tokenizer_path,
            trust_remote_code=True,
            use_fast=False
        )
        print(f"Tokenizer loaded successfully from {local_tokenizer_path.resolve()}.")
        print(f"Tokenizer class: {tokenizer.__class__.__name__}")

        # Poluchaem ID special'nyh tokenov iz zagruzhennogo tokenizatora
        bos_token_id = tokenizer.bos_token_id
        eos_token_id = tokenizer.eos_token_id # Dolzhen byt' <｜end of sentence｜> ID
        user_token_id = tokenizer.encode(user_token, add_special_tokens=False)[0]
        assistant_token_id = tokenizer.encode(assistant_token, add_special_tokens=False)[0]
        # think_token_id = tokenizer.encode(think_token, add_special_tokens=False)[0]
        # nl_id = tokenizer.encode(nl_token, add_special_tokens=False)[0]

        print(f"Found BOS token ID: {bos_token_id} ('{tokenizer.decode([bos_token_id]) if bos_token_id is not None else 'None'}')")
        print(f"Found EOS token ID: {eos_token_id} ('{tokenizer.decode([eos_token_id]) if eos_token_id is not None else 'None'}')")
        print(f"Found User token ID: {user_token_id} ('{user_token}')")
        print(f"Found Assistant token ID: {assistant_token_id} ('{assistant_token}')")
        # print(f"Found Think token ID: {think_token_id} ('{think_token}')")
        # print(f"Found Newline token ID: {nl_id} ('\\n')")

        # Proverka pad_token
        if tokenizer.pad_token_id is None:
            if eos_token_id is not None:
                tokenizer.pad_token_id = eos_token_id
                tokenizer.pad_token = tokenizer.decode([eos_token_id])
                print(f"Set pad_token_id = eos_token_id ({eos_token_id})")
            else:
                print("WARN: EOS token ID not found in tokenizer. Adding '<pad>' token.")
                tokenizer.add_special_tokens({'pad_token': '<pad>'})
                tokenizer.pad_token_id = tokenizer.encode('<pad>', add_special_tokens=False)[0]
                print(f"Added and set pad_token_id to '<pad>' ({tokenizer.pad_token_id})")
        else:
             print(f"Found PAD token ID: {tokenizer.pad_token_id} ('{tokenizer.pad_token}')")


    except Exception as e: print(f"ERROR: Failed tokenizer load from '{local_tokenizer_path}': {e}"); traceback.print_exc(); return
    if eos_token_id is None: print("WARN: EOS token ID not determined. Generation might not stop correctly.")
    if None in [user_token_id, assistant_token_id]: # Ubral proverku think/nl
        print("ERROR: Failed to get IDs for User/Assistant tokens.")
        return

    # --- Zagruzka Karty Processorov ---
    map_model_name = model_name_hf.split('/')[-1]
    processor_map: Dict[str, str] = {}
    proc_map_file = db_path / f"{map_model_name}_proc_map.pkl"
    if proc_map_file.is_file():
        try:
            with open(proc_map_file, 'rb') as f: processor_map = pickle.load(f)
            print(f"Loaded processor map ({len(processor_map)} entries) from {proc_map_file}")
        except Exception as e: print(f"Warning: Failed processor map load: {e}.")
    else: print(f"ERROR: Processor map file not found: {proc_map_file}"); return

    # --- Inicializacija Veector ---
    vec: Optional[Veector] = None
    try:
        vec = Veector(db_dir=db_path)
        print(f"Veector core v{CORE_VERSION} initialized using DB at: {vec.db.db_root_path.resolve()}")
    except Exception as e: print(f"FATAL: Veector init failed: {e}"); return

    # --- Proverka Processorov ---
    required_proc_keys = ["embedding", "final_norm", "lm_head"]
    for i in range(num_layers): required_proc_keys.extend([f"attn_{i}", f"ffn_{i}"])
    missing_procs = [key for key in required_proc_keys if key not in processor_map]
    if missing_procs: print(f"ERROR: Required processors missing from map: {missing_procs}"); vec.db.close(); return
    embedding_processor_id = processor_map["embedding"]
    final_norm_id = processor_map["final_norm"]
    lm_head_id = processor_map["lm_head"]
    print("All required processor IDs found in map.")

    # --- Podgotovka Vhodnyh Dannyh (Ruchnoe Formirovanie v stile GGUF) ---
    prompt_input_ids: Optional[np.ndarray] = None
    try:
        print("Manually constructing prompt tokens (GGUF-style)...")
        # Struktura: BOS + <｜User｜> + message + <｜Assistant｜>
        user_text_ids = tokenizer.encode(text, add_special_tokens=False)

        input_ids_list = []
        if bos_token_id is not None:
            input_ids_list.append(bos_token_id)

        input_ids_list.append(user_token_id)
        input_ids_list.extend(user_text_ids)
        input_ids_list.append(assistant_token_id)

        prompt_input_ids = np.array([input_ids_list], dtype=np.int64)

        print(f"\n--- Prepared Input (Manual GGUF-style Construction) ---")
        print(f"Initial Prompt IDs shape: {prompt_input_ids.shape}")
        print(f"Initial Prompt IDs: {prompt_input_ids[0].tolist()}")
        print(f"Initial Decoded Tokens: {tokenizer.convert_ids_to_tokens(prompt_input_ids[0].tolist())}")
        print(f"Initial Decoded String: '{tokenizer.decode(prompt_input_ids[0])}'")

    except Exception as e: print(f"Error constructing prompt tokens: {e}"); traceback.print_exc(); vec.db.close(); return
    if prompt_input_ids is None: print("ERROR: prompt_input_ids are None after manual construction."); vec.db.close(); return

    # --- Inicializacija KV Kjesha ---
    kv_cache_list: List[Tuple[np.ndarray, np.ndarray]] = []
    cache_dtype = np.float16
    batch_size = 1
    print(f"Initializing KV Cache for {num_layers} layers...")
    cache_shape = (batch_size, num_kv_heads, max_seq_len, head_dim)
    print(f"  Shape per layer: K={cache_shape}, V={cache_shape}, dtype={cache_dtype}")
    for i in range(num_layers):
        k_cache_layer = np.zeros(cache_shape, dtype=cache_dtype)
        v_cache_layer = np.zeros(cache_shape, dtype=cache_dtype)
        kv_cache_list.append((k_cache_layer, v_cache_layer))
    print("KV Cache initialized.")
    log_memory_usage("After KV Cache Init")

    # --- Zapusk Avtoregressionnoj Generacii ---
    start_inference_time = time.time()
    knowledge_group_id = GROUP_IDX_QWEN_KNOWLEDGE

    print(f"\n--- Starting Autoregressive Generation with KV Cache ---")
    generated_ids: List[int] = []
    current_input_ids_for_step: np.ndarray = prompt_input_ids
    prompt_len = current_input_ids_for_step.shape[1]
    total_seq_len = prompt_len

    try:
        for step in range(max_new_tokens):
            step_start_time = time.time()
            current_seq_length = current_input_ids_for_step.shape[1]
            start_pos = total_seq_len - current_seq_length
            position_ids = np.arange(start_pos, total_seq_len, dtype=np.int64).reshape(1, current_seq_length)

            if total_seq_len > max_seq_len:
                print(f"\nERROR: total_seq_len ({total_seq_len}) exceeds max_seq_len ({max_seq_len}). Cannot continue.")
                break

            print(f"\n--- Step {step + 1}/{max_new_tokens} (Pos: {start_pos}..{total_seq_len-1}) ---")
            log_tensor_stats("Input IDs", current_input_ids_for_step)

            # 1. Embedding
            print(f"  Running Embedding...")
            compute_context_embed = {
                "input_data": current_input_ids_for_step,
                "required_nest": nest_level,
                "target_knowledge_group": knowledge_group_id
            }
            embed_result = vec.compute(embedding_processor_id, context=compute_context_embed)
            if not (embed_result and embed_result.get("status") == "completed"):
                raise RuntimeError(f"Embedding failed at step {step+1}: {embed_result}")
            current_hidden_states = embed_result.get("data")
            log_tensor_stats("Embedding Output", current_hidden_states, log_values=(step < 2))
            if current_hidden_states is None:
                raise RuntimeError(f"Embedding returned None data at step {step+1}.")

            # 2. Sloi Transformera
            residual_input = current_hidden_states

            for layer_idx in range(num_layers):
                # Uproshhennoe logirovanie dlja posledujushhih shagov
                if step > 0 and layer_idx % 5 != 0:
                     if layer_idx == num_layers -1:
                          print(f"\n  Layer {layer_idx}: Processing...")
                     else:
                          continue
                else:
                    print(f"\n  Layer {layer_idx}: Processing...")

                log_tensor_stats(f"L{layer_idx} Input (Hidden State)", current_hidden_states, log_values=(step < 2))

                attn_proc_id = processor_map[f"attn_{layer_idx}"]
                ffn_proc_id = processor_map[f"ffn_{layer_idx}"]

                past_key, past_value = kv_cache_list[layer_idx]
                log_tensor_stats(f"L{layer_idx} Input past_key", past_key)
                log_tensor_stats(f"L{layer_idx} Input past_value", past_value)

                # Attention
                attn_context = {
                    "input_data": current_hidden_states,
                    "residual_input": residual_input,
                    "required_nest": nest_level,
                    "target_knowledge_group": knowledge_group_id,
                    "position_ids": position_ids,
                    "past_key": past_key,
                    "past_value": past_value,
                    "start_pos": start_pos,
                    "total_seq_len": total_seq_len
                }
                attn_result = vec.compute(attn_proc_id, context=attn_context)

                if not (attn_result and attn_result.get("status") == "completed"):
                    prov = attn_result.get("provenance", {})
                    error_msg = prov.get("error", "Unknown error")
                    print(f"    ERROR: Attn L{layer_idx} failed at step {step+1}: Status={attn_result.get('status')}, Error='{error_msg}'")
                    print(f"    Failed Attn Context: {attn_context}")
                    log_tensor_stats(f"L{layer_idx} FAILED Attn Input HS", current_hidden_states, log_values=True)
                    log_tensor_stats(f"L{layer_idx} FAILED Attn Input Res", residual_input, log_values=True)
                    raise RuntimeError(f"Attn L{layer_idx} failed at step {step+1}")

                attn_output = attn_result.get("data")
                result_step_context = attn_result.get("step_context", {})
                log_tensor_stats(f"L{layer_idx} Attn Output", attn_output, log_values=(step < 2))

                new_key = result_step_context.get('k_cache_out')
                new_value = result_step_context.get('v_cache_out')

                if new_key is not None and new_value is not None:
                    if np.any(np.isnan(new_key)) or np.any(np.isnan(new_value)):
                        print(f"    ERROR: NaN detected in new K/V cache for L{layer_idx}! NOT updating cache.")
                        log_tensor_stats(f"L{layer_idx} NaN New Key", new_key, log_values=True)
                        log_tensor_stats(f"L{layer_idx} NaN New Value", new_value, log_values=True)
                    else:
                        kv_cache_list[layer_idx] = (new_key, new_value)
                        log_tensor_stats(f"L{layer_idx} Updated K Cache", new_key)
                        log_tensor_stats(f"L{layer_idx} Updated V Cache", new_value)
                else:
                    print(f"    WARN: K/V cache values ('k_cache_out', 'v_cache_out') not found in attn_result step_context for L{layer_idx}. Cache NOT updated.")

                if attn_output is None:
                    raise RuntimeError(f"Attn L{layer_idx} returned None data at step {step+1}.")

                current_hidden_states = attn_output
                residual_input_ffn = attn_output

                # FFN
                ffn_context = {
                    "input_data": current_hidden_states,
                    "residual_input": residual_input_ffn,
                    "required_nest": nest_level,
                    "target_knowledge_group": knowledge_group_id
                }
                ffn_result = vec.compute(ffn_proc_id, context=ffn_context)

                if not (ffn_result and ffn_result.get("status") == "completed"):
                    prov = ffn_result.get("provenance", {})
                    error_msg = prov.get("error", "Unknown error")
                    print(f"    ERROR: FFN L{layer_idx} failed at step {step+1}: Status={ffn_result.get('status')}, Error='{error_msg}'")
                    log_tensor_stats(f"L{layer_idx} FAILED FFN Input HS", current_hidden_states, log_values=True)
                    log_tensor_stats(f"L{layer_idx} FAILED FFN Input Res", residual_input_ffn, log_values=True)
                    raise RuntimeError(f"FFN L{layer_idx} failed at step {step+1}")

                ffn_output = ffn_result.get("data")
                log_tensor_stats(f"L{layer_idx} FFN Output", ffn_output, log_values=(step < 2))
                if ffn_output is None:
                    raise RuntimeError(f"FFN L{layer_idx} returned None data at step {step+1}.")

                current_hidden_states = ffn_output
                residual_input = ffn_output
                # --- Konec cikla po slojam ---

            # 3. Final Norm
            print("  Running Final Norm...")
            log_tensor_stats("Input to Final Norm", current_hidden_states, log_values=(step < 2))
            norm_context = {
                "input_data": current_hidden_states,
                "required_nest": nest_level,
                "target_knowledge_group": knowledge_group_id
            }
            norm_result = vec.compute(final_norm_id, context=norm_context)
            if not (norm_result and norm_result.get("status") == "completed"):
                raise RuntimeError(f"Final Norm failed at step {step+1}: {norm_result}")
            final_normed_states = norm_result.get("data")
            log_tensor_stats("Final Norm Output", final_normed_states, log_values=(step < 2))
            if final_normed_states is None:
                raise RuntimeError(f"Final Norm returned None data at step {step+1}.")

            # 4. LM Head
            print("  Running LM Head...")
            last_token_hidden_state = final_normed_states[:, -1:, :]
            log_tensor_stats("Input to LM Head", last_token_hidden_state, log_values=(step < 2))
            lm_head_context = {
                "input_data": last_token_hidden_state,
                "required_nest": nest_level,
                "target_knowledge_group": knowledge_group_id
            }
            logits_result = vec.compute(lm_head_id, context=lm_head_context)

            if not (logits_result and logits_result.get("status") == "completed"):
                raise RuntimeError(f"LM Head failed at step {step+1}: {logits_result}")
            final_logits = logits_result.get("data")
            log_tensor_stats("LM Head Output (Logits)", final_logits, log_values=(step < 2))
            if final_logits is None:
                raise RuntimeError(f"LM Head returned None data at step {step+1}.")

            # 5. Семплирование следующего токена
            print("  Sampling next token...")
            last_token_logits = final_logits[0, 0, :]
            log_tensor_stats(f"Logits for Sampling (Step {step+1})", last_token_logits, log_values=True)

            predicted_token_id = sample_top_p(
                logits=last_token_logits,
                temperature=temperature,
                top_p=top_p
            )
            if np.isnan(predicted_token_id):
                 print("FATAL ERROR: Sampling returned NaN token ID!")
                 predicted_token_id = 0

            predicted_token_id = int(predicted_token_id)

            print(f"Step {step+1}: Generated token ID = {predicted_token_id}, Decoded = '{tokenizer.decode([predicted_token_id])}'")

            # 6. Проверка условия остановки (EOS)
            if eos_token_id is not None and predicted_token_id == eos_token_id:
                if step == 0:
                    print("WARN: EOS token generated as the first token. Something might be wrong.")
                else:
                    print(f"\nEOS token ({eos_token_id}) generated. Stopping generation.")
                    break

            # 7. Подготовка к следующей итерации
            generated_ids.append(predicted_token_id)
            current_input_ids_for_step = np.array([[predicted_token_id]], dtype=np.int64)
            total_seq_len += 1

            if total_seq_len >= max_seq_len:
                print(f"\nMaximum sequence length ({max_seq_len}) reached. Stopping generation.")
                break

            if tokenizer:
                current_token_str = tokenizer.decode([predicted_token_id], skip_special_tokens=True)
                print(current_token_str, end='', flush=True)
            else:
                print(f" [ID:{predicted_token_id}]", end='', flush=True)

            if vec:
                vec.clear_cache(clear_knowledge=False, clear_compute=True)

            log_memory_usage(f"End of Step {step+1}")
            print(f"  Step {step+1} time: {time.time() - step_start_time:.3f}s")

        # --- Konec cikla generacii ---
        print()

        # --- Vyvod rezul'tata ---
        print("\n--- Final Generated Sequence (Decoded) ---")
        if tokenizer:
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            print(f"Generated Text Only: '{generated_text}'")
            full_output_ids = np.concatenate([prompt_input_ids[0], np.array(generated_ids, dtype=np.int64)])
            full_response = tokenizer.decode(full_output_ids, skip_special_tokens=False)
            print(f"Full Response (incl. prompt, special tokens): '{full_response}'")
            print(f"Generated IDs: {generated_ids}")
        elif not generated_ids:
            print("No tokens were generated.")
        else:
            print("Cannot decode: tokenizer unavailable.")
            print(f"Generated IDs: {generated_ids}")

    except Exception as e:
        print(f"\n--- ERROR during inference execution ---")
        print(f"{e}")
        traceback.print_exc()
    finally:
        if vec and hasattr(vec, 'db') and vec.db:
            try:
                vec.db.close()
                print("\nDatabase connection closed.")
            except Exception as db_close_e:
                print(f"Error closing DB connection: {db_close_e}")

    end_inference_time = time.time()
    print(f"\n--- Inference Script Finished in {end_inference_time - start_inference_time:.3f} seconds ---")
    log_memory_usage("End of inference function")


# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Qwen inference using Veector.")
    parser.add_argument("--prompt", type=str, default="Hello!", help="Input prompt.")
    parser.add_argument("--db_path", type=str, default="../data/db", help="Path to the Veector database.")
    # Argument dlja puti k modeli/tokenizatoru (mozhet byt' ID HF ili lokal'nyj put')
    parser.add_argument("--model_path", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", help="HF model identifier or local path to model/tokenizer files.")
    parser.add_argument("--nest", type=int, default=1, help="Target nest level for processors.")
    parser.add_argument("--temp", type=float, default=TEMPERATURE, help="Sampling temperature.")
    parser.add_argument("--top_p", type=float, default=TOP_P, help="Top-p nucleus sampling.")
    parser.add_argument("--max_tokens", type=int, default=MAX_NEW_TOKENS, help="Maximum new tokens to generate.")
    parser.add_argument("--max_len", type=int, default=MAX_SEQ_LEN, help="Maximum sequence length for KV cache.")

    args = parser.parse_args()
    target_db_path = Path(args.db_path)
    # Ispol'zuem argument --model_path dlja zagruzki tokenizatora i konfiga
    model_identifier_or_path = args.model_path

    print(f"--- Starting Inference ---")
    print(f"DB: {target_db_path.resolve()}")
    print(f"Input: '{args.prompt}'")
    print(f"Model Source: {model_identifier_or_path}") # Logiruem istochnik modeli/tokenizatora
    print(f"Nest: {args.nest}")
    try: print(f"Core: {CORE_VERSION}")
    except NameError: print("Core version unknown")
    try: print(f"Tensors: {TENSORS_VERSION}")
    except NameError: print("Tensors version unknown")
    try: from veectordb import VEECTORDB_VERSION; print(f"DB: {VEECTORDB_VERSION}")
    except (ImportError, NameError): print("DB version unknown")
    try: print(f"Ops: {OPERATIONS_VERSION}")
    except NameError: print("Ops version unknown")

    if target_db_path.is_dir():
          run_qwen_inference(
              text=args.prompt,
              db_path=target_db_path,
              model_name_hf=model_identifier_or_path, # Peredaem put' ili ID
              nest_level=args.nest,
              temperature=args.temp,
              top_p=args.top_p,
              max_new_tokens=args.max_tokens,
              max_seq_len=args.max_len
          )
    else:
        print(f"ERROR: Database directory not found at {target_db_path}")

