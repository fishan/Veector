# === Skript dlja prohoda HF modeli v float32 i sohranenija promezhutochnyh vyhodov ===

import time
import pickle
import numpy as np
import traceback
import os
from pathlib import Path
from functools import partial

# --- Neobhodimye biblioteki ---
try:
    import torch
    from torch import nn
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
    print("Torch and Transformers imported successfully.")
except ImportError as e:
    print(f"FATAL ERROR: Missing essential libraries (torch, transformers): {e}")
    print("Please install them: pip install torch transformers accelerate")
    exit()

# --- Konfiguracija ---
# Ubedites', chto eti peremennye sootvetstvujut vashemu okruzheniju
MODEL_SOURCE = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
TOKENIZER_SOURCE = MODEL_SOURCE

PROMPT = "Hello, how are you?" # Tot zhe prompt, chto i v skripte sravnenija
# >>> IZMENENO: Novoe imja fajla dlja float32 vyhodov <<<
OUTPUT_FILENAME = "hf_reference_outputs_fp32.pkl"

# --- Zagruzka Tokenizatora ---
print("\\n--- Loading Tokenizer ---")
tokenizer = None
try:
    print(f"Loading Tokenizer from: {TOKENIZER_SOURCE}")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_SOURCE, trust_remote_code=True)
    print(f"Tokenizer class: {tokenizer.__class__.__name__}")
except Exception as e:
    print(f"FATAL ERROR loading tokenizer: {e}")
    exit()

# --- Podgotovka vhodnyh dannyh ---
print("\\n--- Preparing Input IDs ---")
input_ids_torch = None
input_seq_len = 0
try:
    messages = [{"role": "user", "content": PROMPT}]
    print("Applying chat template...")
    prompt_input_ids_np = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="np"
    )
    if prompt_input_ids_np.ndim == 1:
        prompt_input_ids_np = np.expand_dims(prompt_input_ids_np, axis=0)

    input_seq_len = prompt_input_ids_np.shape[1]
    # Poka ostavljaem na CPU, model' budet zagruzhena na CPU ili GPU nizhe
    input_ids_torch = torch.tensor(prompt_input_ids_np)

    print(f"Input IDs shape: {input_ids_torch.shape}")
    print(f"Input Sequence Length: {input_seq_len}")
    print(f"Decoded Input: '{tokenizer.decode(input_ids_torch[0].cpu().numpy())}'")
except Exception as e:
    print(f"FATAL ERROR preparing input: {e}")
    exit()

# --- Zagruzka i Progon Etalonnoj Modeli v Float32 ---
print(f"\\n--- Loading and Running HF Model ({MODEL_SOURCE}) in float32 ---")
hf_outputs = {}
hook_handles = []
model_fp32 = None

def get_hook(name):
    def hook_fn(module, input, output):
        actual_output = output[0] if isinstance(output, tuple) else output
        print(f"  [HOOK] Captured output for: {name} (Shape: {actual_output.shape}, Device: {actual_output.device})")
        # Sohranjaem na CPU v formate NumPy float32
        hf_outputs[name] = actual_output.detach().cpu().numpy().astype(np.float32)
    return hook_fn

try:
    print(f"Loading HF Model {MODEL_SOURCE} with float32...")
    # >>> IZMENENO: Zagruzhaem s torch_dtype=torch.float32 <<<
    model_fp32 = AutoModelForCausalLM.from_pretrained(MODEL_SOURCE, torch_dtype=torch.float32, trust_remote_code=True)
    model_fp32.eval()
    # Peremestite na GPU, esli neobhodimo i vozmozhno
    # model_fp32.to('cuda')
    # input_ids_torch = input_ids_torch.to(model_fp32.device) # Peremestit' vhodnye dannye tozhe
    print(f"HF Model loaded to device: {model_fp32.device}")

    # Registracija Hukov
    print("Registering hooks for float32 model...")
    model_config = model_fp32.config
    num_layers = model_config.num_hidden_layers
    hook_handles.append(model_fp32.model.embed_tokens.register_forward_hook(get_hook("embed_tokens")))
    for i in range(num_layers):
        hook_handles.append(model_fp32.model.layers[i].register_forward_hook(get_hook(f"layer_{i}_output")))
    hook_handles.append(model_fp32.model.norm.register_forward_hook(get_hook("final_norm")))
    hook_handles.append(model_fp32.lm_head.register_forward_hook(get_hook("lm_head")))
    print(f"Registered {len(hook_handles)} hooks.")

    # Prjamoj prohod
    print("Running HF model forward pass (float32)...")
    with torch.no_grad():
        hf_model_output = model_fp32(input_ids_torch.to(model_fp32.device), use_cache=False) # Ubedimsja chto input na tom zhe device
    print("HF forward pass complete.")

except Exception as e:
    print(f"FATAL ERROR during HF float32 execution: {e}")
    traceback.print_exc()
finally:
    # Vsegda udaljajem huki i model' posle ispol'zovanija
    for handle in hook_handles: handle.remove()
    print("Hooks removed.")
    if 'model_fp32' in locals() and model_fp32 is not None:
        del model_fp32
        if 'torch' in locals() and hasattr(torch, 'cuda'): torch.cuda.empty_cache()
        gc.collect()
        print("Cleaned up float32 model.")

# --- Sohranenie rezul'tatov ---
if hf_outputs: # Sohranjaem tol'ko esli chto-to sobrali
    print(f"\\n--- Saving Captured Float32 Outputs to {OUTPUT_FILENAME} ---")
    try:
        with open(OUTPUT_FILENAME, 'wb') as f:
            pickle.dump(hf_outputs, f, pickle.HIGHEST_PROTOCOL)
        print(f"Successfully saved {len(hf_outputs)} captured outputs.")
        print("Saved keys:", list(hf_outputs.keys()))
    except Exception as e:
        print(f"FATAL ERROR saving outputs: {e}")
        traceback.print_exc()
else:
    print("\\n--- No outputs captured from HF model, skipping save. ---")


print(f"\\n--- Script Finished ---")
