# FILE: veector_models/qwen2/ops.py
# Operacii vysokogo urovnja, specifichnye dlja arhitektury Qwen2
# Version: 0.4 (Realizovan RMSNorm, Attention, MLP)

import numpy as np
from typing import Dict, Any, Optional, Tuple
import traceback # Dlja otladki oshibok

# Importiruem PyTorch, tak kak planiruem ispol'zovat' ego dlja realizacii
try:
    import torch
    import torch.nn.functional as F
    # Importiruem funkciju aktivacii iz transformers
    from transformers.activations import ACT2FN
    TORCH_AVAILABLE = True
except ImportError:
    print("WARN: PyTorch or transformers.activations not available. Qwen2 high-level ops will not function.")
    TORCH_AVAILABLE = False

# --- OP Kody (kopija dlja spravki) ---
OP_QWEN2_RMSNORM = [300, 0, 0]
OP_QWEN2_ATTENTION = [300, 1, 0]
OP_QWEN2_MLP = [300, 2, 0]

# --- Vspomogatel'nye funkcii (adaptirovany iz modeling_qwen2.py) ---

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors."""
    # Izmeneno: unsqueeze_dim=2 dlja formata B, H, S, D
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep).
    Expands KV heads to match query heads for Grouped Query Attention.
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

# --- Realizacija Operacij ---

def qwen2_rmsnorm(current_data: Optional[np.ndarray], **kw) -> Optional[np.ndarray]:
    """
    Vypolnjaet Qwen2 RMS Normalization.
    Ozhidaet v kw: 'norm_weight' (np.ndarray), 'eps' (float, default 1e-6).
    """
    # print(f"--- Executing: qwen2_rmsnorm ---")
    if not TORCH_AVAILABLE: print("  ERROR: qwen2_rmsnorm requires PyTorch."); return None
    if current_data is None: print("  ERROR: qwen2_rmsnorm received None input."); return None
    norm_weight_np = kw.get('norm_weight')
    eps = kw.get('eps', 1e-6)
    if norm_weight_np is None: print("  ERROR: qwen2_rmsnorm requires 'norm_weight'."); return None
    if not isinstance(norm_weight_np, np.ndarray): print(f"  ERROR: qwen2_rmsnorm expects 'norm_weight' as np.ndarray."); return None
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    try:
        hidden_states_torch = torch.from_numpy(current_data).to(device)
        norm_weight_torch = torch.from_numpy(norm_weight_np).to(device)
        input_dtype_torch = hidden_states_torch.dtype
        hidden_states_fp32 = hidden_states_torch.to(torch.float32)
        variance = hidden_states_fp32.pow(2).mean(-1, keepdim=True)
        hidden_states_normalized = hidden_states_fp32 * torch.rsqrt(variance + eps)
        output_torch = norm_weight_torch.to(torch.float32) * hidden_states_normalized
        output_torch_casted = output_torch.to(input_dtype_torch)
        output_np = output_torch_casted.cpu().numpy()
        return output_np
    except Exception as e: print(f"  ERROR during qwen2_rmsnorm execution: {e}"); traceback.print_exc(); return None

def qwen2_attention(current_data: Optional[np.ndarray], **kw) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Vypolnjaet Qwen2 Attention blok (bez final'nogo residual add).
    Ozhidaet v kw: hidden_states (current_data), ves/biasy QKV,O, kontekst (pos_ids, cache, start_pos...), parametry (heads, dims...).
    Vozvrashhaet kortezh: (attn_output_proj_np, updated_k_np, updated_v_np)
    """
    print(f"--- Executing: qwen2_attention (Layer {kw.get('layer_idx', 'N/A')}) ---")
    if not TORCH_AVAILABLE: print("  ERROR: qwen2_attention requires PyTorch."); return None
    if current_data is None: print("  ERROR: qwen2_attention received None input (hidden_states)."); return None

    # Izvlechenie vhodnyh dannyh i parametrov
    hidden_states_np = current_data
    dtype_np = hidden_states_np.dtype
    q_weights_np = kw.get('q_weights'); k_weights_np = kw.get('k_weights'); v_weights_np = kw.get('v_weights'); o_weights_np = kw.get('o_weights')
    q_bias_np = kw.get('q_bias'); k_bias_np = kw.get('k_bias'); v_bias_np = kw.get('v_bias')
    position_ids_np = kw.get('position_ids'); past_key_np = kw.get('past_key'); past_value_np = kw.get('past_value')
    start_pos = kw.get('start_pos'); layer_idx = kw.get('layer_idx')
    num_heads = kw.get('num_heads'); num_kv_heads = kw.get('num_kv_heads'); head_dim = kw.get('head_dim')
    rope_theta = kw.get('rope_theta', 1000000.0)

    required_args = {"q_weights": q_weights_np, "k_weights": k_weights_np, "v_weights": v_weights_np, "o_weights": o_weights_np, "position_ids": position_ids_np, "past_key": past_key_np, "past_value": past_value_np, "start_pos": start_pos, "layer_idx": layer_idx, "num_heads": num_heads, "num_kv_heads": num_kv_heads, "head_dim": head_dim}
    for name, arg in required_args.items():
        if arg is None: print(f"  ERROR: qwen2_attention missing required keyword argument '{name}'."); return None

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype_torch = torch.float16 if dtype_np == np.float16 else torch.float32 # Sohranjaem ishodnyj tip

    try:
        # Konvertacija v Torch Tenzory
        hidden_states = torch.from_numpy(hidden_states_np).to(dtype=dtype_torch, device=device)
        q_weights = torch.from_numpy(q_weights_np).to(dtype=dtype_torch, device=device)
        k_weights = torch.from_numpy(k_weights_np).to(dtype=dtype_torch, device=device)
        v_weights = torch.from_numpy(v_weights_np).to(dtype=dtype_torch, device=device)
        o_weights = torch.from_numpy(o_weights_np).to(dtype=dtype_torch, device=device)
        q_bias = torch.from_numpy(q_bias_np).to(dtype=dtype_torch, device=device) if q_bias_np is not None else None
        k_bias = torch.from_numpy(k_bias_np).to(dtype=dtype_torch, device=device) if k_bias_np is not None else None
        v_bias = torch.from_numpy(v_bias_np).to(dtype=dtype_torch, device=device) if v_bias_np is not None else None
        position_ids = torch.from_numpy(position_ids_np).to(device=device)
        past_key = torch.from_numpy(past_key_np).to(dtype=dtype_torch, device=device)
        past_value = torch.from_numpy(past_value_np).to(dtype=dtype_torch, device=device)

        # QKV Projekcii i Reshape
        bsz, q_len, _ = hidden_states.size()
        query_states = F.linear(hidden_states, q_weights, q_bias).view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
        key_states = F.linear(hidden_states, k_weights, k_bias).view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
        value_states = F.linear(hidden_states, v_weights, v_bias).view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)

        # RoPE
        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
        position_ids_expanded = position_ids[:, None, :].float()
        inv_freq_expanded = inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().to(dtype=dtype_torch)
        sin = emb.sin().to(dtype=dtype_torch)
        query_states_rot, key_states_rot = apply_rotary_pos_emb(query_states, key_states, cos, sin, unsqueeze_dim=2)

        # KV Cache Update
        if start_pos + q_len > past_key.shape[-2]: print(f"  ERROR: KV Cache update out of bounds!"); return None
        updated_key = past_key # Obnovljaem na meste (no Torch sozdaet novyj tensor pri slice assignment)
        updated_value = past_value
        updated_key[:, :, start_pos : start_pos + q_len, :] = key_states_rot
        updated_value[:, :, start_pos : start_pos + q_len, :] = value_states
        # print(f"  KV Cache updated at start_pos={start_pos} for seq_len={q_len}") # Logirovanie obnovlenija

        # Grouped Query Attention (Repeat KV)
        num_key_value_groups = num_heads // num_kv_heads
        key_states_repeated = repeat_kv(updated_key, num_key_value_groups)
        value_states_repeated = repeat_kv(updated_value, num_key_value_groups)

        # Scaled Dot-Product Attention (SDPA)
        attn_output = F.scaled_dot_product_attention(query_states_rot, key_states_repeated, value_states_repeated, attn_mask=None, dropout_p=0.0, is_causal=True)

        # Reshape/Transpose Vyhoda Attention
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, hidden_size)

        # Output Projection
        attn_output_proj = F.linear(attn_output, o_weights, None)

        # Konvertacija Vyhoda obratno v NumPy
        attn_output_proj_np = attn_output_proj.cpu().numpy().astype(dtype_np)
        updated_k_np = updated_key.cpu().numpy().astype(past_key_np.dtype)
        updated_v_np = updated_value.cpu().numpy().astype(past_value_np.dtype)

        # print(f"  qwen2_attention finished. Output shape: {attn_output_proj_np.shape}") # Logirovanie vyhoda
        return (attn_output_proj_np, updated_k_np, updated_v_np)

    except Exception as e:
        print(f"  ERROR during qwen2_attention execution: {e}")
        traceback.print_exc()
        return None

# --- Realizacija qwen2_mlp ---
def qwen2_mlp(current_data: Optional[np.ndarray], **kw) -> Optional[np.ndarray]:
    """
    Vypolnjaet Qwen2 MLP (FFN) blok (bez final'nogo residual add).
    Ozhidaet v kw:
      - hidden_states (current_data): Vhodnoj normirovannyj tenzor (NumPy).
      - Vesa: gate_weights, up_weights, down_weights (NumPy).
      - hidden_act (str, default 'silu'): Imja funkcii aktivacii.
    Vozvrashhaet vyhod Down projekcii (NumPy).
    """
    print(f"--- Executing: qwen2_mlp ---")
    if not TORCH_AVAILABLE: print("  ERROR: qwen2_mlp requires PyTorch."); return None
    if current_data is None: print("  ERROR: qwen2_mlp received None input."); return None

    # Znanija
    gate_weights_np = kw.get('gate_weights')
    up_weights_np = kw.get('up_weights')
    down_weights_np = kw.get('down_weights')

    # Parametry
    hidden_act = kw.get('hidden_act', 'silu') # Berem iz kw ili default

    # Proverka
    required_args = {"gate_weights": gate_weights_np, "up_weights": up_weights_np, "down_weights": down_weights_np}
    for name, arg in required_args.items():
        if arg is None: print(f"  ERROR: qwen2_mlp missing required keyword argument '{name}'."); return None

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype_np = current_data.dtype
    dtype_torch = torch.float16 if dtype_np == np.float16 else torch.float32

    try:
        # Konvertacija v Torch
        hidden_states = torch.from_numpy(current_data).to(dtype=dtype_torch, device=device)
        gate_weights = torch.from_numpy(gate_weights_np).to(dtype=dtype_torch, device=device)
        up_weights = torch.from_numpy(up_weights_np).to(dtype=dtype_torch, device=device)
        down_weights = torch.from_numpy(down_weights_np).to(dtype=dtype_torch, device=device)

        # Poluchaem funkciju aktivacii
        act_fn = ACT2FN[hidden_act]

        # Vychislenija MLP
        gate_proj = F.linear(hidden_states, gate_weights)
        up_proj = F.linear(hidden_states, up_weights)
        activated = act_fn(gate_proj) * up_proj
        down_proj = F.linear(activated, down_weights)

        # Konvertacija obratno v NumPy
        output_np = down_proj.cpu().numpy().astype(dtype_np)

        print(f"  qwen2_mlp finished. Output shape: {output_np.shape}")
        return output_np

    except Exception as e:
        print(f"  ERROR during qwen2_mlp execution: {e}")
        traceback.print_exc()
        return None


# --- Slovar' operacij dlja etogo modulja ---
qwen2_operations = {
    tuple(OP_QWEN2_RMSNORM): qwen2_rmsnorm,
    tuple(OP_QWEN2_ATTENTION): qwen2_attention,
    tuple(OP_QWEN2_MLP): qwen2_mlp,
}

print(f"Veector Qwen2 Ops Module Loaded. Found {len(qwen2_operations)} operations.")

