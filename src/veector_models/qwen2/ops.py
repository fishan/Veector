# FILE: veector_models/qwen2/ops.py
# Operacii vysokogo urovnja, specifichnye dlja arhitektury Qwen2
# Version: 0.9 (Added shape debug before F.linear)

import numpy as np
from typing import Dict, Any, Optional, Tuple
import traceback

try:
    import torch
    import torch.nn.functional as F
    from transformers.activations import ACT2FN
    TORCH_AVAILABLE = True
except ImportError:
    print("WARN: PyTorch or transformers.activations not available.")
    TORCH_AVAILABLE = False

OP_QWEN2_RMSNORM = [300, 0, 0]; OP_QWEN2_ATTENTION = [300, 1, 0]; OP_QWEN2_MLP = [300, 2, 0]

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]; x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim: int = 1):
    try:
        cos = cos.unsqueeze(unsqueeze_dim); sin = sin.unsqueeze(unsqueeze_dim)
        q_embed = (q * cos) + (rotate_half(q) * sin); k_embed = (k * cos) + (rotate_half(k) * sin)
        return q_embed, k_embed
    except Exception as e: print(f"Error in apply_rotary_pos_emb: {e}"); raise

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1: return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def qwen2_rmsnorm(current_data: Optional[np.ndarray], **kw) -> Optional[np.ndarray]:
    if not TORCH_AVAILABLE: print("  ERROR: qwen2_rmsnorm requires PyTorch."); return None
    if current_data is None: print("  ERROR: qwen2_rmsnorm received None input."); return None
    norm_weight_np = kw.get('norm_weight'); eps = float(kw.get('eps', 1e-6))
    if norm_weight_np is None: print("  ERROR: qwen2_rmsnorm requires 'norm_weight'."); return None
    if not isinstance(norm_weight_np, np.ndarray): print(f"  ERROR: qwen2_rmsnorm expects 'norm_weight' as np.ndarray."); return None
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    try:
        hidden_states_torch = torch.from_numpy(current_data).to(device); norm_weight_torch = torch.from_numpy(norm_weight_np).to(device)
        input_dtype_torch = hidden_states_torch.dtype; hidden_states_fp32 = hidden_states_torch.to(torch.float32)
        norm_weight_fp32 = norm_weight_torch.to(torch.float32); variance = hidden_states_fp32.pow(2).mean(-1, keepdim=True)
        hidden_states_normalized = hidden_states_fp32 * torch.rsqrt(variance + eps)
        output_torch_fp32 = norm_weight_fp32 * hidden_states_normalized
        output_torch_casted = output_torch_fp32.to(input_dtype_torch); output_np = output_torch_casted.cpu().numpy()
        return output_np
    except Exception as e: print(f"  ERROR during qwen2_rmsnorm execution: {e}"); traceback.print_exc(); return None

def qwen2_attention(current_data: Optional[np.ndarray], **kw) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    if not TORCH_AVAILABLE: print("  ERROR: qwen2_attention requires PyTorch."); return None
    if current_data is None: print("  ERROR: qwen2_attention received None input (hidden_states)."); return None

    hidden_states_np = current_data; dtype_np = hidden_states_np.dtype
    q_weights_np = kw.get('q_weights'); k_weights_np = kw.get('k_weights'); v_weights_np = kw.get('v_weights'); o_weights_np = kw.get('o_weights')
    q_bias_np = kw.get('q_bias'); k_bias_np = kw.get('k_bias'); v_bias_np = kw.get('v_bias')
    position_ids_np = kw.get('position_ids'); past_key_np = kw.get('past_key'); past_value_np = kw.get('past_value')
    start_pos = kw.get('start_pos'); layer_idx = kw.get('layer_idx')
    num_heads = kw.get('num_heads'); num_kv_heads = kw.get('num_kv_heads'); head_dim = kw.get('head_dim')
    rope_theta = float(kw.get('rope_theta', 1000000.0)); total_seq_len = kw.get('total_seq_len'); hidden_size = kw.get('hidden_size')

    required_args = { "q_weights": q_weights_np, "k_weights": k_weights_np, "v_weights": v_weights_np, "o_weights": o_weights_np, "position_ids": position_ids_np, "past_key": past_key_np, "past_value": past_value_np, "start_pos": start_pos, "layer_idx": layer_idx, "num_heads": num_heads, "num_kv_heads": num_kv_heads, "head_dim": head_dim, "total_seq_len": total_seq_len, "hidden_size": hidden_size }
    for name, arg in required_args.items():
        if arg is None:
            if name in ["q_bias", "k_bias", "v_bias"]: continue
            print(f"  ERROR: qwen2_attention missing required keyword argument '{name}'."); return None

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype_torch = torch.float16 if dtype_np == np.float16 else torch.float32

    try:
        hidden_states = torch.from_numpy(hidden_states_np).to(dtype=dtype_torch, device=device)
        q_weights = torch.from_numpy(q_weights_np).to(dtype=dtype_torch, device=device).T
        k_weights = torch.from_numpy(k_weights_np).to(dtype=dtype_torch, device=device).T
        v_weights = torch.from_numpy(v_weights_np).to(dtype=dtype_torch, device=device).T
        o_weights = torch.from_numpy(o_weights_np).to(dtype=dtype_torch, device=device).T
        q_bias = torch.from_numpy(q_bias_np).to(dtype=dtype_torch, device=device) if q_bias_np is not None else None
        k_bias = torch.from_numpy(k_bias_np).to(dtype=dtype_torch, device=device) if k_bias_np is not None else None
        v_bias = torch.from_numpy(v_bias_np).to(dtype=dtype_torch, device=device) if v_bias_np is not None else None
        position_ids = torch.from_numpy(position_ids_np).to(device=device, dtype=torch.long)
        past_key = torch.from_numpy(past_key_np).to(dtype=dtype_torch, device=device)
        past_value = torch.from_numpy(past_value_np).to(dtype=dtype_torch, device=device)

        bsz, q_len, _ = hidden_states.size()

        # --- ДОБАВЛЕНА ОТЛАДОЧНАЯ ПЕЧАТЬ ---
        print(f"  [DEBUG ops.py] Before F.linear(K): hidden_states.shape={hidden_states.shape}, k_weights.shape={k_weights.shape}")
        # --- КОНЕЦ ОТЛАДКИ ---

        query_states = F.linear(hidden_states, q_weights, q_bias).view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
        key_states = F.linear(hidden_states, k_weights, k_bias).view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2) # Ошибка возникает здесь
        value_states = F.linear(hidden_states, v_weights, v_bias).view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)

        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
        if position_ids.ndim == 1: position_ids = position_ids.unsqueeze(0)
        if position_ids.shape[0] != bsz: position_ids = position_ids.expand(bsz, -1)
        position_ids_expanded = position_ids[:, None, :].float(); inv_freq_expanded = inv_freq[None, :, None].float()
        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1); cos = emb.cos().to(dtype=dtype_torch); sin = emb.sin().to(dtype=dtype_torch)
        query_states_rot, key_states_rot = apply_rotary_pos_emb(query_states, key_states, cos, sin, unsqueeze_dim=1)

        seq_len_cached = past_key.shape[-2]
        if start_pos + q_len > seq_len_cached: print(f"  ERROR: KV Cache update out of bounds!"); return None
        updated_key = past_key.clone(); updated_value = past_value.clone()
        updated_key[:, :, start_pos : start_pos + q_len, :] = key_states_rot
        updated_value[:, :, start_pos : start_pos + q_len, :] = value_states

        num_key_value_groups = num_heads // num_kv_heads; actual_kv_len = int(total_seq_len)
        key_states_sliced = updated_key[:, :, :actual_kv_len, :]; value_states_sliced = updated_value[:, :, :actual_kv_len, :]
        key_states_repeated = repeat_kv(key_states_sliced, num_key_value_groups); value_states_repeated = repeat_kv(value_states_sliced, num_key_value_groups)

        attn_output = F.scaled_dot_product_attention(query_states_rot.to(torch.float32), key_states_repeated.to(torch.float32), value_states_repeated.to(torch.float32), attn_mask=None, dropout_p=0.0, is_causal=True).to(dtype=dtype_torch)
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, hidden_size)
        attn_output_proj = F.linear(attn_output, o_weights, None)

        attn_output_proj_np = attn_output_proj.cpu().numpy().astype(dtype_np)
        updated_k_np = updated_key.cpu().numpy().astype(past_key_np.dtype)
        updated_v_np = updated_value.cpu().numpy().astype(past_value_np.dtype)
        return (attn_output_proj_np, updated_k_np, updated_v_np)
    except Exception as e: print(f"  ERROR during qwen2_attention execution: {e}"); traceback.print_exc(); return None


def qwen2_mlp(current_data: Optional[np.ndarray], **kw) -> Optional[np.ndarray]:
    """ Vypolnjaet Qwen2 MLP (FFN) blok. """
    if not TORCH_AVAILABLE: print("  ERROR: qwen2_mlp requires PyTorch."); return None
    if current_data is None: print("  ERROR: qwen2_mlp received None input."); return None

    gate_weights_np = kw.get('gate_weights'); up_weights_np = kw.get('up_weights'); down_weights_np = kw.get('down_weights')
    hidden_act_name = kw.get('hidden_act')
    # print(f"  [DEBUG qwen2_mlp] Received hidden_act: {hidden_act_name} (type: {type(hidden_act_name)})") # Оставим для отладки

    required_args = {"gate_weights": gate_weights_np, "up_weights": up_weights_np, "down_weights": down_weights_np}
    for name, arg in required_args.items():
        if arg is None: print(f"  ERROR: qwen2_mlp missing required weight '{name}'."); return None
    if not isinstance(hidden_act_name, str) or not hidden_act_name:
            print(f"  ERROR: qwen2_mlp requires a valid string for 'hidden_act', received: {hidden_act_name}"); return None

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype_np = current_data.dtype; dtype_torch = torch.float16 if dtype_np == np.float16 else torch.float32

    try:
        hidden_states = torch.from_numpy(current_data).to(dtype=dtype_torch, device=device)
        gate_weights = torch.from_numpy(gate_weights_np).to(dtype=dtype_torch, device=device).T
        up_weights = torch.from_numpy(up_weights_np).to(dtype=dtype_torch, device=device).T
        down_weights = torch.from_numpy(down_weights_np).to(dtype=dtype_torch, device=device).T

        try: act_fn = ACT2FN[hidden_act_name]
        except KeyError: print(f"  ERROR: Unknown activation function name '{hidden_act_name}' provided in ACT2FN."); return None

        gate_proj = F.linear(hidden_states, gate_weights)
        up_proj = F.linear(hidden_states, up_weights)
        activated = act_fn(gate_proj) * up_proj
        down_proj = F.linear(activated, down_weights)

        output_np = down_proj.cpu().numpy().astype(dtype_np)
        return output_np
    except Exception as e: print(f"  ERROR during qwen2_mlp execution: {e}"); traceback.print_exc(); return None

# --- Slovar' operacij dlja etogo modulja ---
qwen2_operations = { tuple(OP_QWEN2_RMSNORM): qwen2_rmsnorm, tuple(OP_QWEN2_ATTENTION): qwen2_attention, tuple(OP_QWEN2_MLP): qwen2_mlp, }
print(f"Veector Qwen2 Ops Module Loaded. Found {len(qwen2_operations)} operations.")

