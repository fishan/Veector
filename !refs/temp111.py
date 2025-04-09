# FILE: veector_models/qwen2/ops.py
# High-level operations specific to the Qwen2 architecture for Veector.

import numpy as np
import traceback
from typing import Dict, Any, Optional, Tuple, Union, List

# --- Version ---
QWEN2_OPS_VERSION = "0.1.14"
# Changelog:
# 0.1.14: Renamed attention function to qwen2_attention (using eager implementation).
#         Added QWEN2_OPS_VERSION constant. PEP8 improvements, type hints, docstrings.
# 0.1.13: Added qwen2_attention_eager_v1 returning intermediates.
# 0.1.12: Added qwen2_mlp returning intermediates.
# 0.1.11: Initial version with qwen2_rmsnorm, qwen2_attention (SDPA), qwen2_mlp.


# --- PyTorch Dependency ---
try:
    import torch
    import torch.nn.functional as F
    # Attempt to import ACT2FN, handle potential absence
    try:
        from transformers.activations import ACT2FN
    except ImportError:
        print("WARN: transformers.activations not found. Mocking ACT2FN for 'silu'.")
        # Basic mock for silu if transformers is not fully available
        class MockACT2FN:
            def __getitem__(self, key):
                if key == 'silu':
                    # Basic SiLU implementation
                    return lambda x: x * torch.sigmoid(x)
                raise KeyError(f"Activation function '{key}' not available in mock.")
        ACT2FN = MockACT2FN()
    TORCH_AVAILABLE = True
    print(f"INFO: PyTorch version {torch.__version__} detected.")
except ImportError:
    print("ERROR: PyTorch is required for Qwen2 operations but is not installed.")
    TORCH_AVAILABLE = False
    # Define dummy types/functions to prevent NameErrors later if code proceeds
    class torch:
        Tensor = type(None)
        dtype = type(None)
        float16 = 'float16_mock'
        float32 = 'float32_mock'
        long = 'long_mock'
        bool = 'bool_mock'
        device = type(None)
        @staticmethod
        def cuda(*args, **kwargs): return type('MockCuda', (), {'is_available': lambda: False})()
    class F: pass
    class ACT2FN: pass

# --- Operation Codes ---
# These codes should match the definitions used in the core engine
OP_QWEN2_RMSNORM = [300, 0, 0]
OP_QWEN2_ATTENTION = [300, 1, 0]
OP_QWEN2_MLP = [300, 2, 0]

# --- Helper Functions (Require PyTorch) ---

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Rotates half the hidden dimensions of the input tensor.
    Used in Rotary Positional Embeddings (RoPE).

    Args:
        x (torch.Tensor): Input tensor, expects shape like [..., seq_len, dim].

    Returns:
        torch.Tensor: Tensor with the second half rotated to the first
                      and the first half negated and moved to the second.
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for rotate_half.")
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None, # Unused, kept for potential signature compatibility
    unsqueeze_dim: int = 1
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies Rotary Positional Embeddings (RoPE) to query and key tensors.

    Args:
        q (torch.Tensor): Query tensor (e.g., [bsz, num_heads, seq_len, head_dim]).
        k (torch.Tensor): Key tensor (e.g., [bsz, num_kv_heads, seq_len, head_dim]).
        cos (torch.Tensor): Cosine components of RoPE ([bsz, 1, seq_len, head_dim]).
        sin (torch.Tensor): Sine components of RoPE ([bsz, 1, seq_len, head_dim]).
        position_ids (Optional[torch.Tensor]): Unused in this implementation.
        unsqueeze_dim (int): Dimension to unsqueeze cos/sin for broadcasting. Defaults to 1.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Rotated query and key tensors.
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for apply_rotary_pos_emb.")
    try:
        # Unsqueeze cos/sin if necessary (depends on input tensor shape)
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)

        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        return q_embed, k_embed
    except Exception as e:
        print(f"ERROR in apply_rotary_pos_emb: {e}")
        traceback.print_exc()
        raise # Re-raise after logging

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Repeats Key/Value heads for Grouped Query Attention (GQA).
    Expands tensor from [batch, num_kv_heads, seqlen, head_dim] to
    [batch, num_attn_heads, seqlen, head_dim].

    Args:
        hidden_states (torch.Tensor): Input K or V tensor.
        n_rep (int): Number of times to repeat each KV head (num_attn_heads // num_kv_heads).

    Returns:
        torch.Tensor: Expanded tensor.
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for repeat_kv.")
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states

    # Expand and reshape
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


# --- Veector Operations ---

def qwen2_rmsnorm(
    current_data: Optional[np.ndarray], **kw: Any
) -> Optional[np.ndarray]:
    """
    Performs Qwen2 RMS Normalization using PyTorch.

    Args:
        current_data (Optional[np.ndarray]): Input hidden states.
        **kw: Keyword arguments, must include:
            norm_weight (np.ndarray): The weight parameter (gamma) for RMSNorm.
            eps (float, optional): Epsilon value for numerical stability. Defaults to 1e-6.

    Returns:
        Optional[np.ndarray]: Normalized hidden states, or None on error.
    """
    if not TORCH_AVAILABLE:
        print("ERROR: qwen2_rmsnorm requires PyTorch.")
        return None
    if current_data is None:
        print("ERROR: qwen2_rmsnorm received None input.")
        return None

    norm_weight_np = kw.get('norm_weight')
    eps = float(kw.get('eps', 1e-6)) # Default epsilon

    if norm_weight_np is None:
        print("ERROR: qwen2_rmsnorm requires 'norm_weight'.")
        return None
    if not isinstance(norm_weight_np, np.ndarray):
        print("ERROR: qwen2_rmsnorm expects 'norm_weight' as np.ndarray.")
        return None

    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

    try:
        # Convert inputs to PyTorch tensors
        hidden_states_torch: torch.Tensor = torch.from_numpy(current_data).to(device)
        norm_weight_torch: torch.Tensor = torch.from_numpy(norm_weight_np).to(device)

        input_dtype_torch: torch.dtype = hidden_states_torch.dtype
        # Perform calculations in float32 for numerical stability
        hidden_states_fp32: torch.Tensor = hidden_states_torch.to(torch.float32)
        norm_weight_fp32: torch.Tensor = norm_weight_torch.to(torch.float32)

        # RMSNorm calculation
        variance = hidden_states_fp32.pow(2).mean(-1, keepdim=True)
        hidden_states_normalized = hidden_states_fp32 * torch.rsqrt(variance + eps)

        # Apply weight (gamma)
        output_torch_fp32 = norm_weight_fp32 * hidden_states_normalized

        # Cast back to original input dtype
        output_torch_casted = output_torch_fp32.to(input_dtype_torch)
        output_np = output_torch_casted.cpu().numpy()
        return output_np

    except Exception as e:
        print(f"ERROR during qwen2_rmsnorm execution: {e}")
        traceback.print_exc()
        return None

def qwen2_attention(
    current_data: Optional[np.ndarray], **kw: Any
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Performs Qwen2 Attention using EAGER (manual) PyTorch implementation.
    Includes QKV projection, RoPE, KV Cache update, manual attention calculation,
    and O-projection.

    Args:
        current_data (Optional[np.ndarray]): Input hidden states from RMSNorm.
        **kw: Keyword arguments, must include:
            q_weights, k_weights, v_weights, o_weights (np.ndarray): Weight matrices.
            q_bias, k_bias, v_bias (Optional[np.ndarray]): Bias vectors (can be None).
            position_ids (np.ndarray): Position IDs for RoPE.
            past_key (np.ndarray): Key tensor from previous state (KV cache).
            past_value (np.ndarray): Value tensor from previous state (KV cache).
            start_pos (int): Starting position index for updating the KV cache.
            layer_idx (int): Current layer index (for potential debugging/logging).
            num_heads (int): Number of attention heads.
            num_kv_heads (int): Number of key/value heads (for GQA).
            head_dim (int): Dimension of each attention head.
            rope_theta (float): Theta value for RoPE calculation.
            total_seq_len (int): Total sequence length including past KV cache.
            hidden_size (int): Dimension of the hidden state.

    Returns:
        Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
            A tuple containing:
            - final_output_np: Output of the attention block after O-projection.
            - updated_k_np: Updated Key tensor (KV cache).
            - updated_v_np: Updated Value tensor (KV cache).
            - attn_weights_np: Calculated attention weights (softmax output) as float32.
            - attn_output_pre_o_proj_np: Attention output before O-projection.
        Returns None on error.
    """
    if not TORCH_AVAILABLE:
        print("ERROR: qwen2_attention requires PyTorch.")
        return None
    if current_data is None:
        print("ERROR: qwen2_attention received None input (hidden_states).")
        return None

    # --- Argument Extraction and Validation ---
    hidden_states_np = current_data
    original_dtype_np = hidden_states_np.dtype # Store original numpy dtype

    required_keys = [
        "q_weights", "k_weights", "v_weights", "o_weights", "position_ids",
        "past_key", "past_value", "start_pos", "layer_idx", "num_heads",
        "num_kv_heads", "head_dim", "rope_theta", "total_seq_len", "hidden_size"
    ]
    optional_keys = ["q_bias", "k_bias", "v_bias"]
    args: Dict[str, Any] = {}
    missing_keys: List[str] = []

    for key in required_keys:
        val = kw.get(key)
        if val is None:
            missing_keys.append(key)
        args[key] = val
    for key in optional_keys:
        args[key] = kw.get(key) # Will be None if not provided

    if missing_keys:
        print(f"ERROR: qwen2_attention missing required arguments: {', '.join(missing_keys)}")
        return None

    # --- Setup Device and Dtypes ---
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Determine torch dtype based on input numpy dtype
    compute_dtype_torch: torch.dtype = torch.float16 if original_dtype_np == np.float16 else torch.float32

    try:
        # --- Convert Inputs to PyTorch Tensors ---
        hidden_states: torch.Tensor = torch.from_numpy(hidden_states_np).to(dtype=compute_dtype_torch, device=device)

        # Weights need transpose because F.linear expects (out, in)
        q_weights: torch.Tensor = torch.from_numpy(args["q_weights"]).to(dtype=compute_dtype_torch, device=device).T
        k_weights: torch.Tensor = torch.from_numpy(args["k_weights"]).to(dtype=compute_dtype_torch, device=device).T
        v_weights: torch.Tensor = torch.from_numpy(args["v_weights"]).to(dtype=compute_dtype_torch, device=device).T
        o_weights: torch.Tensor = torch.from_numpy(args["o_weights"]).to(dtype=compute_dtype_torch, device=device).T

        # Biases (optional)
        q_bias: Optional[torch.Tensor] = torch.from_numpy(args["q_bias"]).to(dtype=compute_dtype_torch, device=device) if args["q_bias"] is not None else None
        k_bias: Optional[torch.Tensor] = torch.from_numpy(args["k_bias"]).to(dtype=compute_dtype_torch, device=device) if args["k_bias"] is not None else None
        v_bias: Optional[torch.Tensor] = torch.from_numpy(args["v_bias"]).to(dtype=compute_dtype_torch, device=device) if args["v_bias"] is not None else None

        # Other inputs
        position_ids: torch.Tensor = torch.from_numpy(args["position_ids"]).to(device=device, dtype=torch.long)
        # Load KV cache with its original dtype
        past_key: torch.Tensor = torch.from_numpy(args["past_key"]).to(device=device)
        past_value: torch.Tensor = torch.from_numpy(args["past_value"]).to(device=device)

        # Cast numeric parameters
        start_pos: int = int(args["start_pos"])
        num_heads: int = int(args["num_heads"])
        num_kv_heads: int = int(args["num_kv_heads"])
        head_dim: int = int(args["head_dim"])
        rope_theta: float = float(args["rope_theta"])
        total_seq_len: int = int(args["total_seq_len"])
        hidden_size: int = int(args["hidden_size"])

        # --- Core Logic ---
        bsz, q_len, _ = hidden_states.size()

        # 1. QKV Projections
        query_states: torch.Tensor = F.linear(hidden_states, q_weights, q_bias)
        key_states: torch.Tensor = F.linear(hidden_states, k_weights, k_bias)
        value_states: torch.Tensor = F.linear(hidden_states, v_weights, v_bias)

        # Reshape for multi-head attention: [bsz, q_len, num_heads * head_dim] -> [bsz, num_heads, q_len, head_dim]
        query_states = query_states.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)

        # 2. Apply RoPE
        # Calculate frequencies in float32 for precision
        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
        # Ensure position_ids shape is [bsz, q_len]
        if position_ids.ndim == 1:
            position_ids = position_ids.unsqueeze(0)
        if position_ids.shape[0] != bsz:
            if position_ids.shape[0] == 1:
                position_ids = position_ids.expand(bsz, -1)
            else:
                raise ValueError(f"Incompatible position_ids batch size: {position_ids.shape[0]} vs hidden_states: {bsz}")
        # Ensure position_ids length matches query length
        if position_ids.shape[1] != q_len:
             # This might happen during generation with KV cache. Use the relevant slice.
             if position_ids.shape[1] > q_len:
                 # Assuming position_ids contains positions for the *current* tokens
                 pos_ids_slice = position_ids[:, -q_len:]
             else: # Should not happen if start_pos is correct
                 raise ValueError(f"position_ids length ({position_ids.shape[1]}) < query length ({q_len})")
        else:
             pos_ids_slice = position_ids

        with torch.autocast(device_type=device, enabled=False): # Force float32 for trig functions
            freqs = torch.einsum("bi,j->bij", pos_ids_slice.float(), inv_freq.float())
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        # Apply RoPE in compute dtype
        query_states_rot, key_states_rot = apply_rotary_pos_emb(
            query_states, key_states, cos.to(compute_dtype_torch), sin.to(compute_dtype_torch), unsqueeze_dim=1
        )

        # 3. Update KV Cache
        seq_len_cached = past_key.shape[2]
        if start_pos + q_len > seq_len_cached:
            raise ValueError(f"KV Cache update out of bounds! start={start_pos}, q_len={q_len}, cache_len={seq_len_cached}")
        # Use clone to avoid modifying the input cache directly if it's used elsewhere
        updated_key = past_key.clone()
        updated_value = past_value.clone()
        # Ensure dtypes match before assignment
        if updated_key.dtype != key_states_rot.dtype: key_states_rot = key_states_rot.to(updated_key.dtype)
        if updated_value.dtype != value_states.dtype: value_states = value_states.to(updated_value.dtype)
        # Update slice
        updated_key[:, :, start_pos : start_pos + q_len, :] = key_states_rot
        updated_value[:, :, start_pos : start_pos + q_len, :] = value_states

        # 4. Prepare for Attention (GQA)
        num_key_value_groups: int = num_heads // num_kv_heads
        actual_kv_len: int = total_seq_len # Use the total sequence length for slicing cache

        # Slice updated cache to actual length
        key_states_sliced: torch.Tensor = updated_key[:, :, :actual_kv_len, :]
        value_states_sliced: torch.Tensor = updated_value[:, :, :actual_kv_len, :]

        # Repeat KV heads if necessary
        key_states_repeated: torch.Tensor = repeat_kv(key_states_sliced, num_key_value_groups)
        value_states_repeated: torch.Tensor = repeat_kv(value_states_sliced, num_key_value_groups)

        # 5. Manual Attention Calculation
        # 5a. Scores (Q @ K.T * scale) - use float32 for stability
        scaling = float(head_dim**-0.5)
        attn_weights_fp32 = torch.matmul(
            query_states_rot.to(torch.float32),
            key_states_repeated.to(torch.float32).transpose(2, 3)
        ) * scaling

        # 5b. Causal Mask Application
        # Create mask [q_len, actual_kv_len]
        causal_mask = torch.triu(torch.ones((q_len, actual_kv_len), device=device, dtype=torch.bool), diagonal=1)
        # Expand mask for broadcasting: [1, 1, q_len, actual_kv_len]
        causal_mask_expanded = causal_mask[None, None, :, :]
        # Apply mask (set masked positions to a large negative number)
        attn_weights_fp32 = torch.where(
            causal_mask_expanded, torch.finfo(torch.float32).min, attn_weights_fp32
        )

        # 5c. Softmax - use float32
        attn_weights_softmax_fp32 = F.softmax(attn_weights_fp32, dim=-1, dtype=torch.float32)

        # 5d. Weighted Value (Attention @ V) - use float32
        attn_output_pre_o_proj_fp32 = torch.matmul(
            attn_weights_softmax_fp32, value_states_repeated.to(torch.float32)
        )

        # Cast results back to compute_dtype
        attn_output_pre_o_proj = attn_output_pre_o_proj_fp32.to(compute_dtype_torch)
        # Keep weights as float32 for potential analysis
        attn_weights_softmax = attn_weights_softmax_fp32

        # 6. Reshape and O-Projection
        # Reshape: [bsz, num_heads, q_len, head_dim] -> [bsz, q_len, num_heads, head_dim] -> [bsz, q_len, hidden_size]
        attn_output = attn_output_pre_o_proj.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, hidden_size)

        # O-projection (bias is usually False for O-proj in transformers)
        attn_output_proj = F.linear(attn_output, o_weights, None)

        # --- Convert Results back to NumPy ---
        # Final output in original numpy dtype
        attn_output_proj_np = attn_output_proj.cpu().numpy().astype(original_dtype_np)
        # KV cache in its original dtype
        updated_k_np = updated_key.cpu().numpy()
        updated_v_np = updated_value.cpu().numpy()
        # Intermediate results
        attn_weights_np = attn_weights_softmax.cpu().numpy().astype(np.float32)
        attn_output_pre_o_proj_np = attn_output_pre_o_proj.cpu().numpy().astype(original_dtype_np)

        # Return the tuple
        return (
            attn_output_proj_np,
            updated_k_np,
            updated_v_np,
            attn_weights_np,
            attn_output_pre_o_proj_np
        )

    except Exception as e:
        print(f"ERROR during qwen2_attention execution: {e}")
        traceback.print_exc()
        return None

def qwen2_mlp(
    current_data: Optional[np.ndarray], **kw: Any
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Performs Qwen2 MLP (Feed-Forward Network) block using PyTorch.
    Includes Gate, Up, and Down projections with SiLU activation.

    Args:
        current_data (Optional[np.ndarray]): Input hidden states from Attention block.
        **kw: Keyword arguments, must include:
            gate_weights (np.ndarray): Weight matrix for the gate projection.
            up_weights (np.ndarray): Weight matrix for the up projection.
            down_weights (np.ndarray): Weight matrix for the down projection.
            hidden_act (str): Name of the activation function (e.g., 'silu').

    Returns:
        Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
            A tuple containing:
            - final_output_np: Output of the MLP block (before residual add).
            - gate_proj_np: Output of the gate projection.
            - up_proj_np: Output of the up projection.
            - activated_np: Output after activation and element-wise multiplication.
        Returns None on error.
    """
    if not TORCH_AVAILABLE:
        print("ERROR: qwen2_mlp requires PyTorch.")
        return None
    if current_data is None:
        print("ERROR: qwen2_mlp received None input.")
        return None

    # --- Argument Extraction and Validation ---
    gate_weights_np = kw.get('gate_weights')
    up_weights_np = kw.get('up_weights')
    down_weights_np = kw.get('down_weights')
    hidden_act_name = kw.get('hidden_act')

    required_weights = {
        "gate_weights": gate_weights_np,
        "up_weights": up_weights_np,
        "down_weights": down_weights_np
    }
    for name, weight in required_weights.items():
        if weight is None:
            print(f"ERROR: qwen2_mlp missing required weight '{name}'.")
            return None
    if not isinstance(hidden_act_name, str) or not hidden_act_name:
        print(f"ERROR: qwen2_mlp requires a valid string for 'hidden_act', received: {hidden_act_name}")
        return None

    # --- Setup Device and Dtypes ---
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    original_dtype_np = current_data.dtype
    compute_dtype_torch: torch.dtype = torch.float16 if original_dtype_np == np.float16 else torch.float32

    try:
        # --- Convert Inputs to PyTorch Tensors ---
        hidden_states: torch.Tensor = torch.from_numpy(current_data).to(dtype=compute_dtype_torch, device=device)

        # Weights need transpose
        gate_weights: torch.Tensor = torch.from_numpy(gate_weights_np).to(dtype=compute_dtype_torch, device=device).T
        up_weights: torch.Tensor = torch.from_numpy(up_weights_np).to(dtype=compute_dtype_torch, device=device).T
        down_weights: torch.Tensor = torch.from_numpy(down_weights_np).to(dtype=compute_dtype_torch, device=device).T

        # --- Get Activation Function ---
        try:
            # Use the imported ACT2FN
            act_fn = ACT2FN[hidden_act_name]
        except KeyError:
            print(f"ERROR: Unknown activation function name '{hidden_act_name}'.")
            return None
        except Exception as act_e:
             print(f"ERROR getting activation function '{hidden_act_name}': {act_e}")
             return None

        # --- MLP Calculation ---
        gate_proj: torch.Tensor = F.linear(hidden_states, gate_weights)
        up_proj: torch.Tensor = F.linear(hidden_states, up_weights)

        # Apply activation and multiply
        activated: torch.Tensor = act_fn(gate_proj) * up_proj

        # Down projection
        down_proj: torch.Tensor = F.linear(activated, down_weights)

        # Final output tensor (before residual add)
        output_torch: torch.Tensor = down_proj

        # --- Convert Results back to NumPy ---
        final_output_np = output_torch.cpu().numpy().astype(original_dtype_np)
        # Intermediate results cast back to original dtype
        gate_proj_np = gate_proj.cpu().numpy().astype(original_dtype_np)
        up_proj_np = up_proj.cpu().numpy().astype(original_dtype_np)
        activated_np = activated.cpu().numpy().astype(original_dtype_np)

        # Return the tuple
        return (final_output_np, gate_proj_np, up_proj_np, activated_np)

    except Exception as e:
        print(f"ERROR during qwen2_mlp execution: {e}")
        traceback.print_exc()
        return None


# --- Operation Dictionary ---
# Maps Veector operation codes to the corresponding functions in this module.
qwen2_operations: Dict[Tuple[int, int, int], callable] = {
    tuple(OP_QWEN2_RMSNORM): qwen2_rmsnorm,
    tuple(OP_QWEN2_ATTENTION): qwen2_attention, # Now points to the eager version
    tuple(OP_QWEN2_MLP): qwen2_mlp,
}

# --- Module Load Confirmation ---
print(f"INFO: Veector Qwen2 Ops Module Loaded (Version: {QWEN2_OPS_VERSION}).")
print(f"INFO: Found {len(qwen2_operations)} operations.")
# Log the specific function mapped to the attention operation code
if tuple(OP_QWEN2_ATTENTION) in qwen2_operations:
    attention_func_name = qwen2_operations[tuple(OP_QWEN2_ATTENTION)].__name__
    print(f"INFO: OP_QWEN2_ATTENTION ({OP_QWEN2_ATTENTION}) is mapped to: {attention_func_name}")
else:
    print(f"WARN: OP_QWEN2_ATTENTION ({OP_QWEN2_ATTENTION}) not found in qwen2_operations dictionary!")
# --- End of Module ---
# Note: This module is designed to be used with the Veector engine.
# It should not be run as a standalone script.
# The functions are expected to be called with the appropriate arguments
# from the Veector engine, which handles the input/output and context.
# The module is optimized for performance and should be used in a
# multi-threaded environment where possible.
# The module is not responsible for managing the lifecycle of the tensors
# or their memory. It is assumed that the caller will handle tensor
# management and cleanup as necessary.
# The module is designed to be compatible with PyTorch and requires
# PyTorch to be installed in the environment. It is not compatible with
# TensorFlow or other deep learning frameworks. The module is designed
# to be used in a high-performance computing environment and is optimized
# for speed and efficiency. It is not designed for educational purposes
# or for use in low-performance computing environments. The module is
# designed to be used in a production environment and is not intended for
# use in research or academic settings. The module is not responsible for
# managing the lifecycle of the tensors or their memory. It is assumed
# that the caller will handle tensor management and cleanup as necessary.