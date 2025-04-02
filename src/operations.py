# FILE: operations.py
# Version: 0.7.3 (Corrected RoPE getter warning message)

import numpy as np
# import scipy # Not currently used
# from scipy.signal import convolve2d # Keep if convolution needed later
from typing import Dict, List, Any, Optional, Tuple, Union

# --- Version ---
OPERATIONS_VERSION = "0.7.3"
# --- End Version ---

# --- Getter functions for Qwen2 specific data ---
def get_q_rot(data: Tuple, **kw) -> Optional[np.ndarray]:
    """Extracts q_rot (first element) from input tuple."""
    if isinstance(data, tuple) and len(data) >= 1:
        res = data[0]
        return res
    # <<< ИЗМЕНЕНО: Текст предупреждения >>>
    print(f"WARN: get_q_rot expected tuple input with at least 1 element, got {type(data)}")
    return None

def get_k_rot(data: Tuple, **kw) -> Optional[np.ndarray]:
    """Extracts k_rot (second element) from input tuple."""
    if isinstance(data, tuple) and len(data) >= 2:
        res = data[1]
        return res
    # <<< ИЗМЕНЕНО: Текст предупреждения >>>
    print(f"WARN: get_k_rot expected tuple input with at least 2 elements, got {type(data)}")
    return None

# --- Helper for checking inputs ---
def _check_none(*args):
    """Returns True if any argument is None."""
    if len(args) == 1 and isinstance(args[0], (list, tuple)):
        args = args[0]
    return any(a is None for a in args)

# --- Basic Math & Logic (Formatted, checked None checks) ---

def mod(x, y):
    """Returns x % y."""
    if _check_none(x, y): return None
    y_arr = np.asarray(y)
    if np.any(y_arr == 0): print("Error in mod: Division by zero."); return None
    try: return np.mod(x, y)
    except Exception as e: print(f"Error in mod: {e}"); return None

def floor(x):
    """Element-wise floor."""
    if _check_none(x): return None
    try: return np.floor(x)
    except Exception as e: print(f"Error in floor: {e}"); return x

def ceil(x):
    """Element-wise ceil."""
    if _check_none(x): return None
    try: return np.ceil(x)
    except Exception as e: print(f"Error in ceil: {e}"); return x

def arcsin(x):
    """Element-wise arcsin."""
    if _check_none(x): return None
    try: x_clipped = np.clip(x, -1.0, 1.0); return np.arcsin(x_clipped)
    except Exception as e: print(f"Error in arcsin: {e}"); return x

def arccos(x):
    """Element-wise arccos."""
    if _check_none(x): return None
    try: x_clipped = np.clip(x, -1.0, 1.0); return np.arccos(x_clipped)
    except Exception as e: print(f"Error in arccos: {e}"); return x

def arctan(x):
    """Element-wise arctan."""
    if _check_none(x): return None
    try: return np.arctan(x)
    except Exception as e: print(f"Error in arctan: {e}"); return x

def xor(x, y):
    """Logical XOR (element-wise)."""
    if _check_none(x, y): return None
    try: return np.logical_xor(x, y)
    except Exception as e: print(f"Error in xor: {e}"); return None

def nand(x, y):
    """Logical NAND (NOT AND, element-wise)."""
    if _check_none(x, y): return None
    try: return np.logical_not(np.logical_and(x, y))
    except Exception as e: print(f"Error in nand: {e}"); return None

def nor(x, y):
    """Logical NOR (NOT OR, element-wise)."""
    if _check_none(x, y): return None
    try: return np.logical_not(np.logical_or(x, y))
    except Exception as e: print(f"Error in nor: {e}"); return None

def inverse(matrix):
    """Matrix inverse."""
    if _check_none(matrix): return None
    try: return np.linalg.inv(matrix)
    except np.linalg.LinAlgError: print(f"Error in inverse: Matrix is singular."); return None
    except Exception as e: print(f"Error in inverse: {e}"); return None

def trace(matrix):
    """Matrix trace."""
    if _check_none(matrix): return None
    try: return np.trace(matrix)
    except Exception as e: print(f"Error in trace: {e}"); return None

def random_uniform(min_val=0.0, max_val=1.0):
    """Uniform random number."""
    try: return np.random.uniform(min_val, max_val)
    except Exception as e: print(f"Error in random_uniform: {e}"); return 0.0

def random_normal(mu=0.0, sigma=1.0):
    """Normal random number."""
    try: return np.random.normal(mu, sigma)
    except Exception as e: print(f"Error in random_normal: {e}"); return 0.0

def median(x):
    """Array median."""
    if _check_none(x): return None
    try: return np.median(x)
    except Exception as e: print(f"Error in median: {e}"); return None

def softmax(x, axis=-1):
    """Softmax with numerical stabilization."""
    if _check_none(x): return None
    try:
        x_np = np.asarray(x, dtype=np.float32)
        exp_x = np.exp(x_np - np.max(x_np, axis=axis, keepdims=True))
        sum_exp_x = np.sum(exp_x, axis=axis, keepdims=True)
        result = exp_x / sum_exp_x
        return result.astype(x.dtype if hasattr(x, 'dtype') else np.float32)
    except Exception as e: print(f"Error in softmax: {e}"); return x

def matrix_determinant(a):
    """Matrix determinant."""
    if _check_none(a): return None
    try: return np.linalg.det(a)
    except Exception as e: print(f"Error in matrix_determinant: {e}"); return None

def matrix_eigenvalues(a):
    """Matrix eigenvalues."""
    if _check_none(a): return None
    try: return np.linalg.eigvals(a)
    except Exception as e: print(f"Error in matrix_eigenvalues: {e}"); return None

def convolution(data, kernel, bias=None, **kwargs):
    """Basic 2D convolution placeholder."""
    print("WARN: Using basic convolution placeholder.")
    if _check_none(data, kernel): return None
    try:
        # Basic implementation for 2D grayscale, needs proper handling
        data_2d = np.asarray(data); kernel_2d = np.asarray(kernel)
        if data_2d.ndim > 2: data_2d = data.mean(axis=tuple(range(data.ndim - 2))) # Crude reduction
        if kernel_2d.ndim > 2: kernel_2d = kernel.mean(axis=tuple(range(kernel.ndim - 2)))
        if data_2d.ndim != 2 or kernel_2d.ndim != 2: raise ValueError("Conv placeholder needs 2D arrays")
        result = convolve2d(data_2d, kernel_2d, mode='same', boundary='symm')
        if bias is not None: result = result + np.asarray(bias).item()
        return result
    except Exception as e: print(f"Error in convolution placeholder: {e}"); return None

def transpose(a, axes=None):
    """Transpose matrix/tensor."""
    if _check_none(a): return None
    try: return np.transpose(a, axes=axes)
    except Exception as e: print(f"Error in transpose: {e}"); return a

def mean(x, axis=None, keepdims=False):
    """Mean value."""
    if _check_none(x): return None
    try: return np.mean(x, axis=axis, keepdims=keepdims)
    except Exception as e: print(f"Error in mean: {e}"); return None

def std_dev(x, axis=None, keepdims=False):
    """Standard deviation."""
    if _check_none(x): return None
    try: return np.std(x, axis=axis, keepdims=keepdims)
    except Exception as e: print(f"Error in std_dev: {e}"); return None

# --- NN Related Operations (Implementations & Placeholders) ---

def add(input_a: Optional[np.ndarray], input_b: Optional[np.ndarray], **kwargs) -> Optional[np.ndarray]:
     """Element-wise addition for residual connections. Handles None inputs."""
     if input_a is None and input_b is None:
          print("Error in add: Both inputs are None.")
          return None
     if input_a is None:
          return input_b # Return the non-None input
     if input_b is None:
          return input_a # Return the non-None input
     try:
         # NumPy handles broadcasting
         return np.add(input_a, input_b)
     except ValueError as ve: # Catch broadcasting errors
          print(f"Error in add (ValueError - Broadcasting? Shapes: {getattr(input_a,'shape','N/A')}, {getattr(input_b,'shape','N/A')}): {ve}")
          return None
     except Exception as e:
          print(f"Error in add (shapes: {getattr(input_a,'shape','N/A')}, {getattr(input_b,'shape','N/A')}): {e}")
          return None

def layer_normalization(data: Optional[np.ndarray], *, norm_weight: Optional[np.ndarray] = None, norm_bias: Optional[np.ndarray] = None, eps: float = 1e-5, **kwargs) -> Optional[np.ndarray]:
    
    """
    Performs Layer Normalization (RMSNorm style, compatible with Qwen2).
    Applies learned scale (weight/gamma) and optional shift (bias/beta).
    Expects weight and bias as keyword arguments named 'norm_weight', 'norm_bias'.
    """
    if _check_none(data):
        print("Error in layer_normalization: input data is None")
        return None
    try:
        x = np.asarray(data, dtype=np.float32) # Use float32 for calculation stability
        # Calculate RMS over the last dimension (hidden size)
        variance = np.mean(np.square(x), axis=-1, keepdims=True)
        inv_rms = np.reciprocal(np.sqrt(variance + float(eps)))
        x_norm = x * inv_rms

        # Apply learned weight (gamma)
        if norm_weight is not None:
            gamma = np.asarray(norm_weight, dtype=np.float32)
            # Ensure gamma shape is broadcastable (e.g., (hidden_size,))
            if x_norm.shape[-1] != gamma.shape[-1]:
                 raise ValueError(f"LayerNorm weight shape {gamma.shape} not compatible with input hidden dim {x_norm.shape[-1]}")
            x_norm = x_norm * gamma

        # Apply learned bias (beta) - Qwen2 RMSNorm might NOT use bias.
        if norm_bias is not None:
             beta = np.asarray(norm_bias, dtype=np.float32)
             if x_norm.shape[-1] != beta.shape[-1]:
                  raise ValueError(f"LayerNorm bias shape {beta.shape} not compatible with input hidden dim {x_norm.shape[-1]}")
             x_norm = x_norm + beta

        original_dtype = data.dtype if hasattr(data, 'dtype') else np.float32
        # print(f"DEBUG LN: Input dtype {data.dtype}, Output dtype {original_dtype}") # Debug
        return x_norm.astype(original_dtype) # Cast back
    except Exception as e:
        print(f" Error in layer_normalization: {e}")
        import traceback
        traceback.print_exc()
        return None

def matrix_multiply(a: Optional[np.ndarray], weights: Optional[np.ndarray], bias: Optional[np.ndarray] = None, **kwargs) -> Optional[np.ndarray]:
    """ Performs matrix multiplication (a @ weights) + optional bias. """
    if _check_none(a, weights):
        print("Error in matrix_multiply: Input or weights are None.")
        return None
    try:
        a_np = np.asarray(a)
        w_np = np.asarray(weights)
        # Flexible handling for different dimensions (e.g., batch, sequence)
        # Assume last dim of 'a' matches first dim of 'weights'
        if a_np.shape[-1] != w_np.shape[0]:
            raise ValueError(f"Incompatible shapes for matmul: {a_np.shape} @ {w_np.shape}")

        result = np.matmul(a_np, w_np) # Use matmul for more general high-dim support

        if bias is not None:
            b_np = np.asarray(bias)
            # Bias shape must match the last dimension of the result
            if result.shape[-1:] != b_np.shape:
                 # Allow broadcasting if bias is 1D and matches last dim
                 if b_np.ndim == 1 and result.shape[-1] == b_np.shape[0]:
                      result = result + b_np # Broadcast bias
                 else:
                      raise ValueError(f"Incompatible shapes for bias add: {result.shape} vs {b_np.shape}")
            else:
                 result = result + b_np # Direct addition if shapes match perfectly

        return result
    except ValueError as e:
        a_shape = getattr(a, 'shape', 'N/A'); w_shape = getattr(weights, 'shape', 'N/A'); b_shape = getattr(bias,'shape','N/A')
        print(f"Error in matrix_multiply: shapes {a_shape}, {w_shape}, bias {b_shape}. Error: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error in matrix_multiply: {e}")
        return None

def silu(data: Optional[np.ndarray], **kwargs) -> Optional[np.ndarray]:
     """ SiLU activation function (Sigmoid * x). """
     if _check_none(data):
          return None
     try:
          x = np.asarray(data)
          x_f32 = x.astype(np.float32)
          sigmoid_x = 1 / (1 + np.exp(-x_f32))
          result = x_f32 * sigmoid_x
          return result.astype(x.dtype) # Cast back
     except Exception as e:
          print(f" Error in silu: {e}")
          return None

# --- Activations (Unchanged implementations, formatted) ---
def relu(x):
    """ReLU activation."""
    if _check_none(x): return None
    try: return np.maximum(0, x)
    except Exception as e: print(f"Error in relu: {e}"); return x

def sigmoid(x):
    """Sigmoid activation."""
    if _check_none(x): return None
    try:
        x_f32 = np.asarray(x, dtype=np.float32); res_f32 = 1 / (1 + np.exp(-x_f32))
        return res_f32.astype(x.dtype if hasattr(x, 'dtype') else np.float32)
    except Exception as e: print(f"Error in sigmoid: {e}"); return x

def leaky_relu(x, alpha=0.01):
    """Leaky ReLU activation."""
    if _check_none(x): return None
    try: return np.maximum(alpha * x, x)
    except Exception as e: print(f"Error in leaky_relu: {e}"); return x

def gelu(x):
    """GELU activation (approximation)."""
    if _check_none(x): return None
    try:
        x_f32 = np.asarray(x, dtype=np.float32)
        res_f32 = 0.5 * x_f32 * (1 + np.tanh(np.sqrt(2/np.pi) * (x_f32 + 0.044715 * x_f32**3)))
        return res_f32.astype(x.dtype if hasattr(x, 'dtype') else np.float32)
    except Exception as e: print(f"Error in gelu: {e}"); return x

# --- Other NN Utils (Unchanged logic, formatted) ---
def dropout(data, *, rate=0.1, is_training=False, **kwargs):
    """Dropout regularization."""
    if _check_none(data): return None
    if not is_training or rate <= 0 or rate >= 1: return data
    try: keep_prob = 1.0 - rate; mask = np.random.binomial(1, keep_prob, size=data.shape); return (data * mask) / keep_prob
    except Exception as e: print(f" Error in dropout: {e}"); return data

def normalize(data):
    """Normalize data to [0, 1] range."""
    if _check_none(data): return None
    try: data_min = np.min(data); data_max = np.max(data); denom = data_max - data_min; return (data - data_min) / (denom if denom > 1e-8 else 1e-8)
    except Exception as e: print(f"Error in normalize: {e}"); return data

def interpolate(data, new_length):
    """Linear interpolation."""
    if _check_none(data) or not hasattr(data, '__len__') or new_length is None: return None
    try: old_indices = np.arange(len(data)); new_indices = np.linspace(0, len(data)-1, int(new_length)); return np.interp(new_indices, old_indices, data)
    except Exception as e: print(f"Error in interpolate: {e}"); return data

def batch_norm(data, **kwargs):
    """Basic Batch Normalization (placeholder)."""
    if _check_none(data): return None
    try: x = np.asarray(data); mean_x = np.mean(x, axis=0); std_x = np.std(x, axis=0); return (x - mean_x) / (std_x + 1e-5)
    except Exception as e: print(f" Error in batch_norm: {e}"); return data

def exponential_smoothing(data, alpha=0.5):
    """Exponential smoothing."""
    if _check_none(data) or not isinstance(data, (list, np.ndarray)): return None
    try: data_arr = np.array(data); smoothed = np.zeros_like(data_arr); smoothed[0] = data_arr[0]
    except IndexError: return data # Handle empty array?
    except Exception as e: print(f"Error initializing smoothing: {e}"); return data
    try:
        for i in range(1, len(data_arr)): smoothed[i] = alpha * data_arr[i] + (1 - alpha) * smoothed[i-1]
        return smoothed
    except Exception as e: print(f"Error in exponential_smoothing loop: {e}"); return data

# --- Placeholders for Complex Transformer Ops ---

def embedding_lookup(token_ids: np.ndarray, *, embedding_matrix: np.ndarray, **kw) -> Optional[np.ndarray]:
    """ Looks up embeddings for token IDs using numpy indexing. """
    if _check_none(token_ids, embedding_matrix):
        return None
    print(f"INFO: Executing embedding_lookup (IDs shape: {token_ids.shape}, Matrix shape: {embedding_matrix.shape})")
    try:
        ids = np.asarray(token_ids).astype(np.int64)
        embeddings = np.asarray(embedding_matrix)
        vocab_size = embeddings.shape[0]
        # Check bounds
        if np.any(ids >= vocab_size) or np.any(ids < 0):
             max_id = np.max(ids); min_id = np.min(ids)
             print(f"Error: Token IDs out of bounds [0, {vocab_size-1}]. Got min/max: {min_id}/{max_id}")
             return None
        output = embeddings[ids] # NumPy fancy indexing
        return output.astype(embeddings.dtype)
    except IndexError as ie:
        print(f"Error: Embedding lookup IndexError: {ie}")
    except Exception as e:
        print(f" Error in embedding_lookup: {e}")
    return None

def apply_rope(q: np.ndarray, k: np.ndarray, *, position_ids: np.ndarray, **kw) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """ Placeholder for Rotary Positional Embedding (RoPE). """
    print(f"WARN: Placeholder op: apply_rope (Returning Q, K unchanged)")
    # TODO: Implement RoPE calculation. Needs model config (rope_theta), sin/cos cache.
    if _check_none(q, k):
        return None, None
    return q, k # Return unchanged

def scaled_dot_product_attention(query: np.ndarray, key: np.ndarray, value: np.ndarray, mask: Optional[np.ndarray] = None, **kw) -> Optional[np.ndarray]:
    """ Basic Scaled Dot-Product Attention with GQA repetition & masking. """
    print(f"WARN: Basic scaled_dot_product_attention (No FlashAttention/Optimizations)")
    if _check_none(query, key, value):
        return None
    try:
        # Assume query: (batch, seq_len_q, num_heads_q, head_dim)
        # Assume key/value: (batch, seq_len_kv, num_heads_kv, head_dim)
        q_np = np.asarray(query, dtype=np.float32)
        k_np = np.asarray(key, dtype=np.float32)
        v_np = np.asarray(value, dtype=np.float32)

        # Handle GQA: Repeat K and V heads if num_kv_heads < num_q_heads
        num_heads_q = q_np.shape[-2]
        num_heads_kv = k_np.shape[-2]
        if num_heads_kv != num_heads_q:
             if num_heads_q % num_heads_kv == 0:
                  repeats = num_heads_q // num_heads_kv
                  k_np = np.repeat(k_np, repeats, axis=-2) # Repeat along head dimension
                  v_np = np.repeat(v_np, repeats, axis=-2)
                  print(f"  Applied GQA repetition ({repeats}x) for K/V heads.")
             else:
                  raise ValueError(f"num_heads_q ({num_heads_q}) must be divisible by num_heads_kv ({num_heads_kv}) for GQA")

        # Q @ K.T / sqrt(d_k)
        k_transposed = np.swapaxes(k_np, -1, -2)
        d_k = q_np.shape[-1]
        if d_k == 0:
            raise ValueError("Query head dimension cannot be zero.")
        scale = 1.0 / np.sqrt(d_k).astype(q_np.dtype)
        # Use matmul for broadcasting across batch/head dims
        # (batch, q_len, num_q_heads, dim) @ (batch, kv_len, num_q_heads, dim).T -> (batch, q_len, num_q_heads, kv_len) ?? No.
        # Need (batch, num_q_heads, q_len, dim) @ (batch, num_q_heads, dim, kv_len) -> (batch, num_q_heads, q_len, kv_len)
        q_perm = np.transpose(q_np, (0, 2, 1, 3)) # (b, h, q_len, dim)
        k_perm = np.transpose(k_transposed, (0, 2, 3, 1)) # (b, h, dim, kv_len)
        scores = np.matmul(q_perm, k_perm) * scale

        if mask is not None:
            mask_np = np.asarray(mask)
            # Add dimensions for batch/heads if necessary
            while mask_np.ndim < scores.ndim: mask_np = np.expand_dims(mask_np, axis=0)
            scores = np.where(mask_np == 0, np.finfo(scores.dtype).min, scores)

        attention_weights = softmax(scores, axis=-1) # Softmax over kv_len dimension

        # Weights @ V
        # Need V as (batch, num_q_heads, kv_len, dim)
        v_perm = np.transpose(v_np, (0, 2, 1, 3))
        # (batch, num_q_heads, q_len, kv_len) @ (batch, num_q_heads, kv_len, dim) -> (batch, num_q_heads, q_len, dim)
        output_perm = np.matmul(attention_weights.astype(v_perm.dtype), v_perm)

        # Transpose back to (batch, q_len, num_q_heads, dim) -> (batch, q_len, hidden_size)
        output = np.transpose(output_perm, (0, 2, 1, 3))
        # Reshape last two dims: (batch, q_len, num_q_heads * head_dim)
        output = output.reshape(output.shape[0], output.shape[1], -1)

        return output.astype(query.dtype if hasattr(query, 'dtype') else np.float16)

    except ValueError as ve: print(f" Error in SDPA (ValueError): {ve}"); return None
    except Exception as e: print(f" Error in SDPA: {e}"); return None

def multi_head_attention(*args, **kw):
     """ Placeholder - logic should be in ops_sequence using smaller ops. """
     print(f"WARN: Placeholder op: multi_head_attention called directly (NOP)")
     if args: return args[0] # Return first positional arg (likely query/data)
     return kw.get("data")

# --- Other utility ops (Unchanged) ---
def causal_mask(size):
    try: mask = np.triu(np.ones((1, size, size)), k=1); return (1.0 - mask).astype(bool)
    except Exception as e: print(f"Error in causal_mask: {e}"); return None

def masked_fill(tensor, mask, value):
    if _check_none(tensor, mask): return None
    try: return np.where(mask, value, tensor)
    except Exception as e: print(f"Error in masked_fill: {e}"); return tensor

def linear(data, *, weights, bias=None, **kwargs):
    """ Placeholder for Linear Layer (MatMul + Bias Add). """
    print(f"WARN: Placeholder op: linear (using matrix_multiply)")
    # Используем существующий matrix_multiply, который теперь умеет добавлять bias
    return matrix_multiply(data, weights=weights, bias=bias)

def ffn_mlp(data, *, gate_weights, up_weights, down_weights, gate_bias=None, up_bias=None, down_bias=None, **kwargs) -> Optional[np.ndarray]:
    """ Placeholder for the core FFN MLP logic (SwiGLU style). """
    print(f"WARN: Placeholder op: ffn_mlp")
    if _check_none(data, gate_weights, up_weights, down_weights): return None
    try:
        # 1. Gate = Linear(data, gate_weights, gate_bias)
        gate = linear(data, weights=gate_weights, bias=gate_bias)
        if gate is None: return None
        # 2. Up   = Linear(data, weights=up_weights, bias=up_bias)
        up = linear(data, weights=up_weights, bias=up_bias)
        if up is None: return None
        # 3. Activated = silu(Gate) * Up
        activated = silu(gate) * up # Assumes silu is implemented
        # 4. Output = Linear(activated, weights=down_weights, bias=down_bias)
        output = linear(activated, weights=down_weights, bias=down_bias)
        return output
    except Exception as e:
        print(f"Error in ffn_mlp placeholder: {e}")
        return None

def linear(data: Optional[np.ndarray], *, weights: np.ndarray, bias: Optional[np.ndarray] = None, **kwargs) -> Optional[np.ndarray]:
    """ Выполняет Linear = data @ weights + bias. Ожидает 'weights', 'bias' в kw. """
    print(f"INFO: Executing linear (using matrix_multiply)")
    if _check_none(data, weights):
        return None
    # Используем существующий matrix_multiply, который умеет добавлять bias
    return matrix_multiply(data, weights=weights, bias=bias)

def apply_rope(q: np.ndarray, k: np.ndarray, *, position_ids: np.ndarray, **kw) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """ Placeholder for Rotary Positional Embedding (RoPE). """
    print(f"WARN: Placeholder op: apply_rope (Returning Q, K unchanged)")
    # TODO: Implement RoPE calculation based on Qwen2 config (rope_theta)
    if _check_none(q, k, position_ids): return None, None
    # Возвращаем без изменений
    return q, k

def reshape_heads(data: np.ndarray, num_heads: int, head_dim: int, **kw) -> Optional[np.ndarray]:
    """ Placeholder: Reshapes tensor for multi-head attention. """
    print(f"WARN: Placeholder op: reshape_heads")
    if _check_none(data): return None
    try:
        # Example: (batch, seq_len, num_heads * head_dim) -> (batch, num_heads, seq_len, head_dim)
        batch_size, seq_len, _ = data.shape
        reshaped = data.reshape(batch_size, seq_len, num_heads, head_dim)
        transposed = np.transpose(reshaped, (0, 2, 1, 3)) # (batch, num_heads, seq_len, head_dim)
        return transposed
    except Exception as e:
        print(f"Error in reshape_heads placeholder: {e}")
        return None

def repeat_kv_heads(data: np.ndarray, repeats: int, **kw) -> Optional[np.ndarray]:
    """ Placeholder: Repeats K/V heads for Grouped Query Attention (GQA). """
    print(f"WARN: Placeholder op: repeat_kv_heads (factor: {repeats})")
    if _check_none(data): return None
    try:
        # Example: data shape (batch, num_kv_heads, seq_len, head_dim)
        # Repeats along the head dimension (axis=1)
        return np.repeat(data, repeats, axis=1)
    except Exception as e:
        print(f"Error in repeat_kv_heads placeholder: {e}")
        return None

def scaled_dot_product_attention(query: np.ndarray, key: np.ndarray, value: np.ndarray, mask: Optional[np.ndarray] = None, **kw) -> Optional[np.ndarray]:
    """ Scaled Dot-Product Attention (Basic implementation with masking). """
    # [Используем реализацию из предыдущего ответа]
    print(f"INFO: Executing scaled_dot_product_attention")
    if _check_none(query, key, value): return None
    try:
        q_np = np.asarray(query, dtype=np.float32)
        k_np = np.asarray(key, dtype=np.float32)
        v_np = np.asarray(value, dtype=np.float32)

        # Check for GQA repetition done outside or handle here? Assume done outside for now.
        # Check shapes: Q (b, h, q_len, dim), K (b, h, kv_len, dim), V (b, h, kv_len, dim)
        if q_np.shape[-2] != k_np.shape[-2] or k_np.shape[-2] != v_np.shape[-2]:
             print(f"WARN SDPA: Mismatched head count between Q/K/V after potential GQA repeat? Q:{q_np.shape}, K:{k_np.shape}, V:{v_np.shape}")
             # This might indicate GQA wasn't handled correctly before calling

        k_transposed = np.swapaxes(k_np, -1, -2) # (b, h, dim, kv_len)
        d_k = q_np.shape[-1]
        if d_k == 0: raise ValueError("Query head dimension zero.")
        scale = 1.0 / np.sqrt(d_k).astype(q_np.dtype)

        # Scores: (b, h, q_len, dim) @ (b, h, dim, kv_len) -> (b, h, q_len, kv_len)
        scores = np.matmul(q_np, k_transposed) * scale

        if mask is not None:
            mask_np = np.asarray(mask)
            while mask_np.ndim < scores.ndim: mask_np = np.expand_dims(mask_np, axis=0) # Add B/H dims
            scores = np.where(mask_np == 0, np.finfo(scores.dtype).min, scores)

        attention_weights = softmax(scores, axis=-1) # Softmax over kv_len

        # Output: (b, h, q_len, kv_len) @ (b, h, kv_len, dim) -> (b, h, q_len, dim)
        output = np.matmul(attention_weights.astype(v_np.dtype), v_np)

        return output.astype(query.dtype if hasattr(query, 'dtype') else np.float16)

    except ValueError as ve: print(f" Error in SDPA (ValueError): {ve}"); return None
    except Exception as e: print(f" Error in SDPA: {e}"); return None

def merge_heads(data: np.ndarray, **kw) -> Optional[np.ndarray]:
    """ Placeholder: Merges attention heads back. """
    print(f"WARN: Placeholder op: merge_heads")
    if _check_none(data): return None
    try:
        # Example: (batch, num_heads, seq_len, head_dim) -> (batch, seq_len, num_heads * head_dim)
        transposed = np.transpose(data, (0, 2, 1, 3)) # (batch, seq_len, num_heads, head_dim)
        batch_size, seq_len, _, _ = transposed.shape
        merged = transposed.reshape(batch_size, seq_len, -1) # Reshape last two dims
        return merged
    except Exception as e:
        print(f"Error in merge_heads placeholder: {e}")
        return None

# Добавь функцию для добавления bias, если matrix_multiply его не поддерживает
def add_bias(data: np.ndarray, bias: Optional[np.ndarray] = None, **kw) -> Optional[np.ndarray]:
     """Adds bias vector to the last dimension."""
     print(f"INFO: Executing add_bias")
     if _check_none(data): return None
     if bias is None: return data # No bias to add
     try:
         b_np = np.asarray(bias)
         if data.shape[-1:] != b_np.shape:
             if b_np.ndim == 1 and data.shape[-1] == b_np.shape[0]:
                  return data + b_np # Broadcast bias
             else: raise ValueError(f"Incompatible shapes for bias add: {data.shape} vs {b_np.shape}")
         else: return data + b_np
     except Exception as e: print(f"Error in add_bias: {e}"); return None


# --- Example usage block ---
if __name__ == "__main__":
    # Example testing of some functions
    print("--- Operations Test ---")
    x_test = np.array([[1.0, -2.0], [3.0, -4.0]], dtype=np.float16)
    print(f"Input x:\n{x_test}")
    print(f"\nSiLU:")
    silu_out = silu(x_test)
    print(silu_out)

    print(f"\nLayerNorm (RMS):")
    w_ln = np.array([0.5, 1.5], dtype=np.float16)
    b_ln = np.array([0.1, -0.1], dtype=np.float16)
    ln_out = layer_normalization(x_test, norm_weight=w_ln, norm_bias=b_ln, eps=1e-5)
    print(ln_out)

    print(f"\nMatMul + Bias:")
    w_mm = np.random.randn(2, 3).astype(np.float16)
    b_mm = np.random.randn(3).astype(np.float16)
    mm_out = matrix_multiply(x_test, weights=w_mm, bias=b_mm)
    print(f"Output shape: {mm_out.shape}\n{mm_out}")

    print(f"\nResidual Add:")
    res_out_ok = add(x_test, x_test*0.5 + 0.1) # Add compatible shape
    print(f"OK Output shape: {res_out_ok.shape}\n{res_out_ok}")
    res_out_bad = add(x_test, mm_out) # Add incompatible shapes
    print(f"Bad Output (should be None or Error): {res_out_bad}")

    print(f"\nEmbedding Lookup:")
    embed_matrix = np.arange(20).reshape(10, 2).astype(np.float16) # Vocab 10, dim 2
    ids = np.array([[0, 1, 9, 2], [5, 0, 3, 8]])
    embed_out = embedding_lookup(ids, embedding_matrix=embed_matrix)
    print(f"Output shape: {embed_out.shape if embed_out is not None else 'None'}\n{embed_out}")
    ids_bad = np.array([[10]]) # Out of bounds
    embed_out_bad = embedding_lookup(ids_bad, embedding_matrix=embed_matrix)
    print(f"Bad ID Output: {embed_out_bad}")

    print(f"Input tuple: {rope_output_tuple}")
    print(f"get_q_rot output: {q}")
    print(f"get_k_rot output: {k}")
    print(f"get_q_rot on wrong type: {get_q_rot(np.array([1]))}")
    print(f"get_k_rot on short tuple: {get_k_rot((np.array([1]),))}")