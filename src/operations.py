# FILE: operations.py
# Version: 0.8.9 (Use float32 for matmul/softmax within SDPA for stability)

import math
import traceback
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# --- Version ---
# IZMENENO: Obnovlena versija
OPERATIONS_VERSION = "0.8.9" # Use float32 for matmul/softmax within SDPA for stability
# --- End Version ---

# --- RoPE Getters ---
def get_q_rot(data: Tuple, **kw) -> Optional[np.ndarray]:
    if isinstance(data, tuple) and len(data) >= 1: return data[0]
    return None
def get_k_rot(data: Tuple, **kw) -> Optional[np.ndarray]:
    if isinstance(data, tuple) and len(data) >= 2: return data[1]
    return None

# --- Helper for checking inputs ---
def _check_none(*args):
    if len(args) == 1 and isinstance(args[0], (list, tuple)): args = args[0]
    return any(a is None for a in args)

# --- Basic Math & Logic ---
# (Bez izmenenij)
def mod(x, y):
    if _check_none(x, y): return None
    y_arr = np.asarray(y);
    if np.any(y_arr == 0): print("Error in mod: Division by zero."); return None
    try: return np.mod(x, y)
    except Exception as e: print(f"Error in mod: {e}"); return None
def floor(x):
    if _check_none(x): return None
    try: return np.floor(x)
    except Exception as e: print(f"Error in floor: {e}"); return x
def ceil(x):
    if _check_none(x): return None
    try: return np.ceil(x)
    except Exception as e: print(f"Error in ceil: {e}"); return x
def arcsin(x):
    if _check_none(x): return None
    try: x_clipped = np.clip(x, -1.0, 1.0); return np.arcsin(x_clipped)
    except Exception as e: print(f"Error in arcsin: {e}"); return x
def arccos(x):
    if _check_none(x): return None
    try: x_clipped = np.clip(x, -1.0, 1.0); return np.arccos(x_clipped)
    except Exception as e: print(f"Error in arccos: {e}"); return x
def arctan(x):
    if _check_none(x): return None
    try: return np.arctan(x)
    except Exception as e: print(f"Error in arctan: {e}"); return x
def xor(x, y):
    if _check_none(x, y): return None
    try: return np.logical_xor(x, y)
    except Exception as e: print(f"Error in xor: {e}"); return None
def nand(x, y):
    if _check_none(x, y): return None
    try: return np.logical_not(np.logical_and(x, y))
    except Exception as e: print(f"Error in nand: {e}"); return None
def nor(x, y):
    if _check_none(x, y): return None
    try: return np.logical_not(np.logical_or(x, y))
    except Exception as e: print(f"Error in nor: {e}"); return None
def inverse(matrix):
    if _check_none(matrix): return None
    try: return np.linalg.inv(matrix)
    except np.linalg.LinAlgError: print("Error in inverse: Matrix is singular."); return None
    except Exception as e: print(f"Error in inverse: {e}"); return None
def trace(matrix):
    if _check_none(matrix): return None
    try: return np.trace(matrix)
    except Exception as e: print(f"Error in trace: {e}"); return None
def random_uniform(min_val=0.0, max_val=1.0):
    try: return np.random.uniform(min_val, max_val)
    except Exception as e: print(f"Error in random_uniform: {e}"); return 0.0
def random_normal(mu=0.0, sigma=1.0):
    try: return np.random.normal(mu, sigma)
    except Exception as e: print(f"Error in random_normal: {e}"); return 0.0
def median(x):
    if _check_none(x): return None
    try: return np.median(x)
    except Exception as e: print(f"Error in median: {e}"); return None
def softmax(x, axis=-1):
    # IZMENENO: Prinuditel'no ispol'zuem float32 dlja stabil'nosti
    if _check_none(x): return None
    try:
        # Vsegda perevodim v float32 dlja vychislenij
        x_f32 = np.asarray(x, dtype=np.float32)
        # Stabil'nyj softmax
        exp_x = np.exp(x_f32 - np.max(x_f32, axis=axis, keepdims=True))
        sum_exp_x = np.sum(exp_x, axis=axis, keepdims=True)
        # Izbegaem delenija na nol'
        result_f32 = exp_x / np.where(sum_exp_x == 0, 1e-9, sum_exp_x)
        # Vozvrashhaem v float32, tak kak vhod byl preobrazovan
        return result_f32
    except Exception as e:
        print(f"Error in softmax: {e}")
        # V sluchae oshibki vozvrashhaem vhodnoj massiv (ili None, ili NaN)
        # Vozvrat NaN mozhet byt' bolee informativnym
        return np.full_like(x, np.nan, dtype=np.float32) # Vozvrashhaem NaN v float32
def matrix_determinant(a):
    if _check_none(a): return None
    try: return np.linalg.det(a)
    except Exception as e: print(f"Error in matrix_determinant: {e}"); return None
def matrix_eigenvalues(a):
    if _check_none(a): return None
    try: return np.linalg.eigvals(a)
    except Exception as e: print(f"Error in matrix_eigenvalues: {e}"); return None
def transpose(a, axes=None):
    if _check_none(a): return None
    try: return np.transpose(a, axes=axes)
    except Exception as e: print(f"Error in transpose: {e}"); return a
def mean(x, axis=None, keepdims=False):
    if _check_none(x): return None
    try: return np.mean(x, axis=axis, keepdims=keepdims)
    except Exception as e: print(f"Error in mean: {e}"); return None
def std_dev(x, axis=None, keepdims=False):
    if _check_none(x): return None
    try: return np.std(x, axis=axis, keepdims=keepdims)
    except Exception as e: print(f"Error in std_dev: {e}"); return None

# --- NN Related Operations ---
def add(input_a: Optional[np.ndarray], input_b: Optional[np.ndarray], **kwargs) -> Optional[np.ndarray]:
    if input_a is None and input_b is None: print("Error in add: Both inputs are None."); return None
    if input_a is None: return input_b
    if input_b is None: return input_a
    try: return np.add(input_a, input_b)
    except ValueError as ve: shape_a = getattr(input_a, 'shape', 'N/A'); shape_b = getattr(input_b, 'shape', 'N/A'); print(f"Error add (Shapes: {shape_a}, {shape_b}): {ve}"); return None
    except Exception as e: print(f"Error add: {e}"); return None

def layer_normalization(data: Optional[np.ndarray], *, norm_weight: Optional[np.ndarray] = None, norm_bias: Optional[np.ndarray] = None, eps: float = 1e-5, **kwargs) -> Optional[np.ndarray]:
    if _check_none(data): print("Error LN: input None"); return None
    try:
        original_dtype = data.dtype
        x_f32 = np.asarray(data, dtype=np.float32) # Vychisljaem v float32
        mean = np.mean(x_f32, axis=-1, keepdims=True)
        variance = np.mean(np.square(x_f32 - mean), axis=-1, keepdims=True) # Pravil'naja variancija
        inv_std = np.reciprocal(np.sqrt(variance + float(eps)))
        x_norm_f32 = (x_f32 - mean) * inv_std # Normalizovannyj float32

        # Mnozhim na ves (gamma)
        if norm_weight is not None:
            gamma = np.asarray(norm_weight, dtype=np.float32) # Privodim gamma k float32
            if x_norm_f32.shape[-1] != gamma.shape[-1]: raise ValueError(f"LN weight shape {gamma.shape[-1]} != input hidden dim {x_norm_f32.shape[-1]}")
            x_norm_f32 = x_norm_f32 * gamma

        # Dodavljaem smeshhenie (beta)
        if norm_bias is not None:
            beta = np.asarray(norm_bias, dtype=np.float32) # Privodim beta k float32
            if x_norm_f32.shape[-1] != beta.shape[-1]: raise ValueError(f"LN bias shape {beta.shape[-1]} != input hidden dim {x_norm_f32.shape[-1]}")
            x_norm_f32 = x_norm_f32 + beta

        # Vozvrashhaem v ishodnom tipe dannyh
        return x_norm_f32.astype(original_dtype)
    except Exception as e: print(f" Error in layer_normalization: {e}"); traceback.print_exc(); return None

def matrix_multiply(a: Optional[np.ndarray], weights: Optional[np.ndarray], bias: Optional[np.ndarray] = None, **kwargs) -> Optional[np.ndarray]:
    if _check_none(a, weights): return None
    try:
        a_np = np.asarray(a)
        w_np = np.asarray(weights)
        input_a_dtype = a_np.dtype # Sohranjaem tip vhoda 'a'

        # Vychisljaem matmul
        # NumPy avtomaticheski povysit tochnost', esli nuzhno (naprimer, float16 @ float16 -> float16)
        result = np.matmul(a_np, w_np)

        # Dobavljaem bias
        if bias is not None:
            b_np = np.asarray(bias)
            # Privodim bias k tipu rezul'tata pered slozheniem
            if b_np.dtype != result.dtype:
                 b_np = b_np.astype(result.dtype)

            if result.ndim < 1: raise ValueError("Result of matmul has 0 dimensions.")
            if result.shape[-1:] != b_np.shape:
                if b_np.ndim == 1 and result.shape[-1] == b_np.shape[0]: result = result + b_np # Broadcasting
                else: raise ValueError(f"Incompatible shapes for bias add: {result.shape} vs {b_np.shape}")
            else: result = result + b_np

        # Prinuditel'no privodim rezul'tat k float16, esli vhod 'a' byl float16
        # Eto nuzhno, chtoby izbezhat' nakoplenija float32 v posledovatel'nosti operacij
        if input_a_dtype == np.float16 and result.dtype != np.float16:
            result = result.astype(np.float16)

        return result
    except ValueError as e: print(f"Error MM (Shapes A:{getattr(a,'shape','N/A')}, W:{getattr(weights,'shape','N/A')}, B:{getattr(bias,'shape','N/A')}): {e}"); return None
    except Exception as e: print(f"Unexpected error in matrix_multiply: {e}"); return None

def silu(data: Optional[np.ndarray], **kwargs) -> Optional[np.ndarray]:
    if _check_none(data): return None
    try:
        x = np.asarray(data)
        # Vychisljaem sigmoid v float32 dlja stabil'nosti
        sigmoid_x = 1.0 / (1.0 + np.exp(-x.astype(np.float32)))
        # Umnozhaem ishodnyj x na float32 sigmoid, rezul'tat budet float32 ili ishodnyj tip
        result = x * sigmoid_x
        # Vozvrashhaem v ishodnom tipe
        return result.astype(x.dtype)
    except Exception as e: print(f" Error in silu: {e}"); return None

# --- Activations ---
def relu(x):
    if _check_none(x): return None
    try: return np.maximum(0, x)
    except Exception as e: print(f"Error relu: {e}"); return x

def sigmoid(x):
    if _check_none(x): return None
    try:
        x_np = np.asarray(x)
        # Vychisljaem v float32
        result_f32 = 1.0 / (1.0 + np.exp(-x_np.astype(np.float32)))
        # Vozvrashhaem v ishodnom tipe
        return result_f32.astype(x.dtype if hasattr(x, "dtype") else np.float16)
    except Exception as e: print(f"Error sigmoid: {e}"); return x

def leaky_relu(x, alpha=0.01):
    if _check_none(x): return None
    try: return np.maximum(alpha * x, x)
    except Exception as e: print(f"Error leaky_relu: {e}"); return x

def gelu(x):
    if _check_none(x): return None
    try:
        x_np = np.asarray(x)
        # Vychisljaem v float32
        x_f32 = x_np.astype(np.float32)
        M_SQRT2_OVER_PI = np.sqrt(2.0 / np.pi).astype(np.float32)
        COEFF = np.array(0.044715).astype(np.float32)
        ONE = np.array(1.0).astype(np.float32)
        HALF = np.array(0.5).astype(np.float32)
        cdf = HALF * (ONE + np.tanh(M_SQRT2_OVER_PI * (x_f32 + COEFF * np.power(x_f32, 3))))
        result_f32 = x_f32 * cdf
        # Vozvrashhaem v ishodnom tipe
        return result_f32.astype(x.dtype if hasattr(x, "dtype") else np.float16)
    except Exception as e: print(f"Error gelu: {e}"); return x

# --- Other NN Utils ---
def dropout(data, *, rate=0.1, is_training=False, **kwargs):
    if _check_none(data) or not is_training or rate <= 0 or rate >= 1: return data
    try: keep = 1.0 - rate; mask = np.random.binomial(1, keep, size=data.shape); return (data * mask) / keep
    except Exception as e: print(f"Error dropout: {e}"); return data
def normalize(data):
    if _check_none(data): return None
    try: dmin = np.min(data); dmax = np.max(data); denom = dmax - dmin; return (data - dmin) / (denom if denom > 1e-8 else 1e-8)
    except Exception as e: print(f"Error normalize: {e}"); return data
def interpolate(data, new_length):
    if _check_none(data, new_length) or not hasattr(data, "__len__"): return None
    try: old_idx = np.arange(len(data)); new_idx = np.linspace(0, len(data) - 1, int(new_length)); return np.interp(new_idx, old_idx, data)
    except Exception as e: print(f"Error interpolate: {e}"); return data
def batch_norm(data, **kwargs):
    if _check_none(data): return None
    try: x = np.asarray(data); m = np.mean(x, axis=0); s = np.std(x, axis=0); return (x - m) / (s + 1e-5)
    except Exception as e: print(f"Error batch_norm: {e}"); return data
def exponential_smoothing(data, alpha=0.5):
    if _check_none(data) or not isinstance(data, (list, np.ndarray)): return None
    try: d = np.array(data); s = np.zeros_like(d); s[0] = d[0]
    except Exception as e: print(f"Error init smoothing: {e}"); return data
    try:
        for i in range(1, len(d)): s[i] = alpha * d[i] + (1 - alpha) * s[i - 1]
        return s
    except Exception as e: print(f"Error smoothing loop: {e}"); return data

# --- Transformer Ops (Cleaned) ---
def embedding_lookup(token_ids: np.ndarray, *, embedding_matrix: np.ndarray, **kw) -> Optional[np.ndarray]:
    if _check_none(token_ids, embedding_matrix): return None
    try:
        ids = np.asarray(token_ids).astype(np.int64)
        emb = np.asarray(embedding_matrix); vs = emb.shape[0]
        if np.any(ids >= vs) or np.any(ids < 0): print(f"Error: Token IDs out of bounds [0, {vs-1}]. Got min/max: {np.min(ids)}/{np.max(ids)}"); return None
        # Dequantizacija proishodit v core.load_knowledge_tensor_data, zdes' ozhidaem float16/32
        return emb[ids].astype(emb.dtype)
    except Exception as e: print(f" Error in embedding_lookup: {e}"); return None

def apply_rope(q: np.ndarray, k: np.ndarray, *, position_ids: np.ndarray, rope_theta: float = 10000.0, num_heads: Optional[int] = None, head_dim: Optional[int] = None, **kw) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if _check_none(q, k, position_ids): print("ERROR: apply_rope None input."); return None, None
    try:
        original_dtype = q.dtype
        # Vychisljaem v float32 dlja tochnosti trigonometrii
        compute_dtype = np.float32
        if q.ndim != 4 or k.ndim != 4: raise ValueError(f"apply_rope expects 4D input (b, h, s, d), got Q:{q.shape}, K:{k.shape}")
        if q.shape[0] != k.shape[0] or q.shape[2] != k.shape[2]: raise ValueError(f"Batch/SeqLen mismatch in apply_rope: Q:{q.shape}, K:{k.shape}")
        if q.shape[3] != k.shape[3]: raise ValueError(f"Head dimension mismatch in apply_rope: Q:{q.shape}, K:{k.shape}")
        batch_size, num_q_heads, seq_len, q_head_dim = q.shape
        _, num_k_heads, _, k_head_dim = k.shape
        if q_head_dim % 2 != 0: raise ValueError(f"head_dim ({q_head_dim}) must be even.")
        dim = q_head_dim // 2
        if position_ids.ndim == 1 and position_ids.shape[0] == seq_len: position_ids = np.expand_dims(position_ids, axis=0)
        if position_ids.shape == (1, seq_len) and batch_size > 1: position_ids = np.tile(position_ids, (batch_size, 1))
        elif position_ids.shape != (batch_size, seq_len): raise ValueError(f"Incompatible position_ids shape: {position_ids.shape} vs expected ({batch_size}, {seq_len})")
        theta_indices = np.arange(0, dim, dtype=compute_dtype)
        if rope_theta == 0: raise ValueError("rope_theta cannot be zero")
        theta_denominator = rope_theta**(np.array(2.0).astype(compute_dtype) * theta_indices / q_head_dim)
        if np.any(theta_denominator == 0): raise ValueError("Zero denominator in RoPE theta")
        theta = 1.0 / theta_denominator
        freqs = np.outer(position_ids.flatten(), theta).reshape(batch_size, seq_len, dim)
        # Vychisljaem cos/sin v float32
        cos_freqs = np.cos(freqs).astype(compute_dtype); sin_freqs = np.sin(freqs).astype(compute_dtype)
        cos_emb = np.concatenate((cos_freqs, cos_freqs), axis=-1); sin_emb = np.concatenate((sin_freqs, sin_freqs), axis=-1)
        cos_emb = np.expand_dims(cos_emb, axis=1); sin_emb = np.expand_dims(sin_emb, axis=1)
        def rotate_half(x: np.ndarray) -> np.ndarray:
            try: x1, x2 = np.split(x, 2, axis=-1); return np.concatenate((-x2, x1), axis=-1)
            except Exception as e_rot: print(f"ERROR in rotate_half (Input shape: {x.shape}): {e_rot}"); traceback.print_exc(); raise
        # Vychisljaem v float32
        q_rotated = (q.astype(compute_dtype) * cos_emb) + (rotate_half(q.astype(compute_dtype)) * sin_emb)
        k_rotated = (k.astype(compute_dtype) * cos_emb) + (rotate_half(k.astype(compute_dtype)) * sin_emb)
        # Vozvrashhaem v ishodnom tipe
        return q_rotated.astype(original_dtype), k_rotated.astype(original_dtype)
    except ValueError as ve: print(f"ERROR apply_rope (ValueError Captured): {ve}"); traceback.print_exc(); return None, None
    except Exception as e: print(f"ERROR apply_rope (General Exception Captured): {e}"); traceback.print_exc(); return None, None

# --- scaled_dot_product_attention ---
# IZMENENO: Prinuditel'no ispol'zuem float32 dlja vychislenija scores i softmax
def scaled_dot_product_attention(query: np.ndarray, key: np.ndarray, value: np.ndarray, mask: Optional[np.ndarray] = None, **kw) -> Optional[np.ndarray]:
    if _check_none(query, key, value): print("Error SDPA: Q, K, or V is None."); return None
    try:
        original_dtype = query.dtype # Sohranjaem ishodnyj tip (verojatno float16)
        # Prinuditel'no perevodim Q, K, V v float32 dlja vychislenij
        compute_dtype = np.float32
        q_f32 = np.asarray(query, dtype=compute_dtype)
        k_f32 = np.asarray(key, dtype=compute_dtype)
        v_f32 = np.asarray(value, dtype=compute_dtype)

        if q_f32.ndim != 4: raise ValueError(f"SDPA Q expects 4D (b,h_q,s_q,d), got {query.shape}")
        if k_f32.ndim != 4: raise ValueError(f"SDPA K expects 4D (b,h_kv,s_kv_cache,d), got {key.shape}")
        if v_f32.ndim != 4: raise ValueError(f"SDPA V expects 4D (b,h_kv,s_kv_cache,d), got {value.shape}")

        batch_size, num_heads_q, q_len, head_dim = q_f32.shape
        _, num_heads_kv, kv_cache_len, kv_head_dim = k_f32.shape

        if head_dim != kv_head_dim: raise ValueError(f"Head dimension mismatch Q({head_dim}) vs K/V({kv_head_dim})")

        total_seq_len = kw.get('total_seq_len')
        if total_seq_len is None: raise ValueError("SDPA requires 'total_seq_len' in keyword arguments (kw) to correctly slice KV cache.")
        kv_len_actual = int(total_seq_len)

        if kv_len_actual > kv_cache_len: print(f"ERROR SDPA: total_seq_len ({kv_len_actual}) > kv_cache_len ({kv_cache_len})"); return None

        # Povtorjaem golovy K, V, esli nuzhno (Grouped Query Attention)
        if num_heads_kv != num_heads_q:
            if num_heads_q % num_heads_kv == 0:
                repeats = num_heads_q // num_heads_kv
                k_perm = np.repeat(k_f32, repeats, axis=1)
                v_perm = np.repeat(v_f32, repeats, axis=1)
            else: raise ValueError(f"num_heads_q ({num_heads_q}) not divisible by num_heads_kv ({num_heads_kv})")
        else:
            k_perm = k_f32
            v_perm = v_f32

        # Obrezaem K, V do aktual'noj dliny posledovatel'nosti
        k_sliced = k_perm[:, :, :kv_len_actual, :]
        v_sliced = v_perm[:, :, :kv_len_actual, :]

        # Vychisljaem scores v float32
        k_transposed = np.swapaxes(k_sliced, -1, -2)
        scale = (1.0 / np.sqrt(head_dim)).astype(compute_dtype)
        scores = np.matmul(q_f32, k_transposed) * scale # float32 @ float32 -> float32

        # Primenjaem masku
        if mask is not None:
            mask_np = np.asarray(mask)
            expected_mask_len_rows = mask_np.shape[-2]
            expected_mask_len_cols = mask_np.shape[-1]
            if expected_mask_len_rows < q_len or expected_mask_len_cols < kv_len_actual:
                print(f"WARN SDPA: Mask shape {mask_np.shape} too small for required slice ({q_len}, {kv_len_actual}). Trying anyway.")

            # Obrezaem masku i privodim k nuzhnoj forme
            mask_slice = mask_np[..., -q_len:, :kv_len_actual]
            target_ndim = scores.ndim
            while mask_slice.ndim < target_ndim:
                mask_slice = np.expand_dims(mask_slice, axis=1) # Dobavljaem izmerenie dlja golov

            mask_slice_bool = mask_slice.astype(bool)
            # Ispol'zuem ochen' malenkoe chislo dlja float32 vmesto -inf
            scores = np.where(mask_slice_bool == False, np.finfo(scores.dtype).min, scores)

        # Vychisljaem softmax v float32
        attention_weights = softmax(scores, axis=-1) # Softmax teper' vsegda rabotaet s float32

        # Umnozhaem na V (v float32)
        output_f32 = np.matmul(attention_weights, v_sliced) # float32 @ float32 -> float32

        # Reshape vyhodnogo tenzora
        output_transposed = np.transpose(output_f32, (0, 2, 1, 3))
        bs_out, q_len_out, _, _ = output_transposed.shape
        hidden_size_out = num_heads_q * head_dim
        output_final_f32 = output_transposed.reshape(bs_out, q_len_out, hidden_size_out)

        # Vozvrashhaem rezul'tat v ISHODNOM tipe dannyh (verojatno float16)
        return output_final_f32.astype(original_dtype)

    except ValueError as ve: print(f" Error in SDPA (ValueError): {ve}"); print(f"  Shapes involved: Q={query.shape}, K={key.shape}, V={value.shape}, Mask={getattr(mask, 'shape', 'None')}"); traceback.print_exc(); return None
    except Exception as e: print(f" Error in SDPA: {e}"); traceback.print_exc(); return None

# --- update_kv_cache ---
def update_kv_cache(cache: Optional[np.ndarray], new_values: np.ndarray, start_pos: int) -> Optional[np.ndarray]:
    if _check_none(new_values): print("ERROR update_kv_cache: new_values is None"); return cache
    try:
        if cache is None: print("ERROR update_kv_cache: Received None cache. Cache must be pre-initialized."); return None
        if start_pos < 0: print(f"ERROR update_kv_cache: start_pos ({start_pos}) cannot be negative."); return cache
        if new_values.ndim != 4: print(f"ERROR update_kv_cache: new_values must be 4D, got shape {new_values.shape}"); return cache
        new_seq_len = new_values.shape[2]; end_pos = start_pos + new_seq_len
        if end_pos > cache.shape[2]: print(f"ERROR update_kv_cache: Attempting to write beyond cache bounds. Cache shape={cache.shape}, start_pos={start_pos}, new_seq_len={new_seq_len}, end_pos={end_pos}"); return cache
        if (cache.shape[0] != new_values.shape[0] or cache.shape[1] != new_values.shape[1] or cache.shape[3] != new_values.shape[3]): print(f"ERROR update_kv_cache: Shape mismatch (B, H, D). Cache={cache.shape}, New={new_values.shape}"); return cache
        # Proverka tipov pered zapis'ju
        if cache.dtype != new_values.dtype:
            print(f"WARN update_kv_cache: Dtype mismatch. Cache={cache.dtype}, New={new_values.dtype}. Casting new_values.")
            new_values = new_values.astype(cache.dtype)
        cache[:, :, start_pos : end_pos, :] = new_values
        return cache
    except Exception as e: print(f"ERROR during update_kv_cache: {e}"); traceback.print_exc(); return None

# --- create_causal_mask ---
def create_causal_mask(current_data: Any, *, size: int, dtype: np.dtype = np.bool_) -> Optional[np.ndarray]:
    if size <= 0: print("WARN create_causal_mask: size must be positive."); return None
    try:
        mask = np.triu(np.ones((1, size, size), dtype=np.bool_), k=1)
        return (mask == 0).astype(dtype) # True=Attend, False=Mask
    except Exception as e: print(f"ERROR during create_causal_mask: {e}"); return None

# --- make_tuple ---
def make_tuple(current_data: Any, **kw) -> Tuple:
        try:
            sorted_keys = sorted([k for k in kw if k.startswith('elem')], key=lambda x: int(x[4:])); result_tuple = tuple(kw[key] for key in sorted_keys)
            return result_tuple
        except Exception as e: print(f"ERROR during make_tuple: {e}"); return ()

# --- Placeholders / Utils ---
def multi_head_attention(*args, **kw): print(f"WARN: MHA placeholder"); return args[0] if args else kw.get("data")
def masked_fill(tensor, mask, value):
    if _check_none(tensor,mask): return None;
    try: mask_bool = np.asarray(mask, dtype=bool); return np.where(mask_bool, value, tensor)
    except Exception as e: print(f"Error masked_fill: {e}"); return tensor
def linear(data, *, weights, bias=None, **kwargs): return matrix_multiply(data, weights=weights, bias=bias)
def ffn_mlp(data, *, gate_weights, up_weights, down_weights, gate_bias=None, up_bias=None, down_bias=None, **kwargs) -> Optional[np.ndarray]:
    if _check_none(data, gate_weights, up_weights, down_weights): return None
    try:
        gate = linear(data, weights=gate_weights, bias=gate_bias);
        if gate is None: return None
        up = linear(data, weights=up_weights, bias=up_bias);
        if up is None: return None
        activated = silu(gate) * up;
        if activated is None: return None
        output = linear(activated, weights=down_weights, bias=down_bias);
        if output is None: return None
        return output
    except Exception as e: print(f"Error in ffn_mlp: {e}"); return None

def reshape_heads(data: np.ndarray, *, num_heads: int, **kw) -> Optional[np.ndarray]:
    if _check_none(data): print("Error reshape_heads: input None"); return None
    try:
        if data.ndim != 3: raise ValueError(f"reshape_heads expects 3D input (b, s, h*d), got {data.shape}")
        batch_size, seq_len, hidden_size = data.shape
        if num_heads <= 0: raise ValueError(f"num_heads must be positive, got {num_heads}")
        if hidden_size % num_heads != 0: raise ValueError(f"hidden_size {hidden_size} not divisible by num_heads {num_heads}")
        head_dim = hidden_size // num_heads
        reshaped = data.reshape(batch_size, seq_len, num_heads, head_dim)
        transposed = np.transpose(reshaped, (0, 2, 1, 3))
        return transposed.astype(data.dtype)
    except Exception as e: print(f"Error reshape_heads: {e}"); return None

def merge_heads(data: np.ndarray, **kw) -> Optional[np.ndarray]:
    if _check_none(data): print("Error merge_heads: input None"); return None
    try:
        if data.ndim != 4: raise ValueError(f"merge_heads expects 4D input (b, h, s, d), got {data.shape}")
        transposed = np.transpose(data, (0, 2, 1, 3))
        batch_size, seq_len, _, _ = transposed.shape
        merged = transposed.reshape(batch_size, seq_len, -1)
        return merged.astype(data.dtype)
    except Exception as e: print(f"Error merge_heads: {e}"); return None

def repeat_kv_heads(data: np.ndarray, repeats: int, **kw):
    if _check_none(data): print("Error repeat_kv_heads: input None"); return None
    if repeats <= 1: return data
    try:
        if data.ndim != 4: raise ValueError(f"repeat_kv_heads expects 4D input (b, h_kv, s, d), got {data.shape}")
        repeated_data = np.repeat(data, repeats, axis=1)
        return repeated_data.astype(data.dtype)
    except Exception as e: print(f"Error repeat_kv_heads: {e}"); return None

def add_bias(data: np.ndarray, bias: Optional[np.ndarray]=None, **kw):
    if bias is None: return data
    if _check_none(data): return None
    try: return np.add(data, bias)
    except Exception as e: print(f"Error add_bias: {e}"); return None

