import numpy as np
import scipy.linalg  # Для LU-разложения
from scipy.signal import convolve2d  # Для улучшенной свёртки

def mod(x, y):
    """Возвращает остаток от деления x на y."""
    if x is None or y is None or y == 0:
        return None
    return x % y

def floor(x):
    """Округление вниз."""
    if x is None:
        return None
    return np.floor(x)

def ceil(x):
    """Округление вверх."""
    if x is None:
        return None
    return np.ceil(x)

# --- Расширенные тригонометрические функции ---
def arcsin(x):
    """Возвращает арксинус (в радианах)."""
    if x is None:
        return None
    return np.arcsin(x)

def arccos(x):
    """Возвращает арккосинус (в радианах)."""
    if x is None:
        return None
    return np.arccos(x)

def arctan(x):
    """Возвращает арктангенс (в радианах)."""
    if x is None:
        return None
    return np.arctan(x)

def xor(x, y):
    """Логическое XOR."""
    if x is None or y is None:
        return None
    return x ^ y

def nand(x, y):
    """Логическое NAND (НЕ-И)."""
    if x is None or y is None:
        return None
    return ~(x & y)

def nor(x, y):
    """Логическое NOR (НЕ-ИЛИ)."""
    if x is None or y is None:
        return None
    return ~(x | y)

# --- Дополнительные операции с матрицами ---
def inverse(matrix):
    """Обратная матрица."""
    if matrix is None:
        return None
    try:
        return np.linalg.inv(matrix)
    except np.linalg.LinAlgError:
        return None  # Если матрица вырождена, вернуть None

def trace(matrix):
    """След матрицы (сумма диагональных элементов)."""
    if matrix is None:
        return None
    return np.trace(matrix)

# --- Вероятностные функции и статистика ---
def random_uniform(min_val, max_val):
    """Генерирует случайное число с равномерным распределением."""
    return np.random.uniform(min_val, max_val)

def random_normal(mu, sigma):
    """Генерирует случайное число с нормальным распределением."""
    return np.random.normal(mu, sigma)

def median(x):
    """Вычисляет медиану массива."""
    if x is None:
        return None
    return np.median(x)

def matrix_multiply(a, b):
    """Умножение матриц (a @ b.T)."""
    if a is None or b is None:
        return None
    return np.dot(a, b.T) if a.shape[-1] == b.shape[0] else None

def gradient_descent(data, grad, lr=0.01):
    """Градиентный спуск для списка данных."""
    if data is None or grad is None:
        return None
    return [d - lr * g for d, g in zip(data, grad)]

def softmax(x):
    """Softmax с численной стабилизацией."""
    if x is None:
        return None
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def matrix_determinant(a):
    """Определитель матрицы."""
    if a is None:
        return None
    return np.linalg.det(a)

def matrix_eigenvalues(a):
    """Собственные значения матрицы."""
    if a is None:
        return None
    return np.linalg.eigvals(a)

def matrix_lu_decomposition(a):
    """LU-разложение через SciPy."""
    if a is None or not isinstance(a, np.ndarray):
        return None
    try:
        p, l, u = scipy.linalg.lu(a)
        return p, l, u
    except Exception as e:
        print(f"Ошибка LU-разложения: {e}")
        return None

def convolution(data, kernel):
    """Улучшенная свёртка с использованием SciPy."""
    if data is None or kernel is None:
        return None
    try:
        return convolve2d(data, kernel, mode='same', boundary='fill', fillvalue=0)
    except Exception as e:
        print(f"Ошибка свёртки: {e}")
        return None

def transpose(a):
    """Транспонирование для любых размерностей."""
    if a is None:
        return None
    return np.transpose(a)

def mean(x):
    """Среднее значение с проверкой."""
    if x is None:
        return None
    return np.mean(x) if isinstance(x, np.ndarray) else None

def std_dev(x):
    """Стандартное отклонение с проверкой."""
    if x is None:
        return None
    return np.std(x) if isinstance(x, np.ndarray) else None

def relu(x):
    """ReLU для тензоров."""
    if x is None:
        return None
    return np.maximum(0, x)

def sigmoid(x):
    """Сигмоид для тензоров."""
    if x is None:
        return None
    return 1 / (1 + np.exp(-x))

def exponential_smoothing(data, alpha=0.5):
    """Экспоненциальное сглаживание временных рядов."""
    if data is None or not isinstance(data, (list, np.ndarray)):
        return None
    data = np.array(data) if isinstance(data, list) else data
    smoothed = np.zeros_like(data)
    smoothed[0] = data[0]
    for i in range(1, len(data)):
        smoothed[i] = alpha * data[i] + (1 - alpha) * smoothed[i-1]
    return smoothed

def normalize(data):
    """Нормализация в диапазон [0, 1]."""
    if data is None:
        return None
    data_min = np.min(data)
    data_max = np.max(data)
    return (data - data_min) / (data_max - data_min + 1e-8)

def interpolate(data, new_length):
    """Линейная интерполяция для тензоров."""
    if data is None or not hasattr(data, '__len__'):
        return None
    old_indices = np.arange(len(data))
    new_indices = np.linspace(0, len(data)-1, new_length)
    return np.interp(new_indices, old_indices, data)

def self_attention(inputs):
    """Self-Attention механизм для трёх входов [Q, K, V]."""
    if len(inputs) != 3 or any(i is None for i in inputs):
        return None
    q, k, v = inputs
    scores = matrix_multiply(q, k)  # Q @ K^T
    attention = softmax(scores)
    return matrix_multiply(attention, v)

def layer_normalization(inputs):
    """LayerNorm для входных данных."""
    if inputs is None or len(inputs) != 1:
        return None
    x = inputs[0]
    mean_x = np.mean(x, axis=-1, keepdims=True)
    std_x = np.std(x, axis=-1, keepdims=True)
    return (x - mean_x) / (std_x + 1e-5)

def multi_head_attention(inputs, num_heads=8):
    """Multi-Head Attention с разделением на головы."""
    if len(inputs) != 3 or any(i is None for i in inputs):
        return None
    q, k, v = inputs
    head_dim = q.shape[-1] // num_heads
    heads = []
    for i in range(num_heads):
        q_i = q[..., i*head_dim:(i+1)*head_dim]
        k_i = k[..., i*head_dim:(i+1)*head_dim]
        v_i = v[..., i*head_dim:(i+1)*head_dim]
        head_output = self_attention([q_i, k_i, v_i])
        heads.append(head_output)
    return np.concatenate(heads, axis=-1)

# Квантовые операции перенесены в core.py (Qiskit)
def quantum_hadamard(qubit):
    """Заглушка: реализация в core.py через Qiskit."""
    return qubit

def quantum_pauli_x(qubit):
    """Заглушка: реализация в core.py через Qiskit."""
    return qubit

def quantum_cnot(control, target):
    """Заглушка: реализация в core.py через Qiskit."""
    return [control, target]

def quantum_measure(qubit_state):
    """Заглушка: реализация в core.py через Qiskit."""
    return qubit_state

def quantum_superposition(state):
    """Заглушка: реализация в core.py через Qiskit."""
    return state

def quantum_entanglement(qubit1, qubit2):
    """Заглушка: реализация в core.py через Qiskit."""
    return [qubit1, qubit2]

def batch_norm(x):
    """Batch Normalization."""
    if x is None:
        return None
    mean_x = np.mean(x, axis=0)
    std_x = np.std(x, axis=0)
    return (x - mean_x) / (std_x + 1e-5)

def leaky_relu(x, alpha=0.01):
    """Leaky ReLU."""
    if x is None:
        return None
    return np.maximum(alpha * x, x)

def gelu(x):
    """GELU-активация для тензоров."""
    if x is None:
        return None
    return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))

def dropout(x, rate=0.5):
    """Dropout-регуляризация."""
    if x is None:
        return None
    mask = np.random.binomial(1, 1 - rate, size=x.shape)  # Исправлено: 1 - rate для корректного dropout
    return x * mask / (1 - rate)  # Масштабирование для сохранения ожидаемого значения

def scaled_dot_product_attention(query, key, value, mask=None):
    """Scaled Dot-Product Attention с маскированием."""
    if query is None or key is None or value is None:
        return None
    d_k = query.shape[-1]
    scores = matrix_multiply(query, key) / np.sqrt(d_k)
    if mask is not None:
        scores = np.where(mask == 0, -1e9, scores)
    attention = softmax(scores)
    return matrix_multiply(attention, value)

def causal_mask(size):
    """Создаёт causal mask для авторегрессивных моделей."""
    mask = np.triu(np.ones((1, size, size)), k=1)
    return 1 - mask.astype(bool)  # Исправлено: инверсия маски (1 для видимых, 0 для скрытых)

def masked_fill(tensor, mask, value):
    """Заполняет тензор значениями по маске."""
    if tensor is None or mask is None:
        return None
    return np.where(mask, value, tensor)

if __name__ == "__main__":
    # Пример использования
    x = np.array([1, -2, 3, -4])
    print(f"ReLU: {relu(x)}")
    print(f"Sigmoid: {sigmoid(x)}")
    print(f"Softmax: {softmax(x)}")
    print(f"Dropout: {dropout(x, rate=0.5)}")