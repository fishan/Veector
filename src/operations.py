import numpy as np

def matrix_multiply(a, b):
    """Умножение матриц."""
    return np.dot(a, b.T) if len(a) == len(b[0]) else None

def gradient_descent(data, grad, lr=0.01):
    """Градиентный спуск."""
    return [d - lr * g for d, g in zip(data, grad)]

def softmax(x):
    """Softmax function."""
    exp_x = np.exp(x - np.max(x))  # Для численной стабильности
    return exp_x / exp_x.sum()

def matrix_determinant(a):
    """Вычисление определителя матрицы."""
    return np.linalg.det(a)

def matrix_eigenvalues(a):
    """Вычисление собственных значений матрицы."""
    return np.linalg.eigvals(a)

def matrix_lu_decomposition(a):
    """LU-разложение матрицы."""
    try:
        import scipy.linalg
        p, l, u = scipy.linalg.lu(a)
        return p, l, u
    except ImportError:
        print("Scipy is required for LU decomposition.")
        return None

def convolution(data, kernel):
    """Свертка (convolution)."""
    # Простая реализация для 2D данных и 2D ядра
    # TODO: Оптимизировать и расширить для n-мерных данных
    if not isinstance(data, np.ndarray) or not isinstance(kernel, np.ndarray):
        return None
    if data.ndim != 2 or kernel.ndim != 2:
        return None

    data_height, data_width = data.shape
    kernel_height, kernel_width = kernel.shape
    output_height = data_height - kernel_height + 1
    output_width = data_width - kernel_width + 1

    output = np.zeros((output_height, output_width))
    for i in range(output_height):
        for j in range(output_width):
            output[i, j] = np.sum(data[i:i + kernel_height, j:j+kernel_width] * kernel)
    return output

def transpose(a):
    """Транспонирование матрицы/тензора."""
    return np.transpose(a)

def mean(x):
    """Вычисление среднего значения."""
    return np.mean(x)

def std_dev(x):
    """Вычисление стандартного отклонения."""
    return np.std(x)

def relu(x):
    """ReLU activation function."""
    return np.maximum(0, x)

def sigmoid(x):
    """Sigmoid activation function."""
    return 1 / (1 + np.exp(-x))

def exponential_smoothing(data, alpha=0.5):
    """
    Экспоненциальное сглаживание данных.
    """
    smoothed = [data[0]]
    for i in range(1, len(data)):
        smoothed.append(alpha * data[i] + (1 - alpha) * smoothed[-1])
    return smoothed

def normalize(data):
    """
    Нормализация данных в диапазоне [0, 1].
    """
    min_val = np.min(data)
    max_val = np.max(data)
    return (data - min_val) / (max_val - min_val)

def interpolate(data, new_length):
    """
    Линейная интерполяция данных до новой длины.
    """
    x_old = np.linspace(0, 1, len(data))
    x_new = np.linspace(0, 1, new_length)
    return np.interp(x_new, x_old, data)
