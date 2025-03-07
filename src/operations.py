import numpy as np

def matrix_multiply(a, b):
    return np.dot(a, b.T) if len(a) == len(b[0]) else None

def gradient_descent(data, grad, lr=0.01):
    return [d - lr * g for d, g in zip(data, grad)]

def softmax(x):
    exp_x = np.exp(x - np.max(x))  # Для численной стабильности
    return exp_x / exp_x.sum()