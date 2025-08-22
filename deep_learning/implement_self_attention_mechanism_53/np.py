import numpy as np

def softmax(X: np.ndarray):
    X_e = np.exp(X)
    return X_e / X_e.sum(axis=1, keepdims=True)

def compute_qkv(X: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray):
    return X @ W_q, X @ W_k, X @ W_v

def self_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray):
    d = Q.shape[1]
    A = Q @ K.T / d ** 0.5
    P = softmax(A)
    O = P @ V
    return O

X = np.array([[1, 0], [0, 1]])
W_q = np.array([[1, 0], [0, 1]])
W_k = np.array([[1, 0], [0, 1]])
W_v = np.array([[1, 2], [3, 4]])

Q, K, V = compute_qkv(X, W_q, W_k, W_v)
output = self_attention(Q, K, V)

print(output)
