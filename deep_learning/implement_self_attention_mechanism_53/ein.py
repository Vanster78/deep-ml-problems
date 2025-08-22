import einops
import numpy as np

def softmax(X: np.ndarray):
    X_e = np.exp(X)
    return X_e / einops.reduce(X_e, "s1 s2 -> s1 1", "sum")

def compute_qkv(X: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray):
    W_qkv = einops.rearrange([W_q, W_k, W_v], "n d1 d2 -> d1 (n d2)")
    QKV = einops.einsum(X, W_qkv, "s d1, d1 triple_d2 -> s triple_d2")
    Q, K, V = einops.rearrange(QKV, "s (n d2) -> n s d2", n=3)
    return Q, K, V

def self_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray):
    d = Q.shape[1]
    A = einops.einsum(Q, K, "s1 d, s2 d -> s1 s2") / d ** 0.5
    P = softmax(A)
    O = einops.einsum(P, V, "s1 s2, s2 d -> s1 d")
    return O

X = np.array([[1, 0], [0, 1]])
W_q = np.array([[1, 0], [0, 1]])
W_k = np.array([[1, 0], [0, 1]])
W_v = np.array([[1, 2], [3, 4]])

Q, K, V = compute_qkv(X, W_q, W_k, W_v)
output = self_attention(Q, K, V)

print(output)
