import numpy as np

def solve(vectors: list[list[float]]) -> 'np.ndarray':
    vectors = np.array(vectors, dtype=np.float64)
    vectors -= vectors.mean(axis=1, keepdims=True)
    return vectors @ vectors.T / (vectors.shape[1] - 1)

calculate_covariance_matrix = solve
