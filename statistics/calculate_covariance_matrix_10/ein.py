import einops
import numpy as np

def solve(vectors: list[list[float]]) -> 'np.ndarray':
    vectors = np.array(vectors, dtype=np.float64)
    vectors = vectors - einops.reduce(vectors, 'n m -> n 1', 'mean')
    return einops.einsum(vectors, vectors, 'n1 m, n2 m -> n1 n2') / (vectors.shape[1] - 1)

calculate_covariance_matrix = solve
