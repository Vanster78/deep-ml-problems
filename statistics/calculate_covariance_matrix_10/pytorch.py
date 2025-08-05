import torch

def solve(vectors: list[list[float]]) -> 'torch.Tensor':
    vectors = torch.tensor(vectors, dtype=torch.float)
    vectors -= vectors.mean(axis=1, keepdim=True)
    return vectors @ vectors.T / (vectors.shape[1] - 1)

calculate_covariance_matrix = solve
