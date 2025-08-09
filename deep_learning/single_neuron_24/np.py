import numpy as np

def solve(features: list[list[float]], labels: list[int], weights: list[float], bias: float) -> (list[float], float):
	z = np.matmul(features, weights) + bias
	p = 1 / (1 + np.exp(-z))
	return p.round(4), ((p - labels) ** 2).mean().round(4)
    
single_neuron_model = solve
