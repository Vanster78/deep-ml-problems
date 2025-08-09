import torch

def solve(features: list[list[float]], labels: list[int], weights: list[float], bias: float) -> (list[float], float):
	features, labels, weights = map(lambda x: torch.tensor(x, dtype=torch.float), (features, labels, weights))
	z = features @ weights + bias
	p = 1 / (1 + torch.exp(-z))
	return [round(x, 4) for x in p.tolist()], round(((p - labels) ** 2).mean().item(), 4)
    
single_neuron_model = solve
