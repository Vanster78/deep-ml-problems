import torch

def solve(scores: list[float]) -> list[float]:
	exps = torch.exp(torch.tensor(scores, dtype=torch.double))
	exps_sum = exps.sum()
	return (exps / exps_sum).round(decimals=4)

softmax = solve
