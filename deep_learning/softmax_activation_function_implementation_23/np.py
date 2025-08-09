import numpy as np

def solve(scores: list[float]) -> list[float]:
	exps = np.exp(scores)
	exps_sum = exps.sum()
	return (exps / exps_sum).round(4)

softmax = solve
