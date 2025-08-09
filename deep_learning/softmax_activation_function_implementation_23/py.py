import math

def solve(scores: list[float]) -> list[float]:
	exps = [math.exp(x) for x in scores]
	exps_sum = sum(exps)
	return [round(x / exps_sum, 4) for x in exps]

softmax = solve
