import math

def solve(z: float) -> float:
	return round(1 / (1 + math.exp(-z)), 4)

sigmoid = solve
