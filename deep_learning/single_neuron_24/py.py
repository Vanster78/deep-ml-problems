import math

def sigmoid(z: float) -> float:
	return 1 / (1 + math.exp(-z))

def mse(predicted: list[float], expected: list[int]):
	return sum((p - e) ** 2 for p, e in zip(predicted, expected)) / len(predicted)

def solve(features: list[list[float]], labels: list[int], weights: list[float], bias: float) -> (list[float], float):
	z = [sum(w * x for w, x in zip(weights, f)) + bias for f in features]
	p = [sigmoid(x) for x in z]
	return [round(x, 4) for x in p], round(mse(p, labels), 4)
    
single_neuron_model = solve
