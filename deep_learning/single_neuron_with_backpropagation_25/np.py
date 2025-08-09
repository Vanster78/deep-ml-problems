import numpy as np

def solve(features: np.ndarray, labels: np.ndarray, initial_weights: np.ndarray, initial_bias: float, learning_rate: float, epochs: int) -> (np.ndarray, float, list[float]):
	weights = initial_weights.copy()
	bias = initial_bias
	mse_history = []

	for e in range(epochs):
		# compute probs
		z = features @ weights + bias
		p = 1 / (1 + np.exp(-z))
		# compute and store loss
		loss = ((p - labels) ** 2).mean()
		mse_history.append(round(loss, 4))
		# compute gradient
		grad_p = 2 * (p - labels) / len(p)
		grad_z = p * (1 - p) * grad_p
		grad_w = features.T @ grad_z
		grad_b = grad_z.sum()
		# update weights and bias
		weights -= learning_rate * grad_w
		bias -= learning_rate * grad_b

	return weights.round(4), round(bias, 4), mse_history

train_neuron = solve
