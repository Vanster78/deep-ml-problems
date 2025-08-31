import numpy as np

def tanh(z: np.ndarray) -> np.ndarray:
    ez = np.exp(z)
    nez = np.exp(-z)
    return (ez - nez) / (ez + nez)

def solve(input_sequence: list[list[float]], initial_hidden_state: list[float], Wx: list[list[float]], Wh: list[list[float]], b: list[float]) -> list[float]:
	input_sequence = [np.array(i) for i in input_sequence]
	hidden_state = np.array(initial_hidden_state)
	Wx = np.array(Wx)
	Wh = np.array(Wh)
	b = np.array(b)

	for x in input_sequence:
		hidden_state = tanh(Wx @ x + Wh @ hidden_state + b)

	return hidden_state
		

rnn_forward = solve
