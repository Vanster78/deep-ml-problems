import numpy as np

def solve(scores: list) -> np.ndarray:
	scores = np.array(scores)
	shift = np.log(np.exp(scores).sum())
	return scores - shift

log_softmax = solve
