import numpy as np

def solve(input_matrix: np.ndarray, kernel: np.ndarray, padding: int, stride: int):
	input_height, input_width = input_matrix.shape
	kernel_height, kernel_width = kernel.shape
	
	output_height = round((input_height - kernel_height + 2 * padding + 1) / stride + 0.5)
	output_width = round((input_width - kernel_width + 2 * padding + 1) / stride + 0.5)
	output_matrix = np.zeros((output_height, output_width))
	
	padded_matrix = np.pad(input_matrix, ((padding, padding), (padding, padding)))
	padded_height, padded_width = padded_matrix.shape
	
	for i in range(0, padded_height - kernel_height + 1, stride):
		for j in range(0, padded_width - kernel_width + 1, stride):
			oi = i // stride
			oj = j // stride
			v = (padded_matrix[i:i+kernel_height, j:j+kernel_width] * kernel).sum()
			output_matrix[oi][oj] = v

	return output_matrix

simple_conv2d = solve
