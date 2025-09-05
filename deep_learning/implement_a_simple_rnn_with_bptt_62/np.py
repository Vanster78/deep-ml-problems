import numpy as np

def tanh(z: np.ndarray) -> np.ndarray:
    ez = np.exp(z)
    nez = np.exp(-z)
    return (ez - nez) / (ez + nez)

class SimpleRNN:
	def __init__(self, input_size, hidden_size, output_size):
		"""
		Initializes the RNN with random weights and zero biases.
		"""
		self.hidden_size = hidden_size
		self.W_xh = np.random.randn(hidden_size, input_size)*0.01
		self.W_hh = np.random.randn(hidden_size, hidden_size)*0.01
		self.W_hy = np.random.randn(output_size, hidden_size)*0.01
		self.b_h = np.zeros((hidden_size, 1))
		self.b_y = np.zeros((output_size, 1))

	def forward(self, x, cache: bool = False):
		h_t = np.zeros((self.hidden_size, 1))
		output = np.empty((x.shape[0], self.b_y.shape[0], 1))
		
		if cache:
			self.h_cache = np.empty((x.shape[0], self.hidden_size, 1))
		else:
			self.h_cache = None
		
		for t, x_t in enumerate(x):
			h_t = tanh(self.W_xh @ x_t[:, None] + self.W_hh @ h_t + self.b_h)
			o_t = self.W_hy @ h_t + self.b_y
			output[t] = o_t
			
			if cache:
				self.h_cache[t] = h_t
		
		return output[:, :, 0]

	def backward(self, x, y, learning_rate):
		output = self.forward(x, cache=True)
		output_grad = output - y

		h_t_grad = np.zeros((self.hidden_size, 1))

		W_hy_grad = np.zeros_like(self.W_hy)
		b_y_grad = np.zeros_like(self.b_y)
		W_hh_grad = np.zeros_like(self.W_hh)
		b_h_grad = np.zeros_like(self.b_h)
		W_xh_grad = np.zeros_like(self.W_xh)

		for t, x_t in reversed(list(enumerate(x))):
			o_t_grad = output_grad[t, :, None]  # [O, 1]

			W_hy_grad += o_t_grad @ self.h_cache[t].T
			b_y_grad += o_t_grad

			h_t_grad = (self.W_hy.T @ o_t_grad + self.W_hh.T @ h_t_grad) * (1 - self.h_cache[t] ** 2)  # [H, 1]
			b_h_grad += h_t_grad

			if t:
				W_hh_grad += h_t_grad @ self.h_cache[t-1].T

			W_xh_grad += h_t_grad @ x_t[:, None].T
		
		self.W_hy -= W_hy_grad * learning_rate
		self.b_y -= b_y_grad * learning_rate
		self.W_hh -= W_hh_grad * learning_rate
		self.b_h -= b_h_grad * learning_rate
		self.W_xh -= W_xh_grad * learning_rate
	

if __name__ == "__main__":
    np.random.seed(42)

    input_sequence = np.array([[1.0,2.0], [7.0,2.0], [1.0,3.0], [12.0,4.0]])
    expected_output = np.array([[2.0,1.0], [3.0,7.0], [4.0,8.0], [5.0,10.0]])
	
    rnn = SimpleRNN(input_size=2, hidden_size=10, output_size=2)
	
    for epoch in range(50):
        output = rnn.forward(input_sequence)
        rnn.backward(input_sequence, expected_output, learning_rate=0.01)

    print(output)
