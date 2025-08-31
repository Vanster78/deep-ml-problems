import numpy as np

class LSTM:
	def __init__(self, input_size, hidden_size):
		self.input_size = input_size
		self.hidden_size = hidden_size

		# Initialize weights and biases
		self.Wf = np.random.randn(hidden_size, input_size + hidden_size)
		self.Wi = np.random.randn(hidden_size, input_size + hidden_size)
		self.Wc = np.random.randn(hidden_size, input_size + hidden_size)
		self.Wo = np.random.randn(hidden_size, input_size + hidden_size)

		self.bf = np.zeros((hidden_size, 1))
		self.bi = np.zeros((hidden_size, 1))
		self.bc = np.zeros((hidden_size, 1))
		self.bo = np.zeros((hidden_size, 1))

	@staticmethod
	def sigmoid(z: np.ndarray) -> np.ndarray:
		return 1 / (1 + np.exp(-z))
	
	@staticmethod
	def tanh(z: np.ndarray) -> np.ndarray:
		ez = np.exp(z)
		nez = np.exp(-z)
		return (ez - nez) / (ez + nez)

	def forward(self, x, initial_hidden_state, initial_cell_state):
		hidden_state = initial_hidden_state.copy()
		cell_state = initial_cell_state.copy()
		outputs = np.empty((x.shape[0], self.hidden_size, 1))

		for t, x_t in enumerate(x):
			xh_t = np.concatenate([hidden_state, x_t[:, None]], axis=0)
			f_t = self.sigmoid(self.Wf @ xh_t + self.bf)
			i_t = self.sigmoid(self.Wi @ xh_t + self.bi)
			c_t = self.tanh(self.Wc @ xh_t + self.bc)
			cell_state = f_t * cell_state + i_t * c_t
			o_t = self.sigmoid(self.Wo @ xh_t + self.bo)
			hidden_state = o_t * self.tanh(cell_state)
			outputs[t] = hidden_state
		
		return outputs, hidden_state, cell_state

if __name__ == "__main__":
    input_sequence = np.array([[1.0], [2.0], [3.0]])
    initial_hidden_state = np.zeros((1, 1))
    initial_cell_state = np.zeros((1, 1))

    lstm = LSTM(input_size=1, hidden_size=1)

    lstm.Wf = np.array([[0.5, 0.5]])
    lstm.Wi = np.array([[0.5, 0.5]])
    lstm.Wc = np.array([[0.3, 0.3]])
    lstm.Wo = np.array([[0.5, 0.5]])
    lstm.bf = np.array([[0.1]])
    lstm.bi = np.array([[0.1]])
    lstm.bc = np.array([[0.1]])
    lstm.bo = np.array([[0.1]])

    outputs, final_h, final_c = lstm.forward(input_sequence, initial_hidden_state, initial_cell_state)

    print(final_h)

    input_sequence = np.array([[0.1, 0.2], [0.3, 0.4]])
    initial_hidden_state = np.zeros((2, 1))
    initial_cell_state = np.zeros((2, 1))

    lstm = LSTM(input_size=2, hidden_size=2)

    # # Set weights and biases for reproducibility
    lstm.Wf = np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]])
    lstm.Wi = np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]])
    lstm.Wc = np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]])
    lstm.Wo = np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]])
    lstm.bf = np.array([[0.1], [0.2]])
    lstm.bi = np.array([[0.1], [0.2]])
    lstm.bc = np.array([[0.1], [0.2]])
    lstm.bo = np.array([[0.1], [0.2]])

    outputs, final_h, final_c = lstm.forward(input_sequence, initial_hidden_state, initial_cell_state)

    print(final_h)