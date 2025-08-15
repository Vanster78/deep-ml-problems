
import numpy as np
import copy
import math

# DO NOT CHANGE SEED
np.random.seed(42)

# DO NOT CHANGE LAYER CLASS
class Layer(object):

	def set_input_shape(self, shape):
    
		self.input_shape = shape

	def layer_name(self):
		return self.__class__.__name__

	def parameters(self):
		return 0

	def forward_pass(self, X, training):
		raise NotImplementedError()

	def backward_pass(self, accum_grad):
		raise NotImplementedError()

	def output_shape(self):
		raise NotImplementedError()

# Your task is to implement the Dense class based on the above structure
class Dense(Layer):
	def __init__(self, n_units, input_shape=None):
		
		self.layer_input = None
		self.input_shape = input_shape
		self.n_units = n_units
		self.trainable = True
		self.optimizer = None
		self.W = None
		self.w0 = None
	
	def initialize(self, optimizer):
		rng = 1 / self.input_shape[0] ** 0.5
		self.W = np.random.rand(self.input_shape[0], self.n_units) * 2 * rng - rng
		self.w0 = np.zeros((self.n_units,))
		self.optimizer = optimizer

	def forward_pass(self, x: np.ndarray):
		if self.trainable:
			self.layer_input = x

		return x @ self.W + self.w0

	def backward_pass(self, accum_grad):
		gx = accum_grad @ self.W.T
		if self.trainable:
			gW = self.layer_input.T @ accum_grad
			gw = accum_grad.sum(axis=0)
			self.W = self.optimizer.update(self.W, gW)
			self.w0 = self.optimizer.update(self.w0, gw)

		return gx

	def number_of_parameters(self) -> int:
		return self.W.shape[0] * self.W.shape[1] + self.w0.shape[0]
	
	def output_shape(self):
		return (self.n_units,)
