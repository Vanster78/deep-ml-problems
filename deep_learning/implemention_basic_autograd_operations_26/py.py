class Value:
	def __init__(self, data, _children=(), _op=''):
		self.data = data
		self.grad = 0
		self._backward = lambda: None
		self._prev = set(_children)
		self._op = _op
	def __repr__(self):
		return f"Value(data={self.data}, grad={self.grad})"

	def __add__(self, other):
		v = Value(self.data + other.data, _children=(self, other), _op='+')
		def backward():
			self.grad += v.grad
			other.grad += v.grad
		v._backward = backward
		return v

	def __mul__(self, other):
		v = Value(self.data * other.data, _children=(self, other), _op='*')
		def backward():
			self.grad += other.data * v.grad
			other.grad += self.data * v.grad
		v._backward = backward
		return v

	def relu(self):
		pos = self.data > 0
		v = Value(self.data if pos else 0.0, _children=(self,), _op='relu')
		def backward():
			self.grad += pos * v.grad
		v._backward = backward
		return v

	def backward(self):
		def topological_sort(v: Value):
			if v in visited:
				return v
			visited.add(v)
			for c in v._prev:
				topological_sort(c)
			ordered_values.append(v)
		ordered_values = []
		visited = set()
		topological_sort(self)
		self.grad = 1.0
		for v in reversed(ordered_values):
			v._backward()
