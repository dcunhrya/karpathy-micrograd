import math

class Value:
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None  # Function to compute the gradient
        self._prev = set(_children)  # Previous nodes in the computation graph
        self._op = _op  # Operation that produced this value
        self.label = label  # Optional label for the value

    def __repr__(self):
        # Print out the value in a readable format
        return f"Value(data={self.data})"
    
    def __add__(self, other):
        # Ensure that the other operand is also a Value instance in case integer
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, _children=(self, other), _op='+')
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out
    
    def __mul__(self, other):
        # Ensure that the other operand is also a Value instance in case integer
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, _children=(self, other), _op='*')
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out
    
    def exp(self):
        out = Value(math.exp(self.data), _children=(self,), _op='exp')
        def _backward():
            self.grad += out.grad * out.data
        out._backward = _backward
        return out
    
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "Exponent must be a scalar"
        out = Value(self.data ** other, _children=(self,), _op='**')
        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad
        out._backward = _backward
        return out
    
    def __sub__(self, other):
        return self + (-other)
    
    def __neg__(self):
        # Negation of a Value instance
        return self * -1
    
    def __truediv__(self, other):
        return self * other**-1
    
    def __rmul__(self, other):
        # Handle the case where the other operand is a scalar
        return self * other
    
    def __radd__(self, other):
        # Handle the case where the other operand is a scalar
        return self + other

    def tanh(self):
        out = Value(math.tanh(self.data), _children=(self,), _op='tanh')
        def _backward():
            self.grad = (1 - out.data**2) * out.grad
        out._backward = _backward
        return out
    
    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward
        return out
    
    def sigmoid(self):
        out = Value(1 / (1 + math.exp(-self.data)), _children=(self,), _op='sigmoid')
        def _backward():
            self.grad += out.data * (1 - out.data) * out.grad
        out._backward = _backward
        return out
    
    def backward(self):
        # Toplogical sort of the computation graph
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad = 1.0  # Initialize the gradient of the output node
        for node in reversed(topo):
            node._backward()
        