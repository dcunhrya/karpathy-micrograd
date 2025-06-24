import random
from micrograd.engine import Value

class Gradient:
    def zero_grad(self):
        # Reset the gradients of the parameters
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        # Return an empty list by default, to be overridden in subclasses
        return []
    
class Neuron(Gradient):
    def __init__(self, n_in):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(n_in)]  # Weights
        self.b = Value(random.uniform(-1, 1))  # Bias

    def __call__(self, x):
        # Forward pass: compute the output of the neuron
        act = sum((wi * x_i for wi, x_i in zip(self.w, x)), self.b)
        return act.tanh()
    
    def parameters(self):
        # Return the weights and bias of the neuron
        return self.w + [self.b]
    
class Layer(Gradient):
    def __init__(self, n_in, n_out):
        self.neurons = [Neuron(n_in) for _ in range(n_out)]  # Create n_out neurons

    def __call__(self, x):
        # Forward pass: compute the output of the layer
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        # Return the parameters (weights and biases) of all neurons in the layer
        params = []
        for n in self.neurons:
            params.extend(n.parameters())
        return params
    
class MLP(Gradient):
    def __init__(self, n_in, n_outs):
        sz = [n_in] + n_outs  # Hidden layer
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(n_outs))]  # Output layer

    def __call__(self, x):
        # Forward pass: compute the output of the MLP
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        # Return the parameters (weights and biases) of all layers in the MLP
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params