import cupy as cp
from dev.layers.layer import Layer

class Dense(Layer):
    def __init__(self, units, activation=None, name=None, initializer='he', **kwargs):
        super().__init__(name, **kwargs)
        self.units = units
        self.activation = activation
        self.initializer = initializer
        self.input_shape = None
        self.output_shape = (1, units)

        self.weights = None
        self.bias = None
        self.last_input = None
        self.last_z = None
        self.dW = None
        self.db = None

    def build(self, input_shape):
        self.input_shape = input_shape
        input_dim = input_shape[1]

        if self.initializer == 'ones':
            self.weights = cp.ones((input_dim, self.units), dtype=cp.float32)
        elif self.initializer == 'zeros':
            self.weights = cp.zeros((input_dim, self.units), dtype=cp.float32)
        elif self.initializer == 'he':
            stddev = cp.sqrt(2. / input_dim)
            self.weights = cp.random.randn(input_dim, self.units).astype(cp.float32) * stddev
        elif self.initializer == 'xavier':
            stddev = cp.sqrt(1. / input_dim)
            self.weights = cp.random.randn(input_dim, self.units).astype(cp.float32) * stddev
        else:
            raise ValueError(f"Unknown initializer: {self.initializer}")

        self.bias = cp.zeros((1, self.units), dtype=cp.float32)

    def call(self, input_data):
        input_data = cp.atleast_2d(cp.asarray(input_data, dtype=cp.float32))
        if self.weights is None:
            self.build(input_data.shape)

        self.last_input = input_data
        z = cp.dot(input_data, self.weights) + self.bias
        self.last_z = z

        if self.activation:
            z = self._apply_activation(z)

        return z

    def backward(self, grad_output):
        grad_output = cp.asarray(grad_output, dtype=cp.float32)

        if self.activation:
            grad_output = self._apply_activation_grad(grad_output, self.last_z)

        self.dW = cp.dot(self.last_input.T, grad_output)
        self.db = cp.sum(grad_output, axis=0, keepdims=True)
        dx = cp.dot(grad_output, self.weights.T)
        return dx

    def update(self, optimizer):
        optimizer.update(self.weights, self.dW, self.bias, self.db)

    def _apply_activation(self, z):
        name = self.activation.__name__.lower()
        if name == "sigmoid":
            return 1 / (1 + cp.exp(-z))
        elif name == "relu":
            return cp.maximum(0, z)
        elif name == "tanh":
            return cp.tanh(z)
        else:
            raise ValueError(f"Unsupported activation: {name}")

    def _apply_activation_grad(self, grad_output, z):
        name = self.activation.__name__.lower()
        if name == "sigmoid":
            sig = 1 / (1 + cp.exp(-z))
            return grad_output * sig * (1 - sig)
        elif name == "relu":
            return grad_output * (z > 0)
        elif name == "tanh":
            return grad_output * (1 - cp.tanh(z) ** 2)
        else:
            raise ValueError(f"Unsupported activation grad: {name}")
