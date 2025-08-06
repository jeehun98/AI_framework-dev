import cupy as cp
from dev.layers.layer import Layer

import sys
sys.path.append("C:/Users/owner/Desktop/AI_framework-dev/dev/backend/graph_executor")
import graph_executor as ge  # Pybind11 모듈

Shape = ge.Shape
OpExtraParams = ge.OpExtraParams

class Dense(Layer):
    def __init__(self, units, activation=None, input_shape=None, name=None, initializer='he', **kwargs):
        super().__init__(name=name, **kwargs)
        self.layer_name = "dense"
        self.units = units
        self.activation = activation
        self.initializer = initializer
        self.input_shape = input_shape

        self.weights = None
        self.bias = None
        self.last_input = None
        self.last_z = None
        self.dW = None
        self.db = None

        # ✅ GraphCompiler 관련 변수
        self.name = name or f"dense_{id(self)}"
        self.weight_var = f"{self.name}_W"
        self.bias_var = f"{self.name}_b"
        self.output_var = f"{self.name}_out"

        # ✅ Activation name caching
        self.activation_name = getattr(self.activation, "__name__", str(self.activation)).lower() if self.activation else None

    def build(self, input_shape):
        self.input_shape = input_shape
        input_dim = input_shape[1]

        if self.initializer == 'zeros':
            self.weights = cp.zeros((input_dim, self.units), dtype=cp.float32)
        elif self.initializer == 'ones':
            self.weights = cp.ones((input_dim, self.units), dtype=cp.float32)
        elif self.initializer == 'uniform':
            limit = 0.05
            self.weights = cp.random.uniform(-limit, limit, (input_dim, self.units)).astype(cp.float32)
        elif self.initializer == 'normal':
            self.weights = cp.random.normal(0.0, 0.05, (input_dim, self.units)).astype(cp.float32)
        elif self.initializer == 'xavier':
            limit = cp.sqrt(6.0 / (input_dim + self.units))
            self.weights = cp.random.uniform(-limit, limit, (input_dim, self.units)).astype(cp.float32)
        elif self.initializer == 'he':
            std = cp.sqrt(2.0 / input_dim)
            self.weights = cp.random.normal(0.0, std, (input_dim, self.units)).astype(cp.float32)
        elif self.initializer == 'lecun':
            std = cp.sqrt(1.0 / input_dim)
            self.weights = cp.random.normal(0.0, std, (input_dim, self.units)).astype(cp.float32)
        elif self.initializer == 'small_uniform':
            self.weights = cp.random.uniform(-1e-3, 1e-3, (input_dim, self.units)).astype(cp.float32)
        else:
            raise ValueError(f"[Dense] Unknown initializer: {self.initializer}")

        # NaN 방어
        if cp.isnan(self.weights).any() or cp.isinf(self.weights).any():
            raise RuntimeError("[Init] Weight contains NaN or Inf")

        self.bias = cp.random.uniform(-1e-3, 1e-3, (1, self.units)).astype(cp.float32)

        self.built = True
        self.output_shape = self.compute_output_shape(input_shape)



    def __call__(self, x):
        if not self.built:
            self.build(x.shape)
        return self.call(x)

    def call(self, x):
        self.last_input = x
        z = cp.dot(x, self.weights) + self.bias
        self.last_z = z
        return self._apply_activation(z)

    def backward(self, grad_output):
        grad_output = cp.asarray(grad_output, dtype=cp.float32)

        if self.activation_name:
            grad_output = self._apply_activation_grad(grad_output, self.last_z)

        self.dW = cp.dot(self.last_input.T, grad_output)
        self.db = cp.sum(grad_output, axis=0, keepdims=True)
        dx = cp.dot(grad_output, self.weights.T)
        return dx

    def update(self, optimizer):
        reg_term = self.apply_regularizer(self.weights)
        optimizer.update(self.weights, self.dW + reg_term, self.bias, self.db)

    def _apply_activation(self, z):
        if self.activation is None:
            return z
        return self.activation(z)

    def _apply_activation_grad(self, grad_output, z):
        if self.activation_name == "sigmoid":
            sig = 1 / (1 + cp.exp(-z))
            return grad_output * sig * (1 - sig)
        elif self.activation_name == "relu":
            return grad_output * (z > 0)
        elif self.activation_name == "tanh":
            return grad_output * (1 - cp.tanh(z) ** 2)
        else:
            raise ValueError(f"Unsupported activation grad: {self.activation_name}")

    def compute_output_shape(self, input_shape):
        if input_shape is None or len(input_shape) != 2:
            raise ValueError(f"Dense layer expects input shape to be 2D (batch_size, input_dim), got {input_shape}")
        batch_size, _ = input_shape
        return (batch_size, self.units)

    def to_e_matrix(self, input_id):
        if self.input_shape is None:
            raise ValueError("[Dense] input_shape is None. Did you forget to call build()?")

        batch, input_dim = map(int, self.input_shape)
        units = int(self.units)

        weight_id = self.weight_var
        bias_id = self.bias_var
        linear_out_id = f"{self.name}_linear"
        output_id = self.output_var
        preact_id = f"{self.name}_preact"

        extra = OpExtraParams()

        e_block = [
            {
                "op_type": 0,  # matmul
                "input_id": input_id,
                "param_id": weight_id,
                "output_id": linear_out_id,
                "extra_params": extra
            },
            {
                "op_type": 1,  # add
                "input_id": linear_out_id,
                "param_id": bias_id,
                "output_id": preact_id if self.activation_name else output_id,
                "extra_params": extra
            }
        ]

        if self.activation_name:
            activation_map = {
                "relu": 2,
                "sigmoid": 3,
                "tanh": 4
            }
            if self.activation_name not in activation_map:
                raise ValueError(f"Unsupported activation: {self.activation_name}")

            e_block.append({
                "op_type": activation_map[self.activation_name],
                "input_id": preact_id,
                "param_id": "",
                "output_id": output_id,
                "extra_params": extra
            })

        weights = {weight_id: self.weights}
        biases = {bias_id: self.bias}

        shape_map = {
            input_id: Shape(batch, input_dim),
            weight_id: Shape(input_dim, units),
            bias_id: Shape(1, units),
            linear_out_id: Shape(batch, units),
            output_id: Shape(batch, units)
        }
        if self.activation_name:
            shape_map[preact_id] = Shape(batch, units)

        return e_block, weights, biases, output_id, shape_map
