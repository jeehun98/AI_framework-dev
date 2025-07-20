import cupy as cp
from dev.layers.layer import Layer

import sys
sys.path.append("C:/Users/owner/Desktop/AI_framework-dev/dev/backend/graph_executor")
import graph_executor as ge

Shape = ge.Shape

class Conv2D(Layer):
    def __init__(self, filters, kernel_size, strides=(1, 1), padding='valid',
                 activation=None, input_shape=None, name=None, initializer='he', **kwargs):
        super().__init__(name, **kwargs)
        self.layer_name = "conv2d"
        self.filters = filters
        self.kernel_size = kernel_size  # (kh, kw)
        self.strides = strides          # (sh, sw)
        self.padding = padding.lower()
        self.activation = activation
        self.initializer = initializer
        self.input_shape = input_shape  # (batch, h, w, c)

        self.name = name or f"conv2d_{id(self)}"
        self.input_var = None
        self.weight_var = f"{self.name}_W"
        self.bias_var = f"{self.name}_b"
        self.output_var = f"{self.name}_out"

        self.weights = None
        self.bias = None

    def build(self, input_shape):
        self.input_shape = input_shape
        _, in_h, in_w, in_c = input_shape
        kh, kw = self.kernel_size
        self.out_h = (in_h - kh) // self.strides[0] + 1 if self.padding == 'valid' else in_h
        self.out_w = (in_w - kw) // self.strides[1] + 1 if self.padding == 'valid' else in_w

        if self.initializer == 'ones':
            self.weights = cp.ones((self.filters, in_c, kh, kw), dtype=cp.float32)
        elif self.initializer == 'zeros':
            self.weights = cp.zeros((self.filters, in_c, kh, kw), dtype=cp.float32)
        else:
            fan_in = in_c * kh * kw
            limit = cp.sqrt(2 / fan_in)
            self.weights = cp.random.randn(self.filters, in_c, kh, kw).astype(cp.float32) * limit

        self.bias = cp.zeros((self.filters,), dtype=cp.float32)

    def call(self, x):
        raise NotImplementedError("Forward pass is done by CUDA backend")

    def backward(self, grad_output):
        raise NotImplementedError("Backward pass is handled by CUDA backend")

    def update(self, optimizer):
        optimizer.update(self.weights, self.dW, self.bias, self.db)

    def to_e_matrix(self, input_id):
        weight_id = f"{self.name}_W"
        bias_id = f"{self.name}_b"
        conv_out_id = f"{self.name}_conv"
        output_id = f"{self.name}_out"
        preact_id = f"{self.name}_preact"

        e_block = [
            {
                "op_type": 5,  # conv2d 연산 정의
                "input_id": input_id,
                "param_id": weight_id,
                "output_id": conv_out_id,
                "stride": self.strides,
                "padding": self.padding
            },
            {
                "op_type": 1,  # add (bias)
                "input_id": conv_out_id,
                "param_id": bias_id,
                "output_id": preact_id if self.activation else output_id
            }
        ]

        if self.activation:
            activation_name = getattr(self.activation, "__name__", str(self.activation)).lower()
            activation_map = {
                "relu": 2,
                "sigmoid": 3,
                "tanh": 4
            }
            if activation_name not in activation_map:
                raise ValueError(f"Unsupported activation: {activation_name}")
            e_block.append({
                "op_type": activation_map[activation_name],
                "input_id": preact_id,
                "param_id": "",
                "output_id": output_id
            })

        # Shape 정보 구성
        b, in_h, in_w, in_c = self.input_shape
        kh, kw = self.kernel_size
        out_h = self.out_h
        out_w = self.out_w

        shape_map = {
            input_id: Shape(b, in_c, in_h, in_w),
            weight_id: Shape(self.filters, in_c, kh, kw),
            bias_id: Shape(1, self.filters),
            conv_out_id: Shape(b, self.filters, out_h, out_w)
        }

        if self.activation:
            shape_map[preact_id] = Shape(b, self.filters, out_h, out_w)

        shape_map[output_id] = Shape(b, self.filters, out_h, out_w)

        weights = {weight_id: self.weights}
        biases = {bias_id: self.bias}

        return e_block, weights, biases, output_id, shape_map
