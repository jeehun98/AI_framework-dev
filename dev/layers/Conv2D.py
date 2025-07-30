import cupy as cp
import numpy as np
from dev.layers.layer import Layer

import sys
sys.path.append("C:/Users/owner/Desktop/AI_framework-dev/dev/backend/graph_executor")
import graph_executor as ge  # Pybind11 모듈

Shape = ge.Shape
OpExtraParams = ge.OpExtraParams


class Conv2D(Layer):
    def __init__(self, filters, kernel_size, strides=(1, 1), padding='valid',
                 activation=None, input_shape=None, name=None, initializer='he', **kwargs):
        super().__init__(name=name, **kwargs)
        self.layer_name = "conv2d"
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding.lower()
        self.activation = activation
        self.initializer = initializer
        self.input_shape = input_shape

        self.name = name or f"conv2d_{id(self)}"
        self.weight_var = f"{self.name}_W"
        self.bias_var = f"{self.name}_b"
        self.output_var = f"{self.name}_out"

        self.weights = None
        self.bias = None

        self.activation_name = getattr(self.activation, "__name__", str(self.activation)).lower() if self.activation else None

    def __call__(self, x):
        if not self.built:
            self.build(x.shape)
        return self.call(x)

    def build(self, input_shape):
        if len(input_shape) == 3:
            input_shape = (*input_shape, 1)  # (b, h, w) → (b, h, w, 1)
        if len(input_shape) != 4:
            raise ValueError(f"[Conv2D] build: expected input shape (b, h, w, c), got {input_shape}")

        self.input_shape = input_shape
        b, in_h, in_w, in_c = input_shape
        kh, kw = self.kernel_size
        sh, sw = self.strides

        if self.padding == 'valid':
            self.out_h = (in_h - kh) // sh + 1
            self.out_w = (in_w - kw) // sw + 1
        elif self.padding == 'same':
            self.out_h = int(np.ceil(in_h / sh))
            self.out_w = int(np.ceil(in_w / sw))
        else:
            raise ValueError(f"Unsupported padding: {self.padding}")

        if self.out_h <= 0 or self.out_w <= 0:
            raise ValueError(f"[Conv2D] Invalid output shape: {self.out_h}x{self.out_w}")

        self.output_shape = (b, self.out_h, self.out_w, self.filters)

        fan_in = in_c * kh * kw
        if self.initializer == 'ones':
            self.weights = cp.ones((self.filters, in_c, kh, kw), dtype=cp.float32)
        elif self.initializer == 'zeros':
            self.weights = cp.zeros((self.filters, in_c, kh, kw), dtype=cp.float32)
        else:
            limit = cp.sqrt(2 / fan_in)
            self.weights = cp.random.randn(self.filters, in_c, kh, kw).astype(cp.float32) * limit

        self.bias = cp.zeros((self.filters,), dtype=cp.float32)
        self.built = True

    def call(self, x):
        raise NotImplementedError("Forward pass is done by CUDA backend")

    def backward(self, grad_output):
        raise NotImplementedError("Backward pass is handled by CUDA backend")

    def update(self, optimizer):
        optimizer.update(self.weights, self.dW, self.bias, self.db)

    def compute_output_shape(self, input_shape):
        if len(input_shape) == 3:
            input_shape = (*input_shape, 1)
        b, in_h, in_w, in_c = input_shape
        kh, kw = self.kernel_size
        sh, sw = self.strides

        if self.padding == 'valid':
            out_h = (in_h - kh) // sh + 1
            out_w = (in_w - kw) // sw + 1
        elif self.padding == 'same':
            out_h = int(np.ceil(in_h / sh))
            out_w = int(np.ceil(in_w / sw))
        else:
            raise ValueError(f"[Conv2D] Unknown padding: {self.padding}")

        return (b, out_h, out_w, self.filters)

    def to_e_matrix(self, input_id):
        if self.input_shape is None:
            raise ValueError("[Conv2D] input_shape is None. Did you forget to call build()?")

        weight_id = self.weight_var
        bias_id = self.bias_var
        conv_out_id = f"{self.name}_conv"
        output_id = self.output_var
        preact_id = f"{self.name}_preact"

        b, in_h, in_w, in_c = self.input_shape
        kh, kw = self.kernel_size
        out_h, out_w = self.out_h, self.out_w

        extra = OpExtraParams()
        extra.batch_size = b
        extra.input_h = in_h
        extra.input_w = in_w
        extra.kernel_h = kh
        extra.kernel_w = kw
        extra.input_c = in_c
        extra.output_c = self.filters

        e_block = [
            {
                "op_type": 6,  # conv2d
                "input_id": input_id,
                "param_id": weight_id,
                "output_id": conv_out_id,
                "extra_params": extra
            },
            {
                "op_type": 1,  # add
                "input_id": conv_out_id,
                "param_id": bias_id,
                "output_id": preact_id if self.activation else output_id
            }
        ]

        if self.activation:
            activation_map = {"relu": 2, "sigmoid": 3, "tanh": 4}
            if self.activation_name not in activation_map:
                raise ValueError(f"Unsupported activation: {self.activation_name}")
            e_block.append({
                "op_type": activation_map[self.activation_name],
                "input_id": preact_id,
                "param_id": "",
                "output_id": output_id
            })

        shape_map = {
            input_id: Shape(b, in_c * in_h * in_w),
            weight_id: Shape(self.filters, in_c * kh * kw),
            bias_id: Shape(1, self.filters),
            conv_out_id: Shape(b, self.filters * out_h * out_w),
            output_id: Shape(b, self.filters * out_h * out_w)
        }

        if self.activation:
            shape_map[preact_id] = Shape(b, self.filters * out_h * out_w)

        weights = {weight_id: self.weights.reshape(self.filters, -1)}
        biases = {bias_id: self.bias.reshape(1, -1)}

        return e_block, weights, biases, output_id, shape_map
