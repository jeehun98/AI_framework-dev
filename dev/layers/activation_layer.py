import cupy as cp
from dev.layers.layer import Layer
from dev.utils.load_cuda import load_activations_cuda

import sys
sys.path.append("C:/Users/owner/Desktop/AI_framework-dev/dev/backend/graph_executor")
import graph_executor as ge  # Pybind11 모듈

Shape = ge.Shape
OpExtraParams = ge.OpExtraParams

ACTIVATION_OP_TYPES = {
    'relu': 2,
    'sigmoid': 3,
    'tanh': 4
}

class Activation(Layer):
    def __init__(self, activation, name=None, use_backend_init=True, **kwargs):
        super().__init__(name=name, **kwargs)
        self.activation_name = activation.lower()
        if self.activation_name not in ACTIVATION_OP_TYPES:
            raise ValueError(f"[Activation] Unsupported activation: {self.activation_name}")
        
        self.trainable = False
        self.layer_name = "activation"
        self.last_z = None
        self.input_shape = None
        self.use_backend_init = use_backend_init

        self.name = name or f"activation_{id(self)}"
        self.output_var = f"{self.name}_out"

    def __call__(self, x):
        if not self.built:
            self.build(x.shape)
        return self.call(x)

    def build(self, input_shape):
        self.input_shape = input_shape
        self.output_shape = self.compute_output_shape(input_shape)
        self.built = True

    def call(self, inputs):
        inputs = cp.asarray(inputs, dtype=cp.float32)
        self.last_z = inputs
        self.input_shape = inputs.shape

        if cp.isnan(inputs).any() or cp.isinf(inputs).any():
            print("[WARNING] 입력에 NaN 또는 inf 포함됨")

        try:
            self.activations_cuda.apply_activation(inputs, self.activation_name)
        except Exception as e:
            raise RuntimeError(f"[Activation] CUDA forward 실패: {e}")
        return inputs

    def backward(self, grad_output):
        if isinstance(grad_output, tuple):
            grad_output = grad_output[0]

        grad_output = cp.asarray(grad_output, dtype=cp.float32)

        try:
            self.activations_cuda.apply_activation_grad(self.last_z, grad_output, self.activation_name)
        except Exception as e:
            raise RuntimeError(f"[Activation] CUDA backward 실패: {e}")
        return grad_output

    def compute_output_shape(self, input_shape):
        return input_shape

    def to_e_matrix(self, input_id):
        if not self.input_shape or len(self.input_shape) != 2:
            raise ValueError(f"[Activation] input_shape must be 2D (batch, features), got {self.input_shape}")

        output_id = self.output_var
        op_type = ACTIVATION_OP_TYPES[self.activation_name]
        extra = OpExtraParams()

        e_block = [{
            "op_type": op_type,
            "input_id": input_id,
            "param_id": "",  # ❗ Pybind11 expects string
            "output_id": output_id,
            "extra_params": extra
        }]

        shape_map = {
            input_id: Shape(*map(int, self.input_shape)),
            output_id: Shape(*map(int, self.input_shape))
        }

        return e_block, {}, {}, output_id, shape_map
