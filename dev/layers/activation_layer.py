import numpy as np
from dev.layers.layer import Layer
from dev.utils.load_cuda import load_activations_cuda  # CUDA Pybind11 모듈 로더


class Activation(Layer):
    def __init__(self, activation, **kwargs):
        super().__init__(**kwargs)
        self.activation_name = activation.lower()
        self.trainable = False
        self.layer_name = "activation"
        self.last_z = None
        self.input_shape = None

        # ✅ CUDA 모듈 로딩
        self.activations_cuda = load_activations_cuda()

    def call(self, inputs):
        # ✅ CuPy 배열 그대로 사용
        self.input_shape = inputs.shape
        self.last_z = inputs  # CuPy로 유지

        try:
            self.activations_cuda.apply_activation(inputs, self.activation_name)
        except Exception as e:
            raise RuntimeError(f"[ERROR] CUDA 활성화 실패: {e}")
        return inputs  # in-place 연산

    def backward(self, grad_output):
        if isinstance(grad_output, tuple):
            grad_output = grad_output[0]

        try:
            self.activations_cuda.apply_activation_grad(self.last_z, grad_output, self.activation_name)
        except Exception as e:
            raise RuntimeError(f"[ERROR] CUDA backward 실패: {e}")
        return grad_output  # in-place 수정됨

    def compute_output_shape(self, input_shape):
        return input_shape

    def build(self, input_shape):
        self.input_shape = input_shape
        self.output_shape = input_shape
        super().build(input_shape)
