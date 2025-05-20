# dev/layers/activation_layer.py

import numpy as np
from dev.layers.layer import Layer
from dev.backend.backend_ops.activations import activations as cuda_activations

class Activation(Layer):
    def __init__(self, activation, **kwargs):
        super().__init__(**kwargs)
        self.activation_name = activation
        self.trainable = False
        self.layer_name = "activation"

        # ✅ CUDA 함수 매핑
        self.cuda_functions = {
            "relu": cuda_activations.relu,
            "sigmoid": cuda_activations.sigmoid,
            "tanh": cuda_activations.tanh,
        }

        self.last_z = None  # ✅ 입력 저장 (역전파용)

    def call(self, inputs):
        # ✅ 입력 저장 (activation 전에 z 로 사용됨)
        self.last_z = inputs.astype(np.float32)

        # ✅ CUDA 활성화 함수 호출
        try:
            activation_func = self.cuda_functions[self.activation_name]
            output = activation_func(self.last_z)
        except KeyError:
            raise NotImplementedError(f"[ERROR] CUDA 활성화 미지원: '{self.activation_name}'")

        self.output_shape = output.shape
        return output

    def backward(self, grad_output):
        grad_output = grad_output.astype(np.float32)
        z = self.last_z

        if self.activation_name == "sigmoid":
            sig = 1 / (1 + np.exp(-z))
            return grad_output * sig * (1 - sig)
        elif self.activation_name == "relu":
            return grad_output * (z > 0).astype(np.float32)
        elif self.activation_name == "tanh":
            return grad_output * (1 - np.tanh(z) ** 2)
        else:
            raise NotImplementedError(f"[ERROR] '{self.activation_name}' 미분 미지원")

    def compute_output_shape(self, input_shape):
        return input_shape

    def build(self, input_shape):
        self.input_shape = input_shape
        self.output_shape = input_shape
        super().build(input_shape)
