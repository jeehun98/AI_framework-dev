import numpy as np
from dev.layers.layer import Layer
from dev.backend.backend_ops.activations import activations as cuda_activations


class Activation(Layer):
    def __init__(self, activation, **kwargs):
        super().__init__(**kwargs)
        self.activation_name = activation.lower()
        self.trainable = False
        self.layer_name = "activation"
        self.last_z = None

        # ✅ CUDA 함수 매핑 (forward, backward 각각)
        self.cuda_forward = {
            "relu": cuda_activations.relu,
            "sigmoid": cuda_activations.sigmoid,
            "tanh": cuda_activations.tanh,
        }

        self.cuda_backward = {
            "relu": cuda_activations.relu_grad,
            "sigmoid": cuda_activations.sigmoid_grad,
            "tanh": cuda_activations.tanh_grad,
        }

    def call(self, inputs):
        self.last_z = inputs.astype(np.float32)
        try:
            func = self.cuda_forward[self.activation_name]
            output = func(self.last_z)
        except KeyError:
            raise NotImplementedError(f"[ERROR] CUDA 활성화 미지원: '{self.activation_name}'")
        self.output_shape = output.shape
        return output

    def backward(self, grad_output):
        grad_output = grad_output.astype(np.float32)
        z = self.last_z

        try:
            grad_func = self.cuda_backward[self.activation_name]
            return grad_func(z, grad_output)
        except KeyError:
            raise NotImplementedError(f"[ERROR] '{self.activation_name}' 미분 미지원")

    def compute_output_shape(self, input_shape):
        return input_shape

    def build(self, input_shape):
        self.input_shape = input_shape
        self.output_shape = input_shape
        super().build(input_shape)
