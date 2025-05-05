import numpy as np
from dev.layers.layer import Layer
from dev.backend.backend_ops.activations import activations as cuda_activations

class ActivationMat(Layer):
    def __init__(self, activation, **kwargs):
        super().__init__(**kwargs)
        self.activation_name = activation
        self.layer_name = "activation"
        self.trainable = False
        self.input_dim = None
        self.output_dim = None

        # ✅ CUDA 함수 매핑
        self.cuda_functions = {
            "relu": cuda_activations.relu,
            "sigmoid": cuda_activations.sigmoid,
            "tanh": cuda_activations.tanh,
        }

    def build(self, input_dim):
        self.input_dim = input_dim
        self.output_dim = input_dim

    def call(self, inputs):
        inputs = np.atleast_2d(inputs).astype(np.float32)
        try:
            activation_func = self.cuda_functions[self.activation_name]
        except KeyError:
            raise NotImplementedError(f"'{self.activation_name}' CUDA 미지원")

        output = activation_func(inputs)
        self.output_dim = output.shape
        return output
