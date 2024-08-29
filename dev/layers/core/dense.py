from dev.layers.layer import Layer
from dev import activations
import numpy as np

class Dense(Layer):
    def __init__(self, units, activation=None, name=None, **kwargs):
        super().__init__(name)
        self.units = units
        self.output_shape = (units,)
        self.activation = activations.get(activation)
        self.weights = None
        self.bias = None

    def get_config(self):
        base_config = super().get_config()
        config = ({
            'units': self.units,
            'activation': self.activation,
        })
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def build(self, input_shape):
        # 입력 차원에 따라 가중치와 편향을 초기화합니다.
        # 입력 차원은 이전 layer 의 출력 차원이 될 것
        input_dim = input_shape[-1]
        self.weights = np.random.randn(input_dim, self.units)
        self.bias = np.zeros((self.units,))
        #super().build(input_shape)

    def call(self, inputs):
        # 가중치와 편향을 적용하고 활성화 함수를 통해 출력합니다.
        z = np.dot(inputs, self.weights) + self.bias
        if self.activation:
            return self.activation(z)
        return z

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)


