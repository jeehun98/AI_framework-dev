from dev.layers.layer import Layer
from dev import activations
import numpy as np

class Dense(Layer):
    def __init__(self, units, activation=None, name=None, **kwargs):
        super().__init__(name)
        self.units = units
        self.activation = activations.get(activation)
        self.weights = None
        self.bias = None

    def build(self, input_shape):
        # 입력 차원에 따라 가중치와 편향을 초기화합니다.
        input_dim = input_shape[-1]
        self.weights = np.random.randn(input_dim, self.units)
        self.bias = np.zeros((self.units,))
        super().build(input_shape)

    def call(self, inputs):
        # 가중치와 편향을 적용하고 활성화 함수를 통해 출력합니다.
        z = np.dot(inputs, self.weights) + self.bias
        if self.activation:
            return self.activation(z)
        return z

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)

    def get_config(self):
        base_config = super().get_config()
        config = ({
            'input_shape': self.input_shape,
            'activation': self.activation,
        })
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        activation = globals().get(config['activation'])
        return cls(units=config['units'], activation=activation, name=config['name'])
