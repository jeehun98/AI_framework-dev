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
            'class_name': self.__class__.__name__,
            'units': self.units,
            'activation': self.activation.__name__,
            'input_shape': self.input_shape,

        })
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    # 입력 차원에 따라 가중치와 편향을 초기화
    def build(self, input_shape):
        # input_shape가 (784,)와 같은 경우라면, 실제 필요한 것은 input_shape[0]입니다.
        input_dim = input_shape[0]
        self.input_shape = input_shape
        self.weights = np.random.randn(input_dim, self.units)
        self.bias = np.zeros((self.units,))
        super().build()


    def call(self, inputs):
        # 가중치와 편향을 적용하고 활성화 함수를 통해 출력합니다.
        z = np.dot(inputs, self.weights) + self.bias
        if self.activation:
            return self.activation(z)
        return z
