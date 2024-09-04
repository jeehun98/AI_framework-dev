import os
os.add_dll_directory("C:\\msys64\\mingw64\\bin")

from dev.layers.layer import Layer
from dev import activations

from dev.backend.operaters import operations_matrix

import numpy as np


class Dense(Layer):
    # dense layer 에 필요한 내용이 뭐가 있을지, 추가될 수 있어용
    def __init__(self, units, activation=None, name=None, **kwargs):
        super().__init__(name)
        self.units = units
        self.output_shape = (units,)

        if activation is not None:
            self.activation = activations.get(activation)
        else:
            self.activation = activation
            
        self.weights = None
        self.bias = None
        

    def get_config(self):
        base_config = super().get_config()
        config = ({
            'class_name': self.__class__.__name__,
            'units': self.units,
            'activation': self.activation.__name__ if self.activation else None,
            'input_shape': self.input_shape,

        })
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    # 입력 차원에 따라 가중치와 편향을 초기화
    def build(self, input_shape):
        # input_shape가 (784,)와 같은 경우라면, 실제 필요한 것은 input_shape[0]
        # 모델 정보
        input_dim = input_shape[0]
        self.input_shape = input_shape

        # 가중치 생성
        self.weights = np.random.randn(input_dim, self.units)
        self.bias = np.random.rand()
        super().build()


    def call(self, inputs):
        """
        dense 층 연산

        Parameters:
        inputs (n, p) : p 차원 n개의 데이터 입력
        
        Returns (n*, p*): dense 연산 결과
        """
        x = operations_matrix.matrix_multiply(inputs, self.weights)
        if self.bias is not None:
            x = operations_matrix.matrix_add(x, np.tile(self.bias, x.shape))
        if self.activation is not None:
            x = self.activation(x)
            x = np.reshape(x,(-1,self.units))
        return x