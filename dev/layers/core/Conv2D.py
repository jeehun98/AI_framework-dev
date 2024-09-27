import os
os.add_dll_directory("C:\\msys64\\mingw64\\bin")

from dev.layers.layer import Layer
from dev import activations
from dev.node.node import Node

from dev.backend.operaters import operations_matrix
from dev.backend.

import numpy as np

class Conv2D(Layer):
    def __init__(
        self, 
        filters,
        kernel_size,
        strides=(1,1),
        padding="valid",
        activation=None,
        use_bias=True,
        **kwargs
    ):
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.activation = activation
        self.use_biss = use_bias
        self.weights = []
        self.bias = []
    
    # 가중치 생성
    def build(self):
        # 필터의 개수만큼 임의의 가중치 생성
        for filter in range(self.filters):
            self.weights.append(np.random.randn(self.kernel_size[0], self.kernel_size[1]))

        # 필터별 가중치 생성
        if self.use_biss:
            self.bias = np.random.rand(self.filters)

        super().build()
    
    # convolution 연산 수행
    def call(self, input_data):
        """
        Conv2D 연산, 필터 개수만큼의 출력 필요

        
        """
        
        pass