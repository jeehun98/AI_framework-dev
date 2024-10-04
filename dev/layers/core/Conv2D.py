import os
os.add_dll_directory("C:\\msys64\\mingw64\\bin")

from dev.layers.layer import Layer
from dev.backend.convolution import convolution


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
        input_shape=None,
        **kwargs
    ):
        super().__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.activation = activation
        self.use_biss = use_bias
        self.input_shape = input_shape
        self.weights = []
        self.bias = []
        self.node_list = []
        self.built = False
        self.trainable = True
    
    # 가중치 생성
    def build(self):
        # 필터의 개수만큼 임의의 가중치 생성
        # print(self.input_shape, "인풋 쉐이프")
        # 입력 데이터의 차원 수 
        in_channels = self.input_shape[2]

        self.weights = np.random.randn(self.filters, self.kernel_size[0], self.kernel_size[1], in_channels)

        # 필터별 가중치 생성
        if self.use_biss:
            self.bias = np.random.rand(self.filters)

        super().build()
    
    def call(self, input_data):
        """
        Conv2D 연산, 필터 개수만큼의 출력 필요
        """
        # 전달하는 데이터들 타입 확인 및 변환
        input_data = np.asarray(input_data, dtype=np.float64)
        self.weights = np.asarray(self.weights, dtype=np.float64)
        stride = self.strides[0] if isinstance(self.strides, (tuple, list)) else self.strides
        if not isinstance(self.node_list, list):
            self.node_list = []

        # print("타입확인", type(input_data), input_data.dtype, type(self.weights), self.weights.dtype, type(stride), type(self.padding), type(self.node_list))
        print(input_data.shape, self.weights.shape)
        
        # conv2d 함수 호출
        x, self.node_list = convolution.conv2d(input_data, self.weights, stride, self.padding, self.node_list)
        print("call 끝", x.shape)

        return x