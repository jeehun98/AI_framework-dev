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
        self.output_shape = []
        self.layer_name = "conv2d"
    
    # 가중치 생성
    def build(self, input_shape):
        # 필터의 개수만큼 임의의 가중치 생성
        # print(self.input_shape, "인풋 쉐이프")
        # 입력 데이터의 차원 수 
        in_channels = input_shape[2]    

        self.weights = np.random.randn(self.filters, self.kernel_size[0], self.kernel_size[1], in_channels)

        # 필터별 가중치 생성
        if self.use_biss:
            self.bias = np.random.rand(self.filters)

        self.call_output_shape(input_shape)
        super().build()

    # 출력 차원의 크기를 미리 계산한다.
    def call_output_shape(self, input_shape):
        input_height, input_width, input_channels = input_shape
        filter_height, filter_width = self.kernel_size
        stride_height, stride_width = self.strides

        if self.padding == 'same':
            output_height = int((input_height + stride_height - 1) / stride_height)
            output_width = int((input_width + stride_width - 1) / stride_width)

        elif self.padding == 'valid':
            output_height = int((input_height - filter_height) / stride_height) + 1
            output_width = int((input_width - filter_width) / stride_width) + 1

        else:
            raise ValueError("Invalid padding type. Use 'same' or 'valid'.")


        output_channels = self.filters  # 필터 개수만큼 출력 채널 생성

        # 출력 차원을 저장하거나 반환
        self.output_shape = (output_height, output_width, output_channels)
      
        # return self.output_shape
    
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
        
        # conv2d 함수 호출
        # node_list 는 각 성분을 이루는 요소들임, 개많아
        x, self.node_list = convolution.conv2d(input_data, self.weights, stride, self.padding)

        self.output_shape = x.shape

        return x