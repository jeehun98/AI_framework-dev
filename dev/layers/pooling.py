import os
os.add_dll_directory("C:\\msys64\\mingw64\\bin")

from dev.layers.layer import Layer
from dev.backend.pooling import pooling

class Pooling(Layer):
    def __init__(
        self,
        pool_size=(2, 2),
        strides=(1,1),
        padding="valid",
        pool_mode="max",
        **kwargs
    ):
        super().__init__()
        self.pool_size = pool_size
        self.strides = strides
        self.padding = padding
        self.pool_mode = pool_mode
        self.built = True,
        self.trainable = True,
        self.node_list = []
        self.layer_name = "pooling"
        self.output_shape = []
        

    def build(self, input_shape):
        self.call_output_shape(input_shape)
        super().build()

    def call_output_shape(self, input_shape):
        """
        Pooling 레이어의 출력 크기를 계산하는 함수.
        input_shape: 입력 특성 맵의 크기 (height, width, channels)
        """
        input_height, input_width, input_channels = input_shape
        pool_height, pool_width = self.pool_size
        stride_height, stride_width = self.strides

        if self.padding == 'same':
            output_height = int((input_height + stride_height - 1) / stride_height)
            output_width = int((input_width + stride_width - 1) / stride_width)
        elif self.padding == 'valid':
            output_height = int((input_height - pool_height) / stride_height) + 1
            output_width = int((input_width - pool_width) / stride_width) + 1
        else:
            raise ValueError("Invalid padding type. Use 'same' or 'valid'.")

        # 채널 수는 변하지 않음
        output_channels = input_channels

        # 출력 크기 저장 또는 반환
        self.output_shape = (output_height, output_width, output_channels)

    # 입력에 대한 pooling 연산 수행
    def call(self, input_data):
        # 파라미터들을 전달하기
        
        x, self.node_list = pooling.pooling2d(input_data, self.pool_size[0], self.pool_size[1], self.strides, self.pool_mode, self.node_list)
        
        self.output_shape = x.shape

        return x