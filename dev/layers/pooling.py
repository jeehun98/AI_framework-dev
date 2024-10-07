import os
os.add_dll_directory("C:\\msys64\\mingw64\\bin")

from dev.layers.layer import Layer
from dev.backend.pooling import pooling

class Pooling(Layer):
    def __init__(
        self,
        pool_size=(2, 2),
        strides = 1,
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
        

    def build(self, *args, **kwargs):
        super().build()

    # 입력에 대한 pooling 연산 수행
    def call(self, input_data):
        # 파라미터들을 전달하기
        
        x, self.node_list = pooling.pooling2d(input_data, self.pool_size[0], self.pool_size[1], self.strides, self.pool_mode, self.node_list)
        
        self.output_shape = x.shape

        return x