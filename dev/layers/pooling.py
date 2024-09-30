import os
os.add_dll_directory("C:\\msys64\\mingw64\\bin")

from dev.layers.layer import Layer
from dev.backend.pooling import pooling

import numpy as np

class Polling(Layer):
    def __init__(
        self,
        pool_size=(2, 2),
        strides = None,
        padding="valid",
        **kwargs
    ):
        super().__init__(
            pool_size,
            strides,
            pool_dimensions=2,
            pool_mode="max",
            padding=padding,
            **kwargs,
        )
        # 가중치 생성이 필요 없음
        self.built = True,

    # 입력에 대한 pooling 연산 수행
    def call(self, input_data):
        if self.pool_mode == "max":
            x, self.node_list = pooling.max_pooling()