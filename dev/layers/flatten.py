import os
os.add_dll_directory("C:\\msys64\\mingw64\\bin")

from dev.layers.layer import Layer
from dev.backend.flatten import flatten

import numpy as np

from functools import reduce
from operator import mul

class Flatten(Layer):
    def __init__(self, input_shape=None, **kwargs):
        # 인스턴스 생성 시 어떤 값이 추가되어야 할 지에 대한 고민
        super().__init__(input_shape, **kwargs)
        self.input_shape = input_shape
        self.output_shape = input_shape
        # 가중치 갱신이 없는 layer, flatten
        self.trainable = False
        self.node_list = []
        self.layer_name = "flatten"

    # 객체 정보를 저장하는 get_config
    # 향후 from_config 를 통해 해당 객체를 복원할 수 있다. 
    def get_config(self):
        base_config = super().get_config()
        config = {
            "class_name": self.__class__.__name__,
            "input_shape": self.input_shape
        }
        return {**base_config, **config}

    # 저장된 config 파일로 객체 생성
    def from_config(cls, config):
        return(cls **config)

    # 계산시 (batch_size, flattened_dim) 형태,
    # 전체 데이터에 대해 생각해야해, 데이터 하나에 대한 flatten 연산이 아님

    def call(self, inputs):
        """
        n x p 차원 입력 데이터를 펼침

        Parameters:
        inputs: np.ndarray 
            (1, p) 또는 (1, p_1, p_2) 형태의 데이터

        Returns:
        np.ndarray:
            (1, p*) 형태로 펼친 데이터. 단일 배치의 행 벡터 형태로 출력.
        """
        # 입력 데이터를 1차원으로 변환
        flattened_data, self.node_list = flatten.flatten(inputs)

        # 출력 차원 설정
        self.output_shape = flattened_data.shape

        return flattened_data

    def compute_output_shape(self, input_shape):
        # 입력 shape 를 기반으로 출력 shape 를 계산, 모델의 구조 정의
        return (input_shape[0], np.prod(input_shape[1:]))

    def multiply_tuple_elements(self, t):
        return reduce(mul, t, 1)
    
    # build 는 가중치 초기화의 정의, 
    # flatten 은 별도의 가중치를 필요로 하지 않음
    def build(self, input_shape):
        
        result = self.multiply_tuple_elements(input_shape)

        self.output_shape = (1, result)

        super().build()


