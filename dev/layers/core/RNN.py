import os
os.add_dll_directory("C:\\msys64\\mingw64\\bin")

import numpy as np
from dev.layers.layer import Layer
from dev import activations
from dev.backend.recurrent import recurrent


class RNN(Layer):
    def __init__(
        self, 
        units,
        activation="tanh",
        recurrent_activation="sigmoid",
        use_bias=True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.units = units
        self.activation = activation
        self.recurrent_activation = recurrent_activation
        self.use_bias = use_bias
        self.state = None

    def build(self, input_shape):
        """
        rnn 의 가중치
        W (커널) : 입력 데이터에서 은닉 상태로 전달되는 가중치 ( 입력 데이터의 차원, 은닉 유닛의 수 )
        U (순환 커널) : 이전 은닉 상태에서 현재 은닉 상태로 전달되는 가중치 ( 은닉 유닛의 수, 은닉 유닛의 수 )
        b (바이어스) : 각 은닉 상태의 바이어스 ( 은닉 유닛의 수 )

        """
        input_dim = input_shape[-1]
        
        # 가중치 초기화

        # 입력 데이터에 대한 가중치, (벡터화된 토큰의 입력 차원 수, 은닉 차원 수)
        # 연산 이후 차원의 변화 생각을 해야해
        self.weight = np.random.randn(input_dim, self.units)

        # 순환 가중치 
        self.recurrent_weight = np.random.randn(self.units, self.units)
        self.bias = np.zeros((self.units,)) if self.use_bias else None

        # 상태 초기화
        self.state = np.zeros((1, self.units))

    def call(self, inputs):
        timesteps = inputs.shape[1]

        # C++로 구현한 RNN 레이어 호출
        output_sequence, node_list = recurrent.rnn_layer(
            inputs, 
            self.weight, 
            self.recurrent_weight, 
            self.bias, 
            self.activation
        )

        # 결과를 numpy array로 변환하여 반환
        return output_sequence

    def get_config(self):
        config = {
            "units": self.units,
            "activation": self.activation,
            "recurrent_activation": self.recurrent_activation,
            "use_bias": self.use_bias,
        }
        base_config = super().get_config()
        return {**base_config, **config}