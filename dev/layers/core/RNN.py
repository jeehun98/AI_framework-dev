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
        input_dim = input_shape[-1]
        
        # 가중치 초기화
        self.kernel = np.random.randn(input_dim, self.units)
        self.recurrent_kernel = np.random.randn(self.units, self.units)
        self.bias = np.zeros((self.units,)) if self.use_bias else None

        # 상태 초기화
        self.state = np.zeros((1, self.units))

    def call(self, inputs):
        timesteps = inputs.shape[1]

        # C++로 구현한 RNN 레이어 호출
        output_sequence, node_list = recurrent.rnn_layer(
            inputs, 
            self.kernel, 
            self.recurrent_kernel, 
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