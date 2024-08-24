from dev.layers.layer import Layer
import numpy as np

class Flatten(Layer):
    def __init__(self, name=None):
        super().__init__(name)

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs):
        # 입력 텐서를 1차원으로 평평하게 만듭니다.
        return np.reshape(inputs, (inputs.shape[0], -1))

    def compute_output_shape(self, input_shape):
        # 입력의 배치 크기는 유지하고 나머지를 모두 평평하게 만듭니다.
        return (input_shape[0], np.prod(input_shape[1:]))
