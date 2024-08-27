from dev.layers.layer import Layer
import numpy as np

class Flatten(Layer):
    def __init__(self, **kwargs):
        # 인스턴스 생성 시 어떤 값이 추가되어야 할 지에 대한 고민
        super().__init__(**kwargs)

    # (batch_size, flattened_dim) 형태,
    # 전체 데이터에 대해 생각해야해, 데이터 하나에 대한 flatten 연산이 아님

    def call(self, inputs):
        # 데이터를 실제로 변환하는데 초점
        # 배치 크기가 어떻게 유지되는지 알 수 있는 부분
        return np.reshape(inputs, (inputs.shape[0], -1))

    def compute_output_shape(self, input_shape):
        # 입력 shape 를 기반으로 출력 shape 를 계산, 모델의 구조 정의
        return (input_shape[0], np.prod(input_shape[1:]))
