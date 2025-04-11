from dev.layers.layer import Layer
from dev import activations
import numpy as np

class Activation(Layer):
    """
    Activation 레이어 클래스:
    입력된 활성화 함수를 사용하여 입력 데이터에 비선형성을 추가합니다.
    """

    def __init__(self, activation, **kwargs):
        """
        Parameters:
            activation (str or callable): 사용할 활성화 함수의 이름 또는 함수 객체.
            **kwargs: 추가적인 파라미터는 부모 클래스(Layer)로 전달됩니다.
        """
        super().__init__(**kwargs)

        self.activation = activations.get(activation)
        self.node_list = []
        self.trainable = False
        self.layer_name = "activation"

    def call(self, inputs):
        """
        입력 데이터를 활성화 함수에 적용합니다.

        Parameters:
            inputs (np.ndarray): 이전 레이어의 출력 데이터.

        Returns:
            np.ndarray: 활성화 함수 적용 후 출력 데이터.
        """
        output = self.activation(inputs.astype(np.float32))  # ✅ CUDA용 float32 강제
        self.node_list = []  # CUDA 백엔드는 계산 그래프 없음
        
         # ✅ 입력 shape를 그대로 유지하도록 강제
        return output.reshape(inputs.shape)

    def compute_output_shape(self, input_shape):
        return input_shape

    def build(self, input_shape):
        self.input_shape = input_shape
        self.output_shape = input_shape
        super().build(input_shape)
