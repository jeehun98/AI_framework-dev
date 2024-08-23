# 모델에서 값이 입력되는 niput_layer
# 어떠한 값이 입력으로 사용될지 미리 정의
# 값이 입력되는 형태에 대해서만 저장
# shape 는 튜플 형태로 저장되어야 함

import warnings

from dev.layers.layer import Layer

class InputLayer(Layer):
    def __init__(
        self,
        shape=None,
        name=None,
        **kwargs,
    ):
        # TODO: support for ragged.
        super().__init__(name=name)
        
        if "input_shape" in kwargs:
            self.shape = kwargs.pop("input_shape")

        
# 부모 클래스 Layer 가 상속 받는 부모 클래스에서 output 이 정의되어 있음...
# 레이어 선언, 및 모델 구성의 다양한 방식으로의 구현 허용을 위해 추가되는 것 같은데...

def Input(
    shape=None,
    name=None,
):
    layer = InputLayer(
        shape=shape,
        name=name
    )
    return layer.output