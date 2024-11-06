# 모델에서 값이 입력되는 첫 번째 레이어
from dev.layers.layer import Layer

# 첫 번째 레이어, InputLayer class
class InputLayer(Layer):
    def __init__(
        self,
        shape=None,
        name=None,
        **kwargs,
    ):
        # TODO: support for ragged.
        super().__init__(name=name)
        
        # input_shape 속성의 지정, 저장
        if "input_shape" in kwargs:
            self.shape = kwargs.pop("input_shape")

    
    # 첫 번째 layer 에 대해선 연산이 수행되지 않음
    def call(self):
        return 


    def get_config(self):
        config = super().get_config()
        config.update({
            'input_shape':self.input_shape,

        })
        return config


# 부모 클래스 Layer 가 상속 받는 부모 클래스에서 output 이 정의되어 있음...
# 레이어 선언, 및 모델 구성의 다양한 방식으로의 구현 허용을 위해 추가되는 것 같은데...
"""
def Input(
    shape=None,
    name=None,
):
    layer = InputLayer(
        shape=shape,
        name=name
    )
    return layer.output
    
"""