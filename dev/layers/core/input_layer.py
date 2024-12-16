from dev.layers.layer import Layer
from dev.layers.core.output_layer import LayerOutput

class InputLayer(Layer):
    def __init__(self, shape=None, name=None, **kwargs):
        super().__init__(name=name)
        
        # input_shape 속성 지정
        if "input_shape" in kwargs:
            self.shape = kwargs.pop("input_shape")
        elif shape is not None:
            self.shape = shape
        else:
            raise ValueError("Input shape must be defined.")

    # 첫 번째 레이어는 연산을 수행하지 않고 입력만 관리
    def call(self, inputs):
        return LayerOutput(self.shape, inputs)

    def get_config(self):
        config = super().get_config()
        config.update({
            'input_shape': self.shape,
        })
        return config

    @property
    def output(self):
        # 호출 시 새로운 InputLayerOutput 객체 반환
        return LayerOutput(self.shape)

# 해당 클래스 메서드 호출, 생성
def Input(shape=None, name=None):
    layer = InputLayer(shape=shape, name=name)
    return layer.output  # InputLayerOutput 객체를 반환