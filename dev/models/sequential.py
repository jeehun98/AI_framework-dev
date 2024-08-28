import typing

from dev.models.model import Model
from dev.layers.layer import Layer
from dev.layers.core.input_layer import InputLayer

class Sequential(Model):
    def __new__(cls, *args, **kwargs):
        # 부모 클래스의 __new__ 메서드 호출, 인스턴스 생성 
        # typing.cast 를 통해 반환된 인스턴스 타입을 자식 클래스로 명시적 지정
        return typing.cast(cls, super().__new__(cls))
    
    def __call__(self, inputs):
        # 모델에 add 되어 있는 각 layer 의 call 메서드 호출을 통한 연산
        for layer in self.layers:
            outputs = layer(inputs)
            inputs = outputs

        return outputs

    
    # 레이어 객체 상태 초기화, 어떤 객체가 초기화 되어야 할 지에 대한 고민
    def __init__(self, layers=None, trainable=True, name=None):
        super().__init__(trainable=trainable, name=name)
        # Functional API 스타일 변환 여부 확인
        self._functional = None
        self._layers = []
        # 레이어 리스트
        # 여기서 추가되는 레이어에 대한 rebuild = False 의 지정
        # 모두 추가 후 모델의 빌드
        if layers:
            for layer in layers:
                self.add(layer, rebuild=False)
            self._may_rebuild()

    # 레이어 추가
    def add(self, layer):
        # 입력 형태가 layer 인스턴스가 아닐 경우 
        if not isinstance(layer, Layer):
            raise ValueError(
                "Only instances of Layer can be "
                f"added to a Sequential model. Received: {layer} "
                f"(of type {type(layer)})"
            )
        
        # 입력된 layer 의 추가, 
        self._layers.append(layer)