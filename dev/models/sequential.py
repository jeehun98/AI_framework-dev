import typing

from dev.models.model import Model
from dev.layers.layer import Layer
from dev.layers.core.input_layer import InputLayer

class Sequential(Model):
    def __new__(cls, *args, **kwargs):
        # 부모 클래스의 __new__ 메서드 호출, 인스턴스 생성 
        # typing.cast 를 통해 반환된 인스턴스 타입을 자식 클래스로 명시적 지정
        return typing.cast(cls, super().__new__(cls))
    
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


    # build 와 함께 가중치 초기화...
    def build(self, input_shape=None):
        input_shape = input_shape

        if not self._layers:
            raise ValueError(
                f"Sequential model {self.name} cannot be built because it has "
                "no layers. Call `model.add(layer)`."
            )
        
        inputs = self._layers[0].output
        x = inputs

        for layer in self._layers[1:]:
            try:
                # call 메서드 실행, 레이어 출력이 x 에 저장
                x = layer(x)
            except NotImplementedError:
                return         
        
        outputs = x
        self.built = True
    

    def call(self, inputs):
        for layer in self.layers:
            outputs = layer(inputs)
            inputs = outputs

        return outputs