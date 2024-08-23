import typing

from models.model import Model
from layers.layer import Layer

class Seuquential(Model):
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
    def add(self, layer, rebuild=True):
        # 저장되어 있는 레이어가 없으면, 
        # 입력된 layer 의 추가, 
        if not self._layers:
            if getattr(layer, "_input_shape_arg", None) is not None:
                # 인스턴스가 전달됨
                self.add(InputLayer(shape=layer._input_shape_arg))

        if not isinstance(layer, Layer):
            raise ValueError(
                "Only instances of Layer can be "
                f"added to a Sequential model. Received: {layer} "
                f"(of type {type(layer)})"
            )