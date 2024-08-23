from layers.layer import Layer
from decorators import check_layer_type


class Model():
    
    def __new__(cls, *args, **kwargs):
        pass

    def __init__(self, *args, **kwargs):
        # 모델 초기화 시 레이어를 담을 리스트를 준비합니다.
        self._layers = []
        pass

    @check_layer_type
    def add_layer(self, layer):
        # 모델에 레이어를 추가합니다.
        if not isinstance(layer, Layer):
            raise TypeError("Only instances of Layer can be added")
        self._layers.append(layer)
    
    @property
    def layers(self):
        # 모델에 추가된 레이어를 리스트로 반환합니다.
        return list(self._layers)

    @layers.setter
    def layers(self, _):
        # layers 속성에 값을 설정하려고 하면 오류를 발생시킵니다.
        raise AttributeError("`Model.layers` attribute is read-only and cannot be set directly.")
    

