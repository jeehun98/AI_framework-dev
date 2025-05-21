from dev.layers.layer import Layer
from dev.decorators import check_layer_type


class Model():
    
    def __new__(cls, *args, **kwargs):
        pass

    def __init__(self, *args, **kwargs):
        # 모델 초기화 시 레이어를 담을 리스트를 준비합니다.
        self._layers = []
    

    def add(self, layer):
        if len(self.layers) == 0 and layer.input_shape is None:
            raise ValueError("Input shape must be defined for the first layer.")
        if len(self.layers) > 0 and layer.input_shape is None:
            layer.input_shape = self.layers[-1].input_shape
        
        layer.build()
        self.layers.append(layer)

    def save(self):
        # 모델 저장 (구성 정보와 가중치)
        model_config = [layer.get_config() for layer in self.layers]
        return model_config

    @classmethod
    def load(cls, model_config):
        # 저장된 구성 정보로부터 모델 복원
        model = cls()
        for config in model_config:
            layer = Layer.from_config(config)
            model.add(layer)
        return model
    
    @property
    def layers(self):
        # 모델에 추가된 레이어를 리스트로 반환합니다.
        return list(self._layers)

    @layers.setter
    def layers(self, _):
        # layers 속성에 값을 설정하려고 하면 오류를 발생시킵니다.
        raise AttributeError("`Model.layers` attribute is read-only and cannot be set directly.")
    

