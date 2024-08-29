import typing

from dev.layers.layer import Layer
from dev.layers.core.input_layer import InputLayer
#from dev.layers.core.dense import Dense
#from dev.models.model import Model

# Sequential 을 최상위 모델이라고 가정하고 해보자
class Sequential(Layer):
    def __new__(cls, *args, **kwargs):
        # 부모 클래스의 __new__ 메서드 호출, 인스턴스 생성
        # typing.cast 를 통해 반환된 인스턴스 타입을 자식 클래스로 명시적 지정
        # 기본 __new__ 메서드 구조랑 똑같음, typing.cast 가 추가됨
        return typing.cast(cls, super().__new__(cls))

    
    # 레이어 객체 상태 초기화, 어떤 객체가 초기화 되어야 할 지에 대한 고민
    def __init__(self, layers=None, trainable=True, name=None):
        #super().__init__(trainable=trainable, name=name)

        self.built = False

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

    def get_config(self):
        layer_configs = []
        for layer in self._layers:
            
            layer_configs.append(layer.get_config())

        return layer_configs

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


    # build 와 함께 가중치 초기화

    def build(self, input_shape=None):
        #input_shape = input_shape

        if not self._layers:
            raise ValueError(
                f"Sequential model {self.name} cannot be built because it has "
                "no layers. Call `model.add(layer)`."
            )
        
        if isinstance(self._layers[0], InputLayer):
            input_shape = self._layers[0].input_shape

        # 가중치 초기화만 시행할거야

        for layer in self._layers[1:]:
            try:
                # build 메서드 실행, input_shape 와 해당 layer의 output_shape 크기
                # 를 통해 임의의 가중치가 생성된다.
                # 생성된 가중치는 해당 레이어 인스턴스에 저장,
                layer.build(input_shape)
                input_shape = layer.output_shape

            except NotImplementedError:
                return         
        
        self.built = True
    

    def call(self, inputs):
        for layer in self.layers:
            outputs = layer(inputs)
            inputs = outputs

        # fit sequential 클래스 fit 메서드 실행
        # call 메서드 실행
        # y_pred 값, outputs
        return outputs   