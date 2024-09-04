import typing
import json

from dev.layers.layer import Layer
from dev.layers.core.input_layer import InputLayer
from dev import optimizers
from dev import losses
from dev import metrics
#from dev.layers.core.dense import Dense
#from dev.models.model import Model

# Sequential 을 최상위 모델이라고 가정하고 해보자
class Sequential():
    def __new__(cls, *args, **kwargs):
        # 부모 클래스의 __new__ 메서드 호출, 인스턴스 생성
        # typing.cast 를 통해 반환된 인스턴스 타입을 자식 클래스로 명시적 지정
        # 기본 __new__ 메서드 구조랑 똑같음, typing.cast 가 추가됨
        return typing.cast(cls, super().__new__(cls))

    
    # 레이어 객체 상태 초기화, 어떤 객체가 초기화 되어야 할 지에 대한 고민
    def __init__(self, layers=None, trainable=True, name=None):
        #super().__init__(trainable=trainable, name=name)
        self.built = False
        # layer list
        self._layers = []

    # 모델의 layer 정보 전달
    def get_config(self):
        seuqential_config = {
            'name':'seqential',
        }
        layer_configs = []
        for layer in self._layers:
            layer_configs.append(layer.get_config())
        return {**seuqential_config, "layers" : layer_configs}


    # 레이어 추가
    # 파라미터, layer 는 여기서 이미 객체가 생성되면서 여기 메서드 파라미터로 들어가는데...
    def add(self, layer):
        # 입력 형태가 layer 인스턴스가 아닐 경우 
        if not isinstance(layer, Layer):
            raise ValueError(
                "Only instances of Layer can be "
                f"added to a Sequential model. Received: {layer} "
                f"(of type {type(layer)})"
            )
        # 레이어가 존재
        if self._layers:
            # 마지막 layer 선택
            previous_layer = self._layers[-1]   
            # 이전 레이어의 units 개수가 이번 레이어의 input_shape
            if hasattr(previous_layer, 'units') and (previous_layer.units != None):
                input_shape = (previous_layer.units,)
            # units 이 없을 경우 input_shape 가 그대로 복사
            # 이전 layer 가 flatten 일 경우
            elif hasattr(previous_layer, 'input_shape') and (previous_layer.input_shape != None):
                input_shape = previous_layer.input_shape
            
            # input_shape 값이 들어오거나, input_shape 값이 이미 있을 경우 
            if hasattr(layer, "input_shape") or (layer.input_shape == None):
                # build 를 통해 input_shape 지정과 함께, 가중치 초기화
                # 각 객체 클래스 인스턴스에 맞게 build 가 실행된다.
                # dense 의 경우 가중치 초기화
                layer.build(input_shape)

        # 입력된 layer 의 추가, 
        self._layers.append(layer)

    
    # layer build 는 가중치 초기화를 진행했음
    # model build 를 통해 build_config 정보를 구성, input_shape 정보
    # model.compile 을 통해 실행된다.
    def build(self):
        self.input_shape = self._layers[0].input_shape
    
    def get_build_config(self):
        return {
            "input_shape" : self.input_shape
        }


    # compile 시 저장되는 정보
    def compile(self, optimizer=None, loss=None, p_metrics=None):
        self.optimizer = optimizers.get(optimizer)
        self.loss = losses.get(loss)
        self.metric = metrics.get(p_metrics)

        #빌드 수행
        self.build()


    # 모델의 compile 정보 전달 
    def get_compile_config(self):
        optimizer_config = self.optimizer.get_config()
        loss_config = self.loss.get_config()
        metrics_config = self.metric.get_config()
        
        return {
                "optimizer": optimizer_config, 
                "loss": loss_config, 
                "metrics": metrics_config,
            }

    # 각 레이어 방문, 가중치 정보 호출
    def get_weight(self):
        weights = []
        for layer in self._layers:
            if hasattr(layer, 'weights'):
                weights.append(layer.weights)

        return weights


    def serialize_model(self):
        # Step 1: Get the compile config, model config, and build config
        compile_config = self.get_compile_config()
        model_config = self.get_config()
        build_config = self.get_build_config()

        # Step 2: Get the weights
        weights = []
        for layer in self._layers:
            layer_weights = layer.get_weights()  # Get weights of the layer
            serialized_weights = [w.tolist() for w in layer_weights]  # Convert numpy arrays to lists
            weights.append(serialized_weights)

        # Step 3: Create a dictionary to store all the information
        model_data = {
            "compile_config": compile_config,
            "model_config": model_config,
            "build_config": build_config,
            "weights": weights,
        }

        # Step 4: Serialize the dictionary to a JSON string
        serialized_model = json.dumps(model_data)

        return serialized_model
    
    
    # fit 을 구현해보자잇~
    def fit(self, x=None, y=None, epochs = 1):
        """
        모델의 연산 부분
        
        Parameters:
        x (n, p): p 개의 특성을 가진, n 개의 데이터
          (n, p_1, p_2) : p_1,2, 2차원의 특성을 가진 n 개의 데이터
        y (n, 1): n 개의 타겟값

        """
        # 연산 결과를 저장
        result = []

        # 초기 입력값
        output = x
        # 전체 데이터를 처리하도록
        for layer in self._layers:
            output = layer.call(output)
            result.append(output)
            
        return result
    

    def call(self, inputs):
        for layer in self.layers:
            outputs = layer(inputs)
            inputs = outputs

        # fit sequential 클래스 fit 메서드 실행
        # call 메서드 실행
        # y_pred 값, outputs
        return outputs   