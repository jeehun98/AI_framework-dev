import typing
import json

from dev.layers.layer import Layer
from dev import optimizers
from dev import losses
from dev import metrics
from dev.node.node import Node

import numpy as np
#from dev.layers.core.dense import Dense
#from dev.models.model import Model

# Sequential 을 최상위 모델이라고 가정하고 해보자
class Sequential(Node):
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
        self.node_list = []
        self.loss_node_list = []
        

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
        # 기존 레이어가 존재
        if self._layers:
            # input_shape 를 지정하기
            previous_layer = self._layers[-1]   
            input_shape = previous_layer.output_shape

            # 이전 layer 의 출력 차원 가져오기
            if hasattr(previous_layer, 'output_shape') and (previous_layer.output_shape != None):
                input_shape = previous_layer.output_shape
                layer.build(input_shape)
            
            # layer 에 input_shape 값이 이미 있을 경우 
            elif hasattr(layer, "input_shape") and (layer.input_shape != None):
                # build 를 통해 input_shape 지정과 함께, 가중치 초기화
                # 각 객체 클래스 인스턴스에 맞게 build 가 실행된다.
                # dense 의 경우 가중치 초기화
                layer.build(input_shape)

        # 처음 입력되는 layer 에는 반드시 input_shape 가 존재해야 함
        elif hasattr(layer, 'input_shape'):
            # 해당 레이어의 가중치 생성
            input_shape = layer.input_shape
            layer.build(input_shape)

        self._layers.append(layer)

        print(layer.output_shape, "출력 형태 확인")

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
    def compile(self, optimizer=None, loss=None, p_metrics=None, learning_rate=0.001):
        # 옵티마이저 객체가 생성되는...
        self.optimizer = optimizers.get(optimizer, learning_rate = learning_rate)
        self.loss = losses.get(loss)
        self.metric = metrics.get(p_metrics)

        self.build()
        self.get_weight()


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

        self.weigts = weights


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
    
    
    # fit 을 구현해보자잇~ forward 연산이라고 보면 될 듯
    # layer 를 따라 call 연산을 수행하면서 layer 의 계산 그래프 연결하기
    def fit(self, x=None, y=None, epochs = 1, batch_size = -1):
        """
        parameters
        x : 입력 데이터
        y : 타겟 데이터
        epochs : 학습 반복 횟수
        batch_size : 배치 크기, -1일 경우 전체 데이터를 하나의 배치로 사용
        """

        # 배치 사이즈가 주어지지 않았을 때, 전체 데이터를 하나의 배치로 사용
        if batch_size == -1 or batch_size > x.shape[0]:
            batch_size = x.shape[0]

        # 배치 데이터의 개수
        batch_counts = int(np.ceil(x.shape[0] / batch_size))


        # 학습 반복 횟수
        for epoch in range(epochs):
            # 매 에포크마다 데이터를 섞음
            indices = np.random.permutation(x.shape[0])
            x = x[indices]
            y = y[indices]

            # 다음 배치로 넘어갈 때 가중치 갱신량은 초기화, 바뀐 가중치 값은 그대로 들고간다.
            # 배치의 개수만큼 반복
            for batch_idx in range(batch_counts):
                
                # 배치 데이터 추출
                start = batch_idx * batch_size
                end = min(start + batch_size, x.shape[0])
                batch_x = x[start:end]
                batch_y = y[start:end]

                # 배치내 데이터 개수
                batch_datas = batch_x.shape[0]
                
                # 각 배치 데이터에 대한 반복
                for batch_data_idx in range(batch_datas):
                    input_data = batch_x[batch_data_idx]
                    target = batch_y[batch_data_idx]

                    # 입력 데이터를 0번째 layer 의 출력값이라고 생각, output 을 계속 업데이트
                    output = input_data

                    # 가장 첫 번째의 학습에선 계산 그래프를 생성하고 이를 연결하는 과정이 필요
                    if batch_data_idx == 0 and epoch == 0:

                        # 이전에 레이어가 존재할 경우 계산 그래프를 연결해야함
                        for idx, layer in enumerate(self._layers):

                            # 이전 layer
                            previous_layer = self._layers[idx - 1]    
                            
                            # 출력값 갱신, layer 의 call 연산이 호출된다.
                            output = layer.call(output)
                            
                            # 첫번째 레이어의 경우
                            if idx == 0 and layer.trainable:
                                self.node_list = layer.node_list
                                continue
                            
                            # 해당 레이어가 학습 가능한 경우 계산 그래프 연결하기
                            if layer.trainable:
                                # 계산 그래프 연결
                                self.node_list = self.link_node(layer, previous_layer)

                        # loss_node_list 생성,
                        output = np.array(output).reshape(1, -1)
                        target = np.array(target).reshape(1, -1)
                        
                        self.compute_loss_and_metrics(output, target)

                        # loss_node_list 의 연결
                        self.node_list = self.link_loss_node(self.loss_node_list, self.node_list)

                        # 계산 그래프 리스트들의 역전파 연산 수행

                        """
                        NODE 클래스, 혹은 다른 클래스에서 수행하도록 변경해야겠다.
                        """
                        for root_node in  self.node_list:
                            self.backpropagate(root_node)

                    else:
                        # 계산 그래프가 생성되고 난 후...
                        """
                        조건문의 변경을 해야겠다.
                        """ 

                        # 각 layer 의 call 연산, 계산 그래프가 있을 경우
                        # = self.node_list 가 존재할 경우임
                        for layer in self._layers:
                            output = layer.call(output)
                        # loss, metrics 연산의 수행

                        output = np.array(output).reshape(1, -1)

                        self.compute_loss_and_metrics(output, batch_y[batch_data_idx].reshape(1, -1))

                        for root_node in  self.node_list:
                            self.backpropagate(root_node)

                # 배치 반복 끝, 가중치 갱신
                for root_node in self.node_list:
                    self.weight_update(root_node, batch_datas, self.optimizer)

                """
                이후 계산 그래프의 가중치는 동일하게, 가중치 갱신량은 초기화해야함
                """

        # 에포크 끝난 후 평균 손실 출력
        loss_sum = 0
        for data_idx in range(x.shape[0]):
            input_data = x[data_idx]
            target = y[data_idx]
            predict = self.predict(input_data)
            predict = np.array(predict).reshape(1, -1)
            data_loss = self.compute_loss_and_metrics(predict, target.reshape(1, -1))
            loss_sum += data_loss

        print(f"Average Loss: {loss_sum / x.shape[0]}")

    # 예측 수행
    def predict(self, data):
        output = data
        for layer in self._layers:
            output = layer.call(output)
        return output

    # 비용 함수값 계산
    def compute_loss_and_metrics(self, y_pred, y_true):
        # 매 계산 마다 self.loss_node_list 가 갱신,
        self.loss_value, self.loss_node_list = self.loss(y_pred, y_true, self.loss_node_list)
        self.metric_value = self.metric(y_pred, y_true)
        # print(y_pred, y_true, self.loss_value)
        return self.loss_value
        
        
    def call(self, inputs):
        for layer in self.layers:
            outputs = layer(inputs)
            inputs = outputs

        return outputs