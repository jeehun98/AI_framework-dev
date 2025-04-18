import numpy as np

from dev.layers.layer import Layer
from dev import activations
from dev.graph_engine.activations_graph import build_relu_node, build_sigmoid_node, build_tanh_node

class Activation(Layer):
    def __init__(self, activation, **kwargs):
        super().__init__(**kwargs)
        self.activation_name = activation
        self.activation = activations.get(activation)
        self.node_list = []
        self.trainable = False
        self.layer_name = "activation"

        # 계산 그래프 빌더 함수 매핑
        self.graph_builders = {
            "relu": build_relu_node,
            "sigmoid": build_sigmoid_node,
            "tanh": build_tanh_node,
        }

    def call(self, inputs):
        inputs = inputs.astype(np.float32)
        output = self.activation(inputs)

        # 계산 그래프 구성
        builder = self.graph_builders.get(self.activation_name)
        if builder is None:
            raise NotImplementedError(f"'{self.activation_name}' 계산 그래프 생성 함수가 정의되지 않았습니다.")

        # 입력 노드 수에 맞게 계산 그래프 생성
        flat_inputs = inputs.reshape(-1)
        self.node_list = [builder() for _ in range(flat_inputs.shape[0])]

        # 노드 연결: 입력 노드로부터 부모 연결 필요 시 외부에서 처리됨
        # 현재는 연결된 입력 노드가 없는 상태로 노드만 생성됨

        self.output_shape = output.shape

        print(self.node_list, "확인")
        
        return output

    def compute_output_shape(self, input_shape):
        return input_shape

    def build(self, input_shape):
        self.input_shape = input_shape
        self.output_shape = input_shape
        super().build(input_shape)