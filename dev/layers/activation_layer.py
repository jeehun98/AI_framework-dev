from dev.layers.layer import Layer
from dev import activations
from dev.graph_engine.activations_graph import build_relu_node, build_sigmoid_node, build_tanh_node
import numpy as np
class Activation(Layer):
    def __init__(self, activation, use_graph=False, **kwargs):
        super().__init__(**kwargs)
        self.activation_name = activation
        self.activation = activations.get(activation)
        self.use_graph = use_graph
        self.node_list = []
        self.trainable = False
        self.layer_name = "activation"

        # ✅ 계산 그래프 빌더 함수 매핑
        self.graph_builders = {
            "relu": build_relu_node,
            "sigmoid": build_sigmoid_node,
            "tanh": build_tanh_node,
        }

    def call(self, inputs, input_node_list=None):
        inputs = inputs.astype(np.float32)

        if self.use_graph:
            try:
                builder = self.graph_builders[self.activation_name]
            except KeyError:
                raise NotImplementedError(f"'{self.activation_name}' 계산 그래프 생성 함수가 정의되지 않았습니다.")

            # ✅ 1. 그래프 구조만 생성 (입력 없는 노드들)
            raw_nodes = [builder() for _ in range(inputs.size)]  # inputs.shape = (N, ...)
            
            # ✅ 2. 외부에서 받은 input_node_list 와 연결
            if input_node_list is None:
                raise ValueError("계산 그래프 사용 시 input_node_list는 반드시 필요합니다.")
            if len(input_node_list) != len(raw_nodes):
                raise ValueError("입력 노드 수와 활성화 함수 그래프 수가 일치해야 합니다.")

            for output_node, input_node in zip(raw_nodes, input_node_list):
                output_node.add_parent(input_node)

            # ✅ 3. 연결 후 루트 노드들 저장
            self.node_list = raw_nodes

            output = np.array([node.output for node in self.node_list], dtype=np.float32)
            return output.reshape(inputs.shape)

        else:
            self.node_list = []
            output = self.activation(inputs)
            return output.reshape(inputs.shape)

    def compute_output_shape(self, input_shape):
        return input_shape

    def build(self, input_shape):
        self.input_shape = input_shape
        self.output_shape = input_shape
        super().build(input_shape)
