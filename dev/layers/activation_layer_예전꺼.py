# dev/layers/activation_layer.py

import numpy as np

from dev.layers.layer import Layer
from dev.backend.backend_ops.activations import activations as cuda_activations
from dev.activations import activations_mapping as graph_activations

class Activation(Layer):
    
    def __init__(self, activation, **kwargs):
        super().__init__(**kwargs)
        self.activation_name = activation

        self.root_node_list = []
        self.leaf_node_list = []
        self.trainable = False
        self.layer_name = "activation"

        # ✅ CUDA 함수 매핑
        self.cuda_functions = {
            "relu": cuda_activations.relu,
            "sigmoid": cuda_activations.sigmoid,
            "tanh": cuda_activations.tanh,
        }

        # ✅ 계산 그래프 빌더 매핑 (이게 누락됨!)
        self.graph_builders = {
            "relu": graph_activations.relu_graph,
            "sigmoid": graph_activations.sigmoid_graph,
            "tanh": graph_activations.tanh_graph,
        }


    def call(self, inputs):
        # 1️⃣ CUDA 연산 수행
        try:
            activation_func = self.cuda_functions[self.activation_name]
        except KeyError:
            raise NotImplementedError(f"{self.activation_name} CUDA 미지원")

        inputs = inputs.astype(np.float32)
        output = activation_func(inputs)

        # 2️⃣ 계산 그래프 빌더 호출
        builder = self.graph_builders.get(self.activation_name)
        if builder is None:
            raise NotImplementedError(f"[ERROR] '{self.activation_name}' 그래프 미지원")

        flat_inputs = inputs.reshape(-1)

        self.root_node_list = []
        self.leaf_node_list = []

        for idx in range(flat_inputs.shape[0]):
            single_output = output[0, idx]
            root, leaves = builder(single_output)
            self.root_node_list.append(root)
            self.leaf_node_list.extend(leaves)  # ✅ 여러 입력일 경우 확장

        self.output_shape = output.shape
        
        return output

    def compute_output_shape(self, input_shape):
        return input_shape

    def build(self, input_shape):
        self.input_shape = input_shape
        self.output_shape = input_shape
        super().build(input_shape)
