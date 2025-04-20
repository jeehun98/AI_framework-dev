import numpy as np

from dev.layers.layer import Layer
from dev import activations
from dev.graph_engine.activations_graph import build_relu_node, build_sigmoid_node, build_tanh_node

class Activation(Layer):
    def __init__(self, activation, **kwargs):
        super().__init__(**kwargs)
        self.activation_name = activation
        self.activation = activations.get(activation)
        self.root_node_list = []
        self.leaf_node_list = []  # ✅ 리프 노드 따로 저장
        self.trainable = False
        self.layer_name = "activation"

        self.graph_builders = {
            "relu": build_relu_node,
            "sigmoid": build_sigmoid_node,
            "tanh": build_tanh_node,
        }

    def call(self, inputs):
        inputs = inputs.astype(np.float32)
        output = self.activation(inputs)

        builder = self.graph_builders.get(self.activation_name)
        if builder is None:
            raise NotImplementedError(...)

        flat_inputs = inputs.reshape(-1)
        self.root_node_list = []
        self.leaf_node_list = []

        for idx in range(flat_inputs.shape[0]):
            root, leaves = builder()
            self.root_node_list.append(root)
            self.leaf_node_list.extend(leaves)  # ✅ 꼭 extend 해야 함

        self.output_shape = output.shape
        return output


    def compute_output_shape(self, input_shape):
        return input_shape

    def build(self, input_shape):
        self.input_shape = input_shape
        self.output_shape = input_shape
        super().build(input_shape)
