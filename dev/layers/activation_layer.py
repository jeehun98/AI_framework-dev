from dev.layers.layer import Layer
from dev.backend.backend_ops.activations import activations as cuda_activations
from dev.activations import activations_mapping as graph_activations

class Activation(Layer):
    def __init__(self, activation, use_graph=False, **kwargs):
        super().__init__(**kwargs)
        self.activation_name = activation
        self.use_graph = use_graph

        self.root_node_list = []
        self.leaf_node_list = []
        self.trainable = False
        self.layer_name = "activation"

        # ✅ 매핑 정의
        self.cuda_functions = {
            "relu": cuda_activations.relu,
            "sigmoid": cuda_activations.sigmoid,
            "tanh": cuda_activations.tanh,
        }

        self.graph_functions = {
            "relu": graph_activations.relu_graph,
            "sigmoid": graph_activations.sigmoid_graph,
            "tanh": graph_activations.tanh_graph,
        }

    def call(self, x, input_node=None):
        if self.use_graph:
            try:
                builder = self.graph_functions[self.activation_name]
            except KeyError:
                raise NotImplementedError(f"{self.activation_name} 그래프 미지원")

            root_node = builder(input_node)
            self.root_node_list = [root_node]
            self.leaf_node_list = [input_node]
            return root_node

        else:
            try:
                func = self.cuda_functions[self.activation_name]
            except KeyError:
                raise NotImplementedError(f"{self.activation_name} CUDA 미지원")

            return func(x)

    def compute_output_shape(self, input_shape):
        return input_shape

    def build(self, input_shape):
        self.input_shape = input_shape
        self.output_shape = input_shape
        super().build(input_shape)
