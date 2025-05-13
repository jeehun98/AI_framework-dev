import numpy as np
from dev.layers.layer import Layer
from dev.backend.backend_ops.activations import activations as cuda_activations

OP_TYPES = {
    "activation_sigmoid": 3,
    "activation_relu": 4,
    "activation_tanh": 5,
}

class ActivationMat(Layer):
    def __init__(self, activation, **kwargs):
        super().__init__(**kwargs)
        self.activation_name = activation  # 통일된 명칭 사용
        self.layer_name = "activation"
        self.trainable = False
        self.input_dim = None
        self.output_dim = None

        self.cuda_functions = {
            "relu": cuda_activations.relu,
            "sigmoid": cuda_activations.sigmoid,
            "tanh": cuda_activations.tanh,
        }

    def build(self, input_dim):
        self.input_dim = input_dim
        self.output_dim = input_dim  # activation은 shape 유지

    def call(self, inputs):
        inputs = np.atleast_2d(inputs).astype(np.float32)

        try:
            activation_func = self.cuda_functions[self.activation_name]
        except KeyError:
            raise NotImplementedError(f"[ERROR] '{self.activation_name}' CUDA 미지원")

        return activation_func(inputs)

    def forward_matrix(self):
        """컴파일 시 사용되는 activation 정보"""
        return {
            "type": "activation",
            "activation": self.activation_name,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
        }

    def generate_graph_matrices(self, input_ids, node_counter):
        op_matrix = []
        input_matrix = []
        param_vector = []

        act_op_name = f"activation_{self.activation_name}"
        act_op_code = OP_TYPES[act_op_name]

        act_id = node_counter
        op_matrix.append(act_op_code)
        input_matrix.append([input_ids[0], -1])
        param_vector.append(None)
        node_counter += 1

        return {
            "op_matrix": op_matrix,
            "input_matrix": input_matrix,
            "param_vector": param_vector,
            "output_ids": [act_id],
            "next_node_counter": node_counter
        }