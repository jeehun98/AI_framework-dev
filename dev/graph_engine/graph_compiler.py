# dev/graph_engine/graph_compiler.py

import numpy as np

class GraphCompiler:
    def __init__(self):
        self.E_matrix = []  # [op_type, input_1, input_2, output, extra(optional)]
        self.node_count = 0
        self.weights = []
        self.biases = []
        self.var_map = {}  # 기록된 output 노드 ID

    def _new_id(self):
        self.node_count += 1
        return self.node_count - 1

    def _register_weight(self, W):
        idx = len(self.weights)
        self.weights.append(W)
        return -(idx + 1)  # 음수 인덱스: 가중치로 인식

    def _register_bias(self, b):
        idx = len(self.biases)
        self.biases.append(b)
        return -(idx + 1001)  # -1001부터: bias 인덱스

    def add_layer(self, layer):
        if layer.layer_name == "dense":
            self._compile_dense(layer)
        elif layer.layer_name == "activation":
            self._compile_activation(layer)
        elif layer.layer_name == "flatten":
            self._compile_flatten(layer)
        else:
            raise NotImplementedError(f"GraphCompiler: unknown layer {layer.layer_name}")

    def _compile_dense(self, layer):
        input_id = self.var_map.get("last_output", 0)
        W_id = self._register_weight(layer.weights)
        b_id = self._register_bias(layer.bias)

        matmul_out = self._new_id()
        add_out = self._new_id()

        self.E_matrix.append(["matmul", input_id, W_id, matmul_out, -1])
        self.E_matrix.append(["add", matmul_out, b_id, add_out, -1])

        self.var_map["last_output"] = add_out

        if layer.activation:
            act_out = self._new_id()
            self.E_matrix.append([f"activation_{layer.activation_name}", add_out, -1, act_out, -1])
            self.var_map["last_output"] = act_out

    def _compile_activation(self, layer):
        input_id = self.var_map.get("last_output")
        act_out = self._new_id()
        self.E_matrix.append([f"activation_{layer.activation_name}", input_id, -1, act_out, -1])
        self.var_map["last_output"] = act_out

    def _compile_flatten(self, layer):
        input_id = self.var_map.get("last_output", 0)
        flat_out = self._new_id()
        self.E_matrix.append(["flatten", input_id, -1, flat_out, -1])
        self.var_map["last_output"] = flat_out

    def get_compiled(self):
        return {
            "E": self.E_matrix,
            "W": self.weights,
            "b": self.biases,
            "input_node": 0,
            "output_node": self.var_map.get("last_output")
        }
