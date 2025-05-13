# dev/graph_engine/graph_compiler.py

import numpy as np

class GraphCompiler:
    def __init__(self):
        self.op_matrix = []
        self.input_matrix = []
        self.param_vector = []
        self.output_ids = []  # 마지막 노드들의 index
        self.node_count = 0

    def add_layer(self, layer):
        result = layer.generate_graph_matrices(self.output_ids or [0], self.node_count)

        self.op_matrix += result["op_matrix"]
        self.input_matrix += result["input_matrix"]
        self.param_vector += result["param_vector"]
        self.output_ids = result["output_ids"]
        self.node_count = result["next_node_counter"]

    def build(self):
        # 정적 컴파일을 끝내고 numpy 형태로 변환
        self.op_matrix = np.array(self.op_matrix, dtype=np.int32)
        self.input_matrix = np.array(self.input_matrix, dtype=np.int32)
        self.param_vector = self.param_vector  # 이미 numpy 배열들이 있음

    def get_matrices(self):
        return {
            "op_matrix": self.op_matrix,
            "input_matrix": self.input_matrix,
            "param_vector": self.param_vector,
        }
