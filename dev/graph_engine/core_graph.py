# dev/cal_graph/core_graph.py

from . import core_ops, graph_utils
from .activations_graph import build_relu_node, build_sigmoid_node, build_tanh_node

# 나머지 Cal_graph 클래스 정의 ...

class Cal_graph:
    def __init__(self):
        self.node_list = []

    def apply_sigmoid(self):
        self.node_list = [build_sigmoid_node(node) for node in self.node_list]

    def apply_tanh(self):
        self.node_list = [build_tanh_node(node) for node in self.node_list]

    def apply_relu(self):
        self.node_list = [build_relu_node(node) for node in self.node_list]

    def add_matrix_add_graph(self, A, B, result):
        self.node_list = core_ops.matrix_add_nodes(A, B, result)
        return self.node_list

    def add_matrix_multiply_graph(self, A, B, result):
        self.node_list = core_ops.matrix_multiply_nodes(A, B, result)
        return self.node_list

    def update_output_values(self, result):
        for idx, node in enumerate(self.node_list):
            row, col = divmod(idx, len(result[0]))
            node.output = result[row][col]

    # ✅ 외부 유틸리티 사용
    def print_graph(self):
        graph_utils.print_graph(self.node_list)

    def connect_graphs(self, node_list1, node_list2):
        return graph_utils.connect_graphs(node_list1, node_list2)

    def get_leaf_nodes(self, node_list):
        return graph_utils.get_leaf_nodes(node_list)
