# dev/cal_graph/core_graph.py

from . import core_ops, graph_utils
from .activations_graph import build_relu_node, build_sigmoid_node, build_tanh_node

# 나머지 Cal_graph 클래스 정의 ...

class Cal_graph:
    def __init__(self):
        self.root_node_list = []   # ✅ 출력 노드
        self.leaf_node_list = []   # ✅ 입력 연결 대상 노드

    def apply_sigmoid(self):
        new_root_list = []
        new_leaf_list = []
        for _ in self.root_node_list:
            root, leaf = build_sigmoid_node()
            new_root_list.append(root)
            new_leaf_list.extend(leaf)
        self.root_node_list = new_root_list
        self.leaf_node_list = new_leaf_list

    def apply_tanh(self):
        new_root_list = []
        new_leaf_list = []
        for _ in self.root_node_list:
            root, leaf = build_tanh_node()
            new_root_list.append(root)
            new_leaf_list.extend(leaf)
        self.root_node_list = new_root_list
        self.leaf_node_list = new_leaf_list

    def apply_relu(self):
        new_root_list = []
        new_leaf_list = []
        for _ in self.root_node_list:
            root, leaf = build_relu_node()
            new_root_list.append(root)
            new_leaf_list.extend(leaf)
        self.root_node_list = new_root_list
        self.leaf_node_list = new_leaf_list

    def add_matrix_add_graph(self, A, B, result):
        root_nodes, leaf_nodes = core_ops.matrix_add_nodes(A, B, result)
        self.root_node_list = root_nodes
        self.leaf_node_list = leaf_nodes
        return root_nodes, leaf_nodes

    def add_matrix_multiply_graph(self, A, B, result):
        root_nodes, leaf_nodes = core_ops.matrix_multiply_nodes(A, B, result)
        self.root_node_list = root_nodes
        self.leaf_node_list = leaf_nodes
        return root_nodes, leaf_nodes

    def update_output_values(self, result):
        for idx, node in enumerate(self.root_node_list):
            row, col = divmod(idx, len(result[0]))
            node.output = result[row][col]

    def print_graph(self):
        graph_utils.print_graph(self.root_node_list)

    def connect_graphs(self, current_leaf_nodes, prev_root_nodes):
        return graph_utils.connect_graphs(current_leaf_nodes, prev_root_nodes)

    def get_leaf_nodes(self, node_list):
        return graph_utils.get_leaf_nodes(node_list)
