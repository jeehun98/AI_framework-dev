# dev/cal_graph/tests/test_core_graph.py
import numpy as np
from dev.cal_graph.core_graph import Cal_graph
from tests.test_setup import import_cuda_module

matrix_ops = import_cuda_module()

def test_graph_add_mul():
    cal_graph = Cal_graph()

    A = [[1, 2], [3, 4]]
    B = [[10, 20], [30, 40]]
    C = np.zeros((2, 2), dtype=np.float32)

    matrix_ops.matrix_mul(A, B, C)
    node_list1 = cal_graph.add_matrix_multiply_graph(A, B, C.tolist())

    D = [[1, 2], [3, 4]]
    E = np.zeros((2, 2), dtype=np.float32)

    matrix_ops.matrix_add(C, D, E)
    node_list2 = cal_graph.add_matrix_add_graph(C.tolist(), D, E.tolist())

    cal_graph.connect_graphs(node_list2, node_list1)

    assert E.tolist() == [[41, 44], [93, 96]]
