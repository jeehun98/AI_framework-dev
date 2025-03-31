from dev.cal_graph.node import Node

def matrix_add_nodes(A, B, result):
    """행렬 덧셈에 해당하는 계산 노드 생성"""
    rows, cols = len(A), len(A[0])
    if len(B) != rows or len(B[0]) != cols:
        raise ValueError("A와 B의 크기가 일치하지 않습니다.")

    new_nodes = []
    for i in range(rows):
        for j in range(cols):
            add_node = Node(
                operation="add",
                input_value=A[i][j],
                weight_value=B[i][j],
                output=result[i][j],
                bias=0.0
            )
            new_nodes.append(add_node)

    return new_nodes


def matrix_multiply_nodes(A, B, result):
    """행렬 곱셈에 해당하는 계산 노드 생성"""
    rows_A, cols_A = len(A), len(A[0])
    rows_B, cols_B = len(B), len(B[0])

    if cols_A != rows_B:
        raise ValueError("A의 열 크기와 B의 행 크기가 일치해야 합니다.")

    new_nodes = []
    for i in range(rows_A):
        for j in range(cols_B):
            sum_node = Node("add", 0.0, 0.0, result[i][j], 0.0)

            for k in range(cols_A):
                mul_node = Node("multiply", A[i][k], B[k][j], 0.0, 0.0)
                sum_node.add_child(mul_node)
                mul_node.add_parent(sum_node)

            new_nodes.append(sum_node)

    return new_nodes
