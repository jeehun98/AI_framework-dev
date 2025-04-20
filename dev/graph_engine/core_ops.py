from .node import Node

def matrix_add_nodes(A, B, result):
    """행렬 덧셈에 해당하는 계산 노드 생성"""
    rows, cols = len(A), len(A[0])
    if len(B) != rows or len(B[0]) != cols:
        raise ValueError("A와 B의 크기가 일치하지 않습니다.")

    root_node_list = []
    leaf_node_list = []

    for i in range(rows):
        for j in range(cols):
            add_node = Node(
                operation="add",
                input_value=A[i][j],
                weight_value=B[i][j],
                output=result[i][j],
                bias=0.0
            )
            root_node_list.append(add_node)
            leaf_node_list.append(add_node)  # 덧셈 노드는 입력에 직접 해당됨

    return root_node_list, leaf_node_list

from .node import Node

def matrix_multiply_nodes(A, B, result):
    """행렬 곱셈에 해당하는 계산 노드 생성
    - root_node_list: 출력 노드 (sum/add)
    - leaf_node_list: 입력 A에 해당하는 mul 노드만 수집
    """
    rows_A, cols_A = len(A), len(A[0])
    rows_B, cols_B = len(B), len(B[0])

    if cols_A != rows_B:
        raise ValueError("A의 열 크기와 B의 행 크기가 일치해야 합니다.")

    root_node_list = []
    leaf_node_list = []

    # ✅ 입력 A에 해당하는 mul 노드 추적용
    input_nodes = []

    for i in range(rows_A):         # 예: batch size (보통 1)
        for j in range(cols_B):     # 예: 출력 유닛 수
            sum_node = Node("add", 0.0, 0.0, result[i][j], 0.0)
            root_node_list.append(sum_node)

            for k in range(cols_A):  # 입력 차원 수
                mul_node = Node(
                    operation="multiply",
                    input_value=A[i][k],     # A의 값
                    weight_value=B[k][j],    # B의 weight
                    output=0.0
                )

                sum_node.add_child(mul_node)
                mul_node.add_parent(sum_node)

                # ✅ 최초 유닛에서만 입력 A 노드를 leaf로 수집
                if j == 0:  # 각 입력값에 대해 한 번만 leaf에 포함
                    leaf_node_list.append(mul_node)

    return root_node_list, leaf_node_list
