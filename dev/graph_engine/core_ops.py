from .node import Node

# ✅ 행렬 덧셈 계산 그래프
def matrix_add_nodes(A, B, result):
    """
    행렬 덧셈: C = A + B

    🔹 계산 그래프 구조 (2x2 행렬 예시)
        각 원소마다 독립적인 add 노드 생성

        [add]   [add]
         A+B     A+B
         │        │
        ...      ...

    👉 특징: 
    - 입력 A[i][j]와 B[i][j] 값으로 각각 add 노드 생성
    - root_node_list == leaf_node_list (덧셈 노드가 곧 입력)
    """
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

# ------------------------------------------------
def matrix_multiply_nodes(A, B, result):
    """
    행렬 곱셈: C = A x B
    - A : 입력 데이터 (Input)
    - B : 가중치 (Weights)
    - leaf_node_list 에는 입력 데이터(A)에 해당하는 노드만 추가
    """

    rows_A, cols_A = len(A), len(A[0])
    rows_B, cols_B = len(B), len(B[0])

    if cols_A != rows_B:
        raise ValueError("A의 열 크기와 B의 행 크기가 일치해야 합니다.")

    root_node_list = []
    leaf_node_list = []

    for i in range(rows_A):         # 보통 batch size (대부분 1)
        for j in range(cols_B):     # 출력 유닛 수
            sum_node = Node("add", input_value=0.0, weight_value=0.0, output=result[i][j], bias=0.0)
            root_node_list.append(sum_node)

            for k in range(cols_A):  # 입력 차원 수
                mul_node = Node(
                    operation="multiply",
                    input_value=A[i][k],     # ✅ 입력 데이터 값
                    weight_value=B[k][j],    # ✅ 가중치 값
                    output=0.0
                )

                sum_node.add_child(mul_node)

                # ✅ 입력값(A)에 해당하는 노드만 leaf_node_list에 추가
                leaf_node_list.append(mul_node)

    return root_node_list, leaf_node_list

