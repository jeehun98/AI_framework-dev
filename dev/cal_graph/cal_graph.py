from dev.node.node import Node
import numpy as np

class Cal_graph:
    def __init__(self):
        """계산 그래프 초기화."""
        self.node_list = []  # 생성된 노드들을 저장

    def matrix_add(self, A, B, result):
        """
        행렬 덧셈 계산 그래프 생성 및 결과값 저장.

        Parameters:
        - A: 입력 행렬 A (2D 리스트)
        - B: 입력 행렬 B (2D 리스트)
        - result: CUDA 연산 후 결과 행렬 (2D 리스트)

        Returns:
        - new_parent_nodes: 생성된 최상위 부모 노드 리스트
        """
        rows, cols = len(A), len(A[0])

        if len(B) != rows or len(B[0]) != cols:
            raise ValueError("A와 B의 크기가 일치하지 않습니다.")

        new_parent_nodes = []
        for i in range(rows):
            for j in range(cols):
                valueA = A[i][j]
                valueB = B[i][j]
                output_value = result[i][j]  # ✅ CUDA 연산 결과값 적용

                # 덧셈 노드 생성
                add_node = Node(
                    operation="add",
                    input_value=valueA,
                    weight_value=valueB,
                    output=output_value,  # ✅ 결과값을 output에 저장
                    bias=0.0
                )

                new_parent_nodes.append(add_node)

        self.node_list = new_parent_nodes

        return new_parent_nodes

    def matrix_multiply(self, A, B, result):
        """
        행렬 곱셈 계산 그래프 생성 및 결과값 저장.

        Parameters:
        - A: 입력 행렬 A (2D 리스트)
        - B: 입력 행렬 B (2D 리스트)
        - result: CUDA 연산 후 결과 행렬 (2D 리스트)

        Returns:
        - new_parent_nodes: 생성된 최상위 부모 노드 리스트
        """
        rows_A, cols_A = len(A), len(A[0])
        rows_B, cols_B = len(B), len(B[0])

        if cols_A != rows_B:
            raise ValueError("A의 열 크기와 B의 행 크기가 일치해야 합니다.")

        rows_result, cols_result = rows_A, cols_B
        new_parent_nodes = []

        for i in range(rows_result):
            for j in range(cols_result):
                sum_node = Node("add", 0.0, 0.0, result[i][j], 0.0)  # ✅ CUDA 결과값 저장

                for k in range(cols_A):
                    valueA = A[i][k]
                    valueB = B[k][j]
                    

                    mul_node = Node("multiply", valueA, valueB, 0.0, 0.0)

                    sum_node.add_child(mul_node)
                    mul_node.add_parent(sum_node)

                new_parent_nodes.append(sum_node)

        self.node_list = new_parent_nodes

        return new_parent_nodes

    def update_output_values(self, result):
        """
        계산 그래프의 노드 `output` 값을 CUDA 연산 결과로 업데이트.

        Parameters:
        - result: CUDA 연산 결과 행렬 (2D 리스트)
        """
        for idx, node in enumerate(self.node_list):
            row, col = divmod(idx, len(result[0]))  # 2D 매핑
            node.output = result[row][col]

    def print_graph(self):
        """
        들여쓰기를 적용하여 계산 그래프를 계층적으로 출력.
        """
        def print_node(node, depth=0, visited=None):
            if visited is None:
                visited = set()
            if node in visited:
                return  # 무한 루프 방지
            visited.add(node)

            indent = "  " * depth  # 들여쓰기 적용
            print(f"{indent}Node(operation={node.operation}, input={node.input_value}, weight={node.weight_value}, output={node.output})")

            for child in node.children:
                print_node(child, depth + 1, visited)

        if not self.node_list:
            print("🚨 [ERROR] 계산 그래프가 비어 있습니다.")
            return

        root_nodes = self.node_list

        if not root_nodes:
            print("🚨 [WARNING] 루트 노드를 찾을 수 없습니다. `node_list` 확인 필요.")
            return

        visited_nodes = set()
        for root in root_nodes:
            print_node(root, depth=0, visited=visited_nodes)
