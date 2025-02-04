from dev.node.node import Node

class Cal_graph:
    def __init__(self):
        """계산 그래프 초기화."""
        self.node_list = []  # 생성된 노드들을 저장

    def matrix_add(self, A, B, node_list=None):
        """
        행렬 덧셈 계산 그래프 생성 및 기존 계산 그래프 확장.

        Parameters:
        - A: 입력 행렬 A (2D 리스트)
        - B: 입력 행렬 B (2D 리스트)
        - node_list: 기존 계산 그래프의 최상위 부모 노드 리스트 (옵션)

        Returns:
        - node_list: 갱신된 최상위 부모 노드 리스트
        """
        rows, cols = len(A), len(A[0])

        if len(B) != rows or len(B[0]) != cols:
            raise ValueError("A와 B의 크기가 일치하지 않습니다.")

        # 기존 node_list가 주어진 경우, 이전 연산의 결과와 새로운 연산을 연결
        if node_list:
            if len(node_list) != rows * cols:
                raise ValueError("node_list 크기가 행렬 크기와 일치해야 합니다.")

            new_parent_nodes = []
            for i in range(rows):
                for j in range(cols):
                    index = i * cols + j
                    prev_node = node_list[index]  # 기존 계산 그래프의 노드

                    valueB = B[i][j]

                    # 새로운 add 노드 생성 (기존 노드 + 새로운 값)
                    add_node = Node(
                        operation="add",
                        input_value=prev_node.output,  # 이전 노드의 출력이 새로운 입력이 됨
                        weight_value=valueB,
                        output=0.0,
                        bias=0.0
                    )

                    # 기존 노드와 새로운 노드를 연결
                    add_node.add_parent(prev_node)
                    prev_node.add_child(add_node)

                    new_parent_nodes.append(add_node)

            self.node_list = new_parent_nodes
        else:
            # 처음 생성하는 경우, 기존 방식으로 수행
            new_parent_nodes = []
            for i in range(rows):
                for j in range(cols):
                    valueA = A[i][j]
                    valueB = B[i][j]

                    # 덧셈 노드 생성
                    add_node = Node(
                        operation="add",
                        input_value=valueA,
                        weight_value=valueB,
                        output=0.0,
                        bias=0.0
                    )
                    new_parent_nodes.append(add_node)

            self.node_list = new_parent_nodes

        return self.node_list

    def matrix_multiply(self, A, B, node_list=None):
        """
        행렬 곱셈 계산 그래프 생성 및 기존 계산 그래프 확장.

        Parameters:
        - A: 입력 행렬 A (2D 리스트)
        - B: 입력 행렬 B (2D 리스트)
        - node_list: 기존 계산 그래프의 최상위 부모 노드 리스트 (옵션)

        Returns:
        - node_list: 갱신된 최상위 부모 노드 리스트
        """
        rows_A, cols_A = len(A), len(A[0])
        rows_B, cols_B = len(B), len(B[0])

        if cols_A != rows_B:
            raise ValueError("A의 열 크기와 B의 행 크기가 일치해야 합니다.")

        rows_result, cols_result = rows_A, cols_B
        new_parent_nodes = []

        if node_list:
            if len(node_list) != rows_result * cols_result:
                raise ValueError("node_list 크기가 결과 행렬 크기와 일치해야 합니다.")

            for i in range(rows_result):
                for j in range(cols_result):
                    index = i * cols_result + j
                    prev_node = node_list[index]  # 기존 계산 그래프의 노드

                    sum_node = Node("add", 0.0, 0.0, 0.0, 0.0)

                    for k in range(cols_A):
                        valueA = A[i][k]
                        valueB = B[k][j]

                        mul_node = Node("multiply", valueA, valueB, 0.0, 0.0)

                        sum_node.add_child(mul_node)
                        mul_node.add_parent(sum_node)

                    prev_node.add_child(sum_node)
                    sum_node.add_parent(prev_node)

                    new_parent_nodes.append(sum_node)

            self.node_list = new_parent_nodes
        else:
            for i in range(rows_result):
                for j in range(cols_result):
                    sum_node = Node("add", 0.0, 0.0, 0.0, 0.0)

                    for k in range(cols_A):
                        valueA = A[i][k]
                        valueB = B[k][j]

                        mul_node = Node("multiply", valueA, valueB, 0.0, 0.0)

                        sum_node.add_child(mul_node)
                        mul_node.add_parent(sum_node)

                    new_parent_nodes.append(sum_node)

            self.node_list = new_parent_nodes

        return self.node_list

    def print_graph(self):
        """
        들여쓰기를 적용하여 계산 그래프를 계층적으로 출력.
        """
        def print_node(node, depth=0, visited=set()):
            if node in visited:
                return  # 무한 루프 방지 (순환 그래프 대비)
            visited.add(node)

            indent = "  " * depth  # 들여쓰기 적용
            print(f"{indent}Node(operation={node.operation}, input={node.input_value}, weight={node.weight_value}, output={node.output})")

            for child in node.children:
                print_node(child, depth + 1, visited)

        root_nodes = [node for node in self.node_list if not node.parents]
        visited_nodes = set()

        for root in root_nodes:
            print_node(root, depth=0, visited=visited_nodes)
