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

        new_parent_nodes = []

        for i in range(rows):
            for j in range(cols):
                index = i * cols + j
                valueB = B[i][j]

                # 기존 노드가 있으면 가져와서 연결
                if node_list and index < len(node_list):
                    prev_node = node_list[index]

                    # 기존 노드의 output을 새로운 add 노드의 input으로 설정
                    add_node = Node(
                        operation="add",
                        input_value=prev_node.output,  # 기존 노드의 output을 새로운 노드의 input으로
                        weight_value=valueB,
                        output=0.0,
                        bias=0.0
                    )

                    # 기존 노드와 새로운 노드 연결
                    add_node.add_parent(prev_node)
                    prev_node.add_child(add_node)

                else:
                    # 기존 노드가 없으면 새로운 add 노드 생성
                    add_node = Node(
                        operation="add",
                        input_value=0.0,  # 초기값
                        weight_value=valueB,
                        output=0.0,
                        bias=0.0
                    )

                new_parent_nodes.append(add_node)

        # ✅ 기존 node_list를 유지하면서 새로운 부모 노드 추가
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

        for i in range(rows_result):
            for j in range(cols_result):
                index = i * cols_result + j
                prev_node = node_list[index] if node_list and index < len(node_list) else None

                sum_node = Node("add", 0.0, 0.0, 0.0, 0.0)

                for k in range(cols_A):
                    valueA = A[i][k]
                    valueB = B[k][j]

                    mul_node = Node("multiply", valueA, valueB, 0.0, 0.0)

                    sum_node.add_child(mul_node)
                    mul_node.add_parent(sum_node)

                if prev_node:
                    prev_node.add_child(sum_node)
                    sum_node.add_parent(prev_node)

                new_parent_nodes.append(sum_node)

        # ✅ 기존 node_list를 유지하면서 새로운 부모 노드 추가
        self.node_list = new_parent_nodes

        return self.node_list

    def print_graph(self):
        """
        계산 그래프를 계층적으로 출력 (들여쓰기 적용).
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

        # ✅ `node_list`가 올바르게 최상위 부모 노드를 포함하는지 검사
        if not self.node_list:
            print("🚨 [ERROR] 계산 그래프가 비어 있습니다.")
            return

        root_nodes = [node for node in self.node_list if not node.parents]

        if not root_nodes:
            print("🚨 [WARNING] 루트 노드를 찾을 수 없습니다. `node_list` 확인 필요.")
            return

        visited_nodes = set()
        for root in root_nodes:
            print_node(root, depth=0, visited=visited_nodes)

