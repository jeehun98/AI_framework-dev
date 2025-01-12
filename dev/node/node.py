class Node:
    def __init__(self, operation, input_value=0.0, weight_value=0.0, output=0.0, bias=0.0):
        """
        Node 클래스 초기화.

        Parameters:
        - operation: 수행할 연산 종류 (예: 'add', 'multiply')
        - input_value: 입력 값
        - weight_value: 가중치 값 (연산에 따라 사용)
        - output: 출력 값
        - bias: 바이어스 값
        """
        self.operation = operation
        self.input_value = input_value
        self.weight_value = weight_value
        self.output = output
        self.bias = bias
        self.grad_weight_total = 0.0  # 누적 그래디언트 (역전파 시 사용)
        self.parents = []  # 부모 노드 리스트
        self.children = []  # 자식 노드 리스트
        self._validate_operation()  # 연산 종류 검증

    def _validate_operation(self):
        """연산 종류가 유효한지 검증."""
        if self.operation not in self._operations():
            raise ValueError-(f"잘못된 연산: {self.operation}. "
                             f"가능한 연산: {', '.join(self._operations().keys())}")

    def add_parent(self, parent):
        """부모 노드 추가."""
        if parent not in self.parents:
            self.parents.append(parent)
            parent.add_child(self)  # 부모-자식 관계 설정

    def add_child(self, child):
        """자식 노드 추가."""
        if child not in self.children:
            self.children.append(child)

    def remove_parent(self, parent):
        """부모 노드 제거."""
        if parent in self.parents:
            self.parents.remove(parent)
            parent.children.remove(self)  # 부모-자식 관계 해제

    def remove_child(self, child):
        """자식 노드 제거."""
        if child in self.children:
            self.children.remove(child)
            child.parents.remove(self)

    def find_leaf_nodes(self):
        """그래프의 리프 노드(자식이 없는 노드)를 찾기."""
        leaf_nodes = []
        self._find_leaf_nodes_recursive(self, leaf_nodes, set())
        return leaf_nodes

    def _find_leaf_nodes_recursive(self, node, leaf_nodes, visited):
        """
        리프 노드 탐색을 위한 재귀 함수.

        Parameters:
        - node: 탐색할 현재 노드
        - leaf_nodes: 리프 노드를 저장할 리스트
        - visited: 방문한 노드를 기록한 집합
        """
        if node in visited:
            return
        visited.add(node)
        if not node.children:  # 자식이 없으면 리프 노드
            leaf_nodes.append(node)
        else:
            for child in node.children:
                self._find_leaf_nodes_recursive(child, leaf_nodes, visited)

    def compute(self):
        """
        현재 노드의 출력 값을 계산.
        
        부모 노드의 출력 값을 가져와 연산 수행.
        """
        input_values = [parent.output for parent in self.parents]
        self.output = self._operations()[self.operation](input_values, self.weight_value, self.bias)
        return self.output

    def backpropagate(self, upstream_gradient=1.0):
        """
        역전파(backpropagation)를 수행하여 그래디언트 계산.
        
        Parameters:
        - upstream_gradient: 상위 노드로부터 전달된 그래디언트
        """
        gradients = self._calculate_gradient(upstream_gradient)
        grad_input, grad_weight = gradients
        self.grad_weight_total += grad_weight  # 누적 그래디언트 업데이트
        for parent in self.parents:  # 부모 노드로 역전파
            parent.backpropagate(grad_input)

    def _calculate_gradient(self, upstream_gradient):
        """
        현재 노드의 그래디언트 계산.

        Parameters:
        - upstream_gradient: 상위 노드로부터 전달된 그래디언트

        Returns:
        - (grad_input, grad_weight): 입력 및 가중치에 대한 그래디언트
        """
        grad_fn = self._operations_gradient()[self.operation]
        return grad_fn(self.input_value, self.weight_value, self.output, upstream_gradient)

    def update_weights(self, learning_rate):
        """
        가중치를 업데이트.
        
        Parameters:
        - learning_rate: 학습률
        """
        self.weight_value -= learning_rate * self.grad_weight_total
        self.grad_weight_total = 0.0  # 누적 그래디언트 초기화

    def print_tree(self, depth=0, visited=None):
        """
        그래프 트리를 출력.

        Parameters:
        - depth: 출력 시 들여쓰기 레벨
        - visited: 이미 방문한 노드 집합
        """
        if visited is None:
            visited = set()
        if self in visited:  # 순환 참조 방지
            print(" " * depth + f"Node({self.operation}): (이미 방문됨)")
            return
        visited.add(self)
        print(" " * depth + f"Node({self.operation}): output={self.output}, weight={self.weight_value}, grad_total={self.grad_weight_total}")
        for child in self.children:
            child.print_tree(depth + 2, visited)

    @staticmethod
    def _operations():
        """지원되는 연산."""
        return {
            "add": lambda inputs, weight, bias: sum(inputs) + bias,
            "subtract": lambda inputs, weight, bias: inputs[0] - inputs[1] + bias,
            "multiply": lambda inputs, weight, bias: inputs[0] * weight + bias,
            "divide": lambda inputs, weight, bias: inputs[0] / (weight if weight != 0 else 1) + bias,
            "square": lambda inputs, weight, bias: inputs[0] ** 2 + bias,
        }

    @staticmethod
    def _operations_gradient():
        """연산별 그래디언트 계산 함수."""
        return {
            "add": lambda input_value, weight, output, upstream: (upstream, upstream),
            "subtract": lambda input_value, weight, output, upstream: (upstream, -upstream),
            "multiply": lambda input_value, weight, output, upstream: (upstream * weight, upstream * input_value),
            "divide": lambda input_value, weight, output, upstream: (
                upstream / (weight if weight != 0 else 1), -upstream * input_value / (weight ** 2 if weight != 0 else 1)
            ),
            "square": lambda input_value, weight, output, upstream: (2 * input_value * upstream, 0.0),
        }
