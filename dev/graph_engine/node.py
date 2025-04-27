import math

# 1️⃣ 연산별 값 저장 정책 정의
VALUE_POLICY = {
    "add": True,
    "subtract": True,
    "multiply": True,
    "divide": True,
    "square": True,
    "exp": True,
    "neg": False,           # 미분이 고정값
    "reciprocal": True,
    "const": True,
    "mean": False,          # 단순 구조적 연산
    "sum": False            # 단순 구조적 연산
}

class Node:
    valid_operations = set(VALUE_POLICY.keys())

    def __init__(self, operation, input_value=0.0, weight_value=0.0, output=0.0, bias=0.0):
        if operation not in self.valid_operations:
            raise ValueError(f"Invalid operation: {operation}. Allowed: {self.valid_operations}")

        self.operation = operation
        self.requires_value = VALUE_POLICY[operation]

        # 2️⃣ 값 저장 정책에 따라 초기화
        if self.requires_value:
            self.input_value = input_value
            self.output = output
        else:
            self.input_value = None
            self.output = None

        self.weight_value = weight_value
        self.bias = bias
        self.grad_weight_total = 0.0

        self.parents = []
        self.children = []

    # 부모-자식 연결 메서드
    def add_parent(self, parent):
        self.parents.append(parent)
        parent.children.append(self)

    def add_child(self, child):
        self.children.append(child)
        child.parents.append(self)

    # 부모/자식 연결 해제
    def remove_parent(self, parent):
        if parent in self.parents:
            self.parents.remove(parent)
            if self in parent.children:
                parent.children.remove(self)

    def remove_child(self, child):
        if child in self.children:
            self.children.remove(child)
            if self in child.parents:
                child.parents.remove(self)

    # 순전파 연산
    def compute(self):
        inputs = [p.output for p in self.parents]
        self.output = self._operation_func(self.operation)(inputs, self.weight_value, self.bias)
        return self.output

    # 역전파 연산
    def backpropagate(self, upstream_gradient=1.0):
        # 3️⃣ 값 필요 여부에 따라 처리
        if self.requires_value:
            grad_input, grad_weight = self._gradient_func(self.operation)(
                self.input_value, self.weight_value, self.output, upstream_gradient
            )
        else:
            grad_input, grad_weight = self._gradient_func(self.operation)(
                0.0, self.weight_value, 0.0, upstream_gradient   # 더미값 처리
            )
        self.grad_weight_total += grad_weight
        for parent in self.parents:
            parent.backpropagate(grad_input)

    # 가중치 업데이트
    def update_weights(self, learning_rate):
        self.weight_value -= learning_rate * self.grad_weight_total
        self.grad_weight_total = 0.0

    # 리프 노드 탐색
    def find_leaf_nodes(self):
        leaf_nodes, visited = [], set()

        def dfs(node):
            if node in visited:
                return
            visited.add(node)
            if not node.children:
                leaf_nodes.append(node)
            for child in node.children:
                dfs(child)

        dfs(self)
        return leaf_nodes

    # 트리 형태 출력
    def print_tree(self, node, prefix="", is_last=True):
        connector = "└── " if is_last else "├── "
        print(prefix + connector +
              f"[{node.operation}] out={node.output} weight={node.weight_value} "
              f"id={id(node)} | children={len(node.children)}")

        child_count = len(node.children)
        for idx, child in enumerate(node.children):
            is_last_child = (idx == child_count - 1)
            next_prefix = prefix + ("    " if is_last else "│   ")
            self.print_tree(child, next_prefix, is_last_child)

    # Node 비교 & 해시
    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return id(self) == id(other)

    # 연산 정의 (Forward)
    @staticmethod
    def _operation_func(op):
        return {
            "add": lambda inputs, w, b: sum(inputs) + b,
            "subtract": lambda inputs, w, b: inputs[0] - inputs[1] + b,
            "multiply": lambda inputs, w, b: inputs[0] * w + b,
            "divide": lambda inputs, w, b: inputs[0] / (w if w != 0 else 1e-6) + b,
            "square": lambda inputs, w, b: inputs[0] ** 2 + b,
            "exp": lambda inputs, w, b: math.exp(inputs[0]) + b,
            "neg": lambda inputs, w, b: -inputs[0] + b,
            "reciprocal": lambda inputs, w, b: 1.0 / (inputs[0] if inputs[0] != 0 else 1e-6) + b,
            "const": lambda inputs, w, b: inputs[0] if inputs else 0.0,
            "mean": lambda inputs, w, b: sum(inputs) / len(inputs) if inputs else 0.0,
            "sum": lambda inputs, w, b: sum(inputs),
        }[op]

    # 미분 정의 (Backward)
    @staticmethod
    def _gradient_func(op):
        return {
            "add": lambda x, w, out, grad: (grad, grad),
            "subtract": lambda x, w, out, grad: (grad, -grad),
            "multiply": lambda x, w, out, grad: (grad * w, grad * x),
            "divide": lambda x, w, out, grad: (
                grad / (w if w != 0 else 1e-6),
                -grad * x / ((w if w != 0 else 1e-6) ** 2),
            ),
            "square": lambda x, w, out, grad: (2 * x * grad, 0.0),
            "exp": lambda x, w, out, grad: (out * grad, 0.0),
            "neg": lambda x, w, out, grad: (-grad, 0.0),
            "reciprocal": lambda x, w, out, grad: (-1.0 / (x ** 2 if x != 0 else 1e-6) * grad, 0.0),
            "const": lambda x, w, out, grad: (0.0, 0.0),
            "mean": lambda x, w, out, grad: (grad / (len(x) if len(x) > 0 else 1), 0.0),
            "sum": lambda x, w, out, grad: (grad, 0.0),
        }[op]
