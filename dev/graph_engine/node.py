import math

class Node:
    valid_operations = {
        "add", "subtract", "multiply", "divide", "square",
        "exp", "neg", "reciprocal", "const"
    }

    def __init__(self, operation, input_value=0.0, weight_value=0.0, output=0.0, bias=0.0):
        if operation not in self.valid_operations:
            raise ValueError(f"Invalid operation: {operation}. Allowed: {self.valid_operations}")

        self.operation = operation
        self.input_value = input_value
        self.weight_value = weight_value
        self.output = output
        self.bias = bias
        self.grad_weight_total = 0.0

        self.parents = []
        self.children = []

    def add_parent(self, parent):
        self.parents.append(parent)
        parent.children.append(self)

    def add_child(self, child):
        self.children.append(child)
        child.parents.append(self)



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

    def compute(self):
        inputs = [p.output for p in self.parents]
        self.output = self._operation_func(self.operation)(inputs, self.weight_value, self.bias)
        return self.output

    def backpropagate(self, upstream_gradient=1.0):
        grad_input, grad_weight = self._gradient_func(self.operation)(
            self.input_value, self.weight_value, self.output, upstream_gradient
        )
        self.grad_weight_total += grad_weight
        for parent in self.parents:
            parent.backpropagate(grad_input)

    def update_weights(self, learning_rate):
        self.weight_value -= learning_rate * self.grad_weight_total
        self.grad_weight_total = 0.0

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

    def print_tree(self, depth=0, visited=None, allow_repeat=False):
        """
        계산 그래프 트리 구조를 출력합니다.
        
        - depth: 현재 트리 깊이 (재귀적으로 증가)
        - visited: 재방문 방지를 위한 노드 집합
        - allow_repeat: True일 경우, 같은 노드를 여러 경로에서 반복 출력함
        """
        if visited is None:
            visited = set()

        indent = " " * depth

        if not allow_repeat and self in visited:
            print(indent + f"↳ Node({self.operation}) (already visited)")
            return

        print(indent + f"Node({self.operation}) → output={self.output}, weight={self.weight_value}, grad_total={self.grad_weight_total}")

        if not allow_repeat:
            visited.add(self)

        for child in self.children:
            child.print_tree(depth + 2, visited, allow_repeat=allow_repeat)

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return id(self) == id(other)
    
    

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
        }[op]

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
        }[op]

    