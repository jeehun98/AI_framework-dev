from dev.ops.node import Node

class Add:
    @staticmethod
    def forward(x, y):
        return Node(x.value + y.value, (x, y), Add)

    @staticmethod
    def grad(self, child):
        return 1.0  # 덧셈에 대한 미분은 항상 1

class Multiply:
    @staticmethod
    def forward(x, y):
        return Node(x.value * y.value, (x, y), Multiply)

    @staticmethod
    def grad(self, child):
        return self.children[1].value if child is self.children[0] else self.children[0].value
