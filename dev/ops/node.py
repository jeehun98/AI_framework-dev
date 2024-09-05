"""
계산 그래프가 생성됨과 함께 노드가 생성된다.
"""

class Node:
    def __init__(self, value, children=(), op=None):
        """
        행렬 곱셈의 생각, 각 원소별 곱셈 연산을 수행하는 n 개의 노드와 
        n 개의 노드 합을 계산하는 노드가 필요, 

        (m,n)(n,l) 로 확장하면, (m * l) * n + (m * l) 개의 노드가 필요함 
        
        """
        
        # 노드에 필요한 정보로 입력값과 가중치값
        self.input = None
        self.weight = None
        
        self.value = value          # 이 노드의 값 (순전파에서 계산된 값)

        self.grad = 0.0             # 이 노드에 대한 그라디언트 (역전파에서 계산될 값)

        self.children = children    # 이 노드를 생성한 입력 노드들

        self.op = op                # 이 노드를 생성한 연산 (예: +, *, etc.)



    def backward(self):
        # 각 자식 노드에 대한 그라디언트를 계산하고 전파
        for child in self.children:
            grad_value = self.op.grad(self, child) * self.grad
            child.grad += grad_value
            child.backward()  # 재귀적으로 역전파 수행
