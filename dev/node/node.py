import os
os.add_dll_directory("C:\\msys64\\mingw64\\bin")

# 노드 클래스 정의
class Node:
    
    def backpropagate(self, node, upstream_gradient=1.0, leaf_nodes=None):
        if leaf_nodes is None:
            leaf_nodes = []

        # 1. 현재 노드에서 그래디언트 계산
        grad_input, grad_weight = node.calculate_gradient(upstream_gradient)

        # 2. 부모 노드로 전파된 그래디언트 합산
        node.grad_input += grad_input
        node.grad_weight += grad_weight

        # 3. 자식 노드로 그래디언트 전파
        children = node.get_children()
        if not children:  # 자식 노드가 없으면 리프 노드
            leaf_nodes.append(node)
        else:
            for child in children:
                # 연산 타입에 따라 어떤 그래디언트를 넘길지 결정
                if node.operation in ['add', 'subtract']:
                    # 덧셈이나 뺄셈은 두 입력에 동일한 upstream_gradient를 전파
                    self.backpropagate(child, upstream_gradient, leaf_nodes)
                elif node.operation == 'multiply':
                    # 곱셈의 경우, 첫 번째 입력은 grad_b를, 두 번째 입력은 grad_a를 받음
                    if child == node.get_children()[0]:
                        self.backpropagate(child, grad_input, leaf_nodes)
                    else:
                        self.backpropagate(child, grad_weight, leaf_nodes)
                else:
                    # 다른 연산의 경우 기본적으로 grad_a를 넘김
                    self.backpropagate(child, grad_input, leaf_nodes)

        return leaf_nodes
    
    # 자식 노드 리스트 반환
    def find_child_node(self, node, leaf_nodes=None):
        if leaf_nodes is None:
            leaf_nodes = []

        children = node.get_children()
        
        if not children:  # 자식 노드가 없으면 리프 노드
            leaf_nodes.append(node)
        else:
            for child in children:
                self.find_child_node(child, leaf_nodes)
        
        return leaf_nodes
    
    def find_root(self, node):
        current_node = node
        while current_node.get_parents():  # 부모 노드가 있으면 계속 탐색
            current_node = current_node.get_parents()[0]
        return current_node