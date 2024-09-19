import os
os.add_dll_directory("C:\\msys64\\mingw64\\bin")

from dev.backend.node import node

# 노드 클래스 정의
class Node:

    def is_root(self):
        return not self.get_parents()

    def is_leaf(self):
        return not self.get_children()

    def print_relationships(self, node, visited=None, indent=0):
        """
        노드의 관계를 출력하는 함수

        Parameters:
        node: 탐색을 시작할 노드
        visited: 이미 방문한 노드를 추적하기 위한 집합 (기본값은 None)
        indent: 출력 시 들여쓰기 수준 (기본값은 0)
        """
        if visited is None:
            visited = set()

        # 순환 참조 방지
        if node in visited:
            print(' ' * indent + f'(Already visited node: {node.operation})')
            return
        visited.add(node)

        # 현재 노드 정보 출력
        print(' ' * indent + f'Node Operation: {node.operation}')
        print(' ' * indent + f'Inputs: {node.input_a}, {node.input_b}')
        print(' ' * indent + f'Output: {node.output}')
        print(' ' * indent + f'Grad Input: {node.grad_input}')
        print(' ' * indent + f'Grad Weight: {node.grad_weight}')

        # 부모 노드 출력
        parents = node.get_parents()
        if parents:
            print(' ' * indent + 'Parents:')
            for parent in parents:
                self.print_relationships(parent, visited, indent + 4)

        # 자식 노드 출력
        children = node.get_children()
        if children:
            print(' ' * indent + 'Children:')
            for child in children:
                self.print_relationships(child, visited, indent + 4)
        else:
            print(' ' * (indent + 4) + 'Leaf node')     
    



    def backpropagate(self, node, upstream_gradient=1.0, leaf_nodes=None):
        if leaf_nodes is None:
            leaf_nodes = []

        # 1. 현재 노드에서 그래디언트 계산
        # node.operation 별로 알맞게 저장됨...
        grad_input, grad_weight = node.calculate_gradient(upstream_gradient)

        # 2. 부모 노드로 전파된 그래디언트 합산
        # 가중치 변화에 대한 비용 함수의 변화량의 값은 누적
        # 향후 가중치 갱신에 사용함
        node.grad_weight += grad_weight

        # 3. 자식 노드로 그래디언트 전파
        children = node.get_children()
        if not children:  # 자식 노드가 없으면 리프 노드
            leaf_nodes.append(node)
        else:
            # 각 자식노드 리스트 내 접근
            for child in children:
                # 각 자식 노드 리스트 접근과 함께 입력 값의 변화량에 대한
                # 비용 함수의 변화량인, grad_input 값을 갱신해준다.
                self.backpropagate(child, grad_input, leaf_nodes)

        return leaf_nodes
    
    # 노드 내 저장된 정보를 통한 가중치 갱신
    def weight_update(self, node, batch_size):
        pass

    def find_child_node(self, node, leaf_nodes=None, visited=None):
        """
        노드의 리프 노드들을 찾는 함수

        Parameters:
        node: 탐색을 시작할 노드
        leaf_nodes: 리프 노드들을 저장할 리스트 (기본값은 None)
        visited: 이미 방문한 노드를 추적하기 위한 집합 (기본값은 None)

        Returns:
        leaf_nodes: 탐색된 리프 노드들의 리스트
        """
        if leaf_nodes is None:
            leaf_nodes = []
        
        if visited is None:
            visited = set()

        # 이미 방문한 노드라면 순환 참조로 간주하고 종료
        if node in visited:
            return leaf_nodes
        
        # 현재 노드를 방문했음으로 표시
        visited.add(node)

        children = node.get_children()
        
        if not children:  # 자식 노드가 없으면 리프 노드
            leaf_nodes.append(node)
        else:
            for child in children:
                self.find_child_node(child, leaf_nodes, visited)
        
        return leaf_nodes

    
    def find_root(node):
        """
        주어진 노드에서 루트 노드를 찾는 함수

        Parameters:
        node: 탐색을 시작할 노드

        Returns:
        root_node: 루트 노드
        """
        current_node = node
        while current_node.get_parents():  # 부모 노드가 있으면 계속 탐색
            current_node = current_node.get_parents()[0]
        return current_node


    def link_node(self, parent_nodes, child_nodes):
        """
        노드 리스트를 받음
        parent_nodes : 해당 노드의 리프 노드와
        child_nodes : 해당 노드의 루트 노드와 연결해야 함
        """
        if not child_nodes:
            return parent_nodes

        for parent_node in parent_nodes:
            leaf_nodes = self.find_child_node(parent_node)

            # 리프 노드와 자식 노드의 길이 확인
            # print(len(leaf_nodes), len(child_nodes))
            if len(leaf_nodes) != len(child_nodes):
                raise ValueError("Mismatch in number of leaf nodes and child nodes.")

            for i in range(len(leaf_nodes)):
                # 순환 참조 방지
                if child_nodes[i] not in leaf_nodes[i].get_children():
                    leaf_nodes[i].add_child(child_nodes[i])
                    child_nodes[i].add_parent(leaf_nodes[i])

        return parent_nodes


    def link_loss_node(self, parent_nodes, child_nodes):
        """
        손실 노드와 이전 노드들을 연결하는 함수

        Parameters:
        parent_nodes: 부모 노드들의 리스트
        child_nodes: 자식 노드들의 리스트

        Returns:
        parent_nodes: 연결된 부모 노드들의 리스트
        """
        if len(parent_nodes) != len(child_nodes):
            raise ValueError("The number of parent nodes and child nodes must be the same.")
        
        for i in range(len(parent_nodes)):
            parent_nodes[i].add_child(child_nodes[i])
            child_nodes[i].add_parent(parent_nodes[i])

        return parent_nodes

    def find_operation_node(self, node, operation):
        
        pass