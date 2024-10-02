import os
os.add_dll_directory("C:\\msys64\\mingw64\\bin")

from dev.backend.operaters import operations_matrix
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
        print(' ' * indent + f"Node: {node.operation}, Weight: {node.weight_value}, Grad Total: {node.grad_weight_total}")

        # 자식 노드 출력
        children = node.get_children()
        if children:
            print(' ' * indent + 'Children:')
            for child in children:
                self.print_relationships(child, visited, indent + 4)
        else:
            print(' ' * (indent + 4) + 'Leaf node')


    
    # 리턴 값이 없음
    # node.h 코드의 실행
    def backpropagate(self, root_node, upstream_gradient = 1.0):
        visited = set()
        root_node.backpropagate(upstream_gradient, visited)
        
    # 새로운 laerning_rate 를 적용하는 방법
    def weight_update(self, root_node, batch_count, optimizer, learning_rate = 0.001):
        adjusted_lr = learning_rate / batch_count
        optimizer.update_all_weights(root_node, adjusted_lr)
        # optimizer.update(root_node)

                      
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


    def link_node(self, parent_nodes, child_nodes, layer_name):
        """
        노드 리스트를 받음
        parent_nodes : 해당 노드의 리프 노드와
        child_nodes : 해당 노드의 루트 노드와 연결해야 함
        """
        if not child_nodes:
            return parent_nodes
        
        
        if layer_name == "activation":
            print("걸렸나", len(parent_nodes), len(child_nodes))
            # 일대일 연결 시행
            self.link_loss_node(parent_nodes, child_nodes)
            return parent_nodes
        

        # 각 부모 노드의 리프 노드 탐색을 하는데...
        # 이렇게 나오게 된 이유가 dense 층끼리의 연결을 위한 방법이었네
        # layer 의 종류별 연결 방법을 다르게
        for parent_node in parent_nodes:
            leaf_nodes = self.find_child_node(parent_node)

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