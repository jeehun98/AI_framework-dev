import os
os.add_dll_directory("C:\\msys64\\mingw64\\bin")

# 노드 클래스 정의
class Node:

    def is_root(self, node):
        if node.get_parents() is not None:
            return False
        else:
            return True
        
    def is_leaf(self, node):
        if node.get_children() is not None:
            return False
        else:
            return True        
    

    def backpropagate(self, node, upstream_gradient=1.0, leaf_nodes=None):
        if leaf_nodes is None:
            leaf_nodes = []

        # 1. 현재 노드에서 그래디언트 계산
        # node.operation 별로 알맞게 저장됨...
        grad_input, grad_weight = node.calculate_gradient(upstream_gradient)

        # 2. 부모 노드로 전파된 그래디언트 합산
        node.grad_input += grad_input
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
    def weight_update(self, node):
        pass

    def find_child_node(self, node, leaf_nodes=None, visited=None):
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

    
    def find_root(self, node):
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

            print(len(leaf_nodes), len(child_nodes), "길이가 달라?")

            # 리프 노드와 자식 노드의 길이 확인
            if len(leaf_nodes) != len(child_nodes):
                raise ValueError("Mismatch in number of leaf nodes and child nodes.")

            for i in range(len(leaf_nodes)):
                # 순환 참조 방지
                if child_nodes[i] not in leaf_nodes[i].get_children():
                    leaf_nodes[i].add_child(child_nodes[i])
                    child_nodes[i].add_parent(leaf_nodes[i])

        return parent_nodes

    def link_loss_node(self, parent_nodes, child_nodes):
        
        for i in range(len(parent_nodes)):
            parent_nodes[i].add_child(child_nodes[i])
            child_nodes[i].add_parent(parent_nodes[i])

        return parent_nodes

    def find_operation_node(self, node, operation):
        
        pass