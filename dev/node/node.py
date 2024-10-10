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

    def print_summary(self, node, visited=None, indent=0):
        """
        노드의 간단한 관계 요약을 출력하는 함수

        Parameters:
        node: 탐색을 시작할 노드
        visited: 이미 방문한 노드를 추적하기 위한 집합 (기본값은 None)
        indent: 출력 시 들여쓰기 수준 (기본값은 0)
        """
        if visited is None:
            visited = set()

        # 순환 참조 방지
        if node in visited:
            return
        visited.add(node)

        # 현재 노드 정보 출력 (간단하게 표현)
        children = node.get_children()
        children_count = len(children)
        if children_count > 0:
            print(' ' * indent + f"Node: {node.operation}, Children Count: {children_count}")
        else:
            print(' ' * indent + f"Node: {node.operation} (Leaf)")

        # 동일한 연산 노드 요약 및 리프 노드 출력
        current_operation = None
        current_count = 0
        previous_child = None
        for child in children:
            if child.operation == current_operation:
                current_count += 1
            else:
                if current_count > 1:
                    print(' ' * (indent + 2) + f"Repeated Node: {current_operation}, Count: {current_count}")
                    if previous_child:
                        self.print_summary(previous_child, visited, indent + 4)
                elif current_count == 1:
                    self.print_summary(previous_child, visited, indent + 2)

                current_operation = child.operation
                current_count = 1
                previous_child = child

        # 마지막 반복 노드 출력 및 리프 노드 출력
        if current_count > 1:
            print(' ' * (indent + 2) + f"Repeated Node: {current_operation}, Count: {current_count}")
            if previous_child:
                self.print_summary(previous_child, visited, indent + 4)
        elif current_count == 1:
            self.print_summary(previous_child, visited, indent + 2)

        # 자식 노드 개별 출력
        for child in children:
            if child.operation != current_operation or current_count == 1:
                self.print_summary(child, visited, indent + 2)


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
            #print(' ' * indent + f'(Already visited node: {node.operation})')
            return
        visited.add(node)

        # 현재 노드 정보 출력
        print(' ' * indent + f"Node: {node.operation}, Weight: {node.weight_value}, Grad Total: {node.grad_weight_total}, node_value: {node.output}")

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

    # dense 층과의 연결, 일대일 대응
    def link_dense_node(self, parent_nodes, child_nodes):
        """
        노드 리스트를 받음
        parent_nodes : 해당 노드의 리프 노드와
        child_nodes : 해당 노드의 루트 노드와 연결해야 함
        """
        if not child_nodes:
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


    def link_node(self, current_layer, previous_layer):
        """
        노드 리스트를 받음
        parent_nodes : 해당 노드의 리프 노드와
        child_nodes : 해당 노드의 루트 노드와 연결해야 함
        """


        if current_layer.layer_name == "dense":
            return self.link_dense_node(current_layer.node_list, previous_layer.node_list)        
        
        elif current_layer.layer_name == "activation":
            return self.link_loss_node(current_layer.node_list, previous_layer.node_list)
        
        elif current_layer.layer_name =="pooling":
            return self.link_pool_node(current_layer, previous_layer)
        
        elif current_layer.layer_name =="conv2d":
            return self.link_conv2d_node(current_layer, previous_layer)



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

    def link_pool_node(self, current_layer, previous_layer):
        """
        풀링 레이어에서 Conv2D의 리프 노드와 Pooling 연산의 루트 노드를 연결하는 함수.
        parent_nodes : Conv2D 레이어의 리프 노드 (풀링 연산의 입력)
        child_nodes : Pooling 레이어의 루트 노드 (풀링 연산의 출력)
        conv_output_shape : Conv2D 레이어의 출력 크기 (height, width)
        stride : 풀링 연산의 stride 값
        pooling_size : 풀링 크기 (예: (2, 2))

        current_layer, previous_layer 를 전달받고 layer 정보로 처리해야겠다.
        """
        stride_height = current_layer.strides[0]
        stride_width = current_layer.strides[1]

        pool_height, pool_width = current_layer.pool_size

        conv_output_height, conv_output_width, num_channels = previous_layer.output_shape

        output_height = (conv_output_height - pool_height) // stride_height + 1
        output_width = (conv_output_width - pool_width) // stride_width + 1

        # 각 루트 노드의 개수들, 
        # previous 5,5,7 - 175, 
        # current 4,4,7 - 112 

        child_nodes = previous_layer.node_list
        parent_nodes = current_layer.node_list

       # Pooling 레이어의 노드 수가 올바르게 계산되었는지 확인
        if len(parent_nodes) != output_height * output_width * num_channels:
            raise ValueError("Mismatch in number of pooling regions and child nodes.")

        # 풀링 연산의 각 자식 노드를 그에 대응하는 부모 노드들과 연결
        for ch in range(num_channels):  # 채널 차원을 고려하여 각 채널별로 연결
            for h in range(output_height):
                for w in range(output_width):
                    parent_index = ch * output_height * output_width + h * output_width + w  # Pooling 노드의 인덱스
                    parent_node = parent_nodes[parent_index]  # 해당 풀링 노드

                    # 부모 노드들 중에서 풀링 영역에 속하는 노드를 찾아 연결
                    for i in range(pool_height):
                        for j in range(pool_width):
                            parent_h = h * stride_height + i  # 풀링에 포함되는 부모 노드의 y좌표
                            parent_w = w * stride_width + j  # 풀링에 포함되는 부모 노드의 x좌표
                            child_index = ch * conv_output_height * conv_output_width + parent_h * conv_output_width + parent_w  # 부모 노드의 인덱스
                            
                            # print(parent_index, child_index)
                            
                            child_node = child_nodes[child_index]  # 해당 부모 노드

                            # 부모 노드와 자식 노드 연결
                            if child_node not in parent_node.get_children():
                                parent_node.add_child(child_node)
                                child_node.add_parent(parent_node)

        current_layer.node_list = parent_nodes

        # self.print_relationships(parent_nodes[0])
        # print("단계 끝")

        return parent_nodes

    def link_conv2d_node(self, current_layer, previous_layer):
        """
        Conv2D 레이어의 리프 노드를 Pooling 레이어의 루트 노드와 연결하는 함수.
        current_layer : Conv2D 레이어 (출력)
        previous_layer : Pooling 레이어 (입력)
        """
        stride_height = current_layer.strides[0]
        stride_width = current_layer.strides[1]
        filter_height, filter_width = current_layer.kernel_size
        pool_output_height, pool_output_width, num_channels = previous_layer.output_shape

        output_height = (pool_output_height - filter_height) // stride_height + 1
        output_width = (pool_output_width - filter_width) // stride_width + 1

        parent_nodes = current_layer.node_list
        child_nodes = previous_layer.node_list

        # Conv2D 레이어의 리프 노드를 찾음
        leaf_nodes = [self.find_child_node(node) for node in parent_nodes]
        leaf_nodes_flat = [leaf for sublist in leaf_nodes for leaf in sublist]

        # 노드 개수 검증
        if len(leaf_nodes_flat) != output_height * output_width * current_layer.output_shape[2] * filter_height * filter_width * num_channels:
            raise ValueError("Mismatch in number of conv regions and leaf nodes.")

        leaf_index = 0  # 리프 노드를 순차적으로 접근하기 위한 인덱스

        for out_ch in range(current_layer.output_shape[2]):  # 출력 채널 반복
            for h in range(output_height):
                for w in range(output_width):
                    for i in range(filter_height):
                        for j in range(filter_width):
                            leaf_node = leaf_nodes_flat[leaf_index]  # 리프 노드 가져오기

                            # Pooling 레이어의 자식 노드를 연결
                            for ch in range(num_channels):  # Pooling 채널 반복
                                child_h = h * stride_height + i
                                child_w = w * stride_width + j

                                if child_h >= pool_output_height or child_w >= pool_output_width:
                                    continue

                                child_index = ch * pool_output_height * pool_output_width + child_h * pool_output_width + child_w
                                child_node = child_nodes[child_index]

                                # 리프 노드와 Pooling 루트 노드 연결
                                if child_node not in leaf_node.get_children():
                                    leaf_node.add_child(child_node)
                                    child_node.add_parent(leaf_node)

                            leaf_index += 1  # 다음 리프 노드로 이동

        current_layer.node_list = parent_nodes

        return parent_nodes