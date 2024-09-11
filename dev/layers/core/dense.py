import os
os.add_dll_directory("C:\\msys64\\mingw64\\bin")

from dev.layers.layer import Layer
from dev import activations
from dev.node.node import Node

from dev.backend.operaters import operations_matrix
from dev.backend.node import node

import numpy as np


class Dense(Layer, Node):
    # dense layer 에 필요한 내용이 뭐가 있을지, 추가될 수 있어용
    def __init__(self, units, activation=None, name=None, **kwargs):
        super().__init__(name)
        self.units = units
        self.output_shape = (units,)
        # 가중치 갱신이 가능한 layer
        self.trainable = True
        # 노드 리스트의 저장
        self.node_list = None

        # activations 오브젝트를 지정하네
        if activation is not None:
            self.activation = activations.get(activation)
        else:
            self.activation = activation
            
        self.weights = None
        self.bias = None

    def get_config(self):
        base_config = super().get_config()
        config = ({
            'class_name': self.__class__.__name__,
            'units': self.units,
            'activation': self.activation.__name__ if self.activation else None,
            'input_shape': self.input_shape,

        })
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    # 입력 차원에 따라 가중치와 편향을 초기화
    def build(self, input_shape):
        """
        해당 레이어의 가중치를 초기화
        input_shape 의 정보가 무조건 잇어야 함... flatten 에서 없을 경우 오류가 발생하는데
        이를 수정해야함

        Parameters:
        input_shape - 해당 레이어에 입력되는 데이터셋의 형태
        """
        # input_shape가 (784,)와 같은 경우라면, 실제 필요한 것은 input_shape[0]
        input_dim = input_shape[0]
        self.input_shape = input_shape

        # 가중치 생성
        self.weights = np.random.randn(input_dim, self.units)
        self.bias = np.random.rand()
        super().build()


    def call(self, inputs):
        """
        dense 층의 연산

        Parameters:
        inputs (배치 크기, 행, 열) 

        Returns:
        result (배치 단위 출력)

        """
        shape = inputs.shape

        n = shape[0]

        # 행렬 출력 결과의 형태에 맞게 node_list 를 재구성 해보자
        # node_list 를 2차원 형태로 재구성하지말고 이미 어떤 형태를 띄어야 하는지는
        # 알고 있으므로 1차원 리스트에 계속 이어서 붙여보자

        # 계산 그래프, 노드의 구성 때문에 아래와 같은 조건문들이 추가되었음...

        # 노드 리스트를 재구성, 행렬 곱이니까안
        # sefl.node_list 의 개수는 배치 데이터 * unit
        x, mul_mat_node_list = operations_matrix.matrix_multiply(inputs, self.weights)

        self.node_list = mul_mat_node_list

        # bias 가 None 이 아닌 경우 - 아직
        # 이거 수정해야 함 루트 노드와 리프 노드를 연결해야 함,
        # 이전에는 루트 노드끼리 연결되어 있었음
        if self.bias is not None:
            x, add_node_list = operations_matrix.matrix_add(x, np.tile(self.bias, x.shape))
            for i in range(len(add_node_list)):
                child_node = self.backpropagate(add_node_list[i])
                root_node = self.find_root(mul_mat_node_list[i])

                child_node[0].add_child(root_node)
                root_node.add_parent(child_node[0])
        
        if self.activation is not None:
            x, act_node_list = self.activation(x)

            for i in range(len(act_node_list)):
                child_node = self.backpropagate(act_node_list[i])
                root_node = self.find_root(mul_mat_node_list[i])

                child_node[0].add_child(root_node)
                root_node.add_parent(child_node[0])

        x = x.reshape(n, 1,-1)

        self.set_root_node()

        return x
    
    def set_root_node(self):
        root_node_list = []
        for node in self.node_list:
            root_node_list.append(self.find_root(node))

        self.node_list = root_node_list