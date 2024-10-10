import os
os.add_dll_directory("C:\\msys64\\mingw64\\bin")

from dev.layers.layer import Layer
from dev import activations
from dev.node.node import Node

from dev.backend.operaters import operations_matrix

import numpy as np


class Dense(Layer, Node):
    # dense layer 에 필요한 내용이 뭐가 있을지, 추가될 수 있어용
    def __init__(self, units, activation=None, name=None, **kwargs):
        super().__init__(name)
        self.units = units
        self.output_shape = (1,units)
        # 가중치 갱신이 가능한 layer
        self.trainable = True
        # 노드 리스트의 저장
        self.node_list = []

        self.mul_mat_node_list = []

        self.add_bias_node_list = []

        self.act_node_list  = []

        # activations 오브젝트를 지정
        if activation is not None:
            self.activation = activations.get(activation)
        else:
            self.activation = activation
            
        self.weights = None
        self.bias = None
        self.layer_name = "dense"

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
        
        input_dim = input_shape[1]
        self.input_shape = input_shape

        # 가중치 생성
        self.weights = np.random.randn(input_dim, self.units)
        self.bias = np.random.rand()

        super().build()


    def call(self, input_data):
        """
        dense 층의 연산

        Parameters:
        inputs (배치 크기, 행, 열) 

        Returns:
        result (배치 단위 출력)

        """
        root_node_list = self.node_list
        
        # 개별 데이터의 행렬 곱셈 수행
        x, mul_mat_node_list = operations_matrix.matrix_multiply(input_data, self.weights, self.mul_mat_node_list)

        root_node_list = mul_mat_node_list
        
        # 행렬 곱 노드 리스트
        self.mul_mat_node_list = mul_mat_node_list

        # bias 가 None 이 아닌 경우
        if self.bias is not None:
            bias_reshaped = np.tile(self.bias, (1, x.shape[1]))
            
            x, add_node_list = operations_matrix.matrix_add(x, bias_reshaped, self.add_bias_node_list)

            self.add_bias_node_list = add_node_list
            # add_node_list 노드들을 mul_mat_node_list에 연결
            # add_node_list 의 leaf_node 들과 연결해야 함
            for j in range(len(add_node_list)):
                leaf_node_list = self.find_child_node(add_node_list[j])
                root_node = root_node_list[j]
                
                for leaf_node in leaf_node_list:
                    leaf_node.add_child(root_node)
                    root_node.add_parent(leaf_node)

            # 최상위 노드 업데이트
            root_node_list = add_node_list

        if self.activation is not None:

            x, act_node_list = self.activation(x, self.act_node_list)

            self.act_node_list = act_node_list

            for j in range(len(act_node_list)):
                leaf_node_list = self.find_child_node(act_node_list[j])
                root_node = root_node_list[j]
                
                for leaf_node in leaf_node_list:
                    leaf_node.add_child(root_node)
                    root_node.add_parent(leaf_node)

            root_node_list = act_node_list

        self.node_list = root_node_list

        return x.reshape(1, -1)
    
    def set_root_node(self):
        root_node_list = []
        for node in self.node_list:
            root_node_list.append(self.find_root(node))

        self.node_list = root_node_list