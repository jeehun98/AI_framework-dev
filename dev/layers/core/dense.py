import os
os.add_dll_directory("C:\\msys64\\mingw64\\bin")

from dev.layers.layer import Layer
from dev import activations

from dev.backend.operaters import operations_matrix

import numpy as np


class Dense(Layer):
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
        x, self.node_list = operations_matrix.matrix_multiply(inputs, self.weights)

        # bias 가 None 이 아닌 경우 - 아직
        if self.bias is not None:
            x, add_node_list = operations_matrix.matrix_add(x, np.tile(self.bias, x.shape))
            for i in range(len(self.node_list)):
                add_node_list[i].add_child(self.node_list[i])
                self.node_list[i].add_parent(add_node_list[i])

                
        
        # bias 가 None 이고, act 이 None 이 아닌 - 완료
        elif self.activation is not None:
            x, act_node_list = self.activation(x)
            
            for i in range(len(add_node_list)):
                act_node_list[i].add_child(add_node_list[i])
                add_node_list[i].add_parent(act_node_list[i])
            
            x = x.reshape(n, 1,-1)

            self.node_list = act_node_list

            return x

        # 둘 다 None 이 아닐 경우 - 완료
        if self.activation is not None:
            x, act_node_list = self.activation(x)
            for i in range(len(add_node_list)):
                act_node_list[i].add_child(self.node_list[i])
                self.node_list[i].add_parent(act_node_list[i])
            
            x = x.reshape(n, 1, -1)

            self.node_list = act_node_list
            #print("둘 다 아님")
            return x

        # bias 만 None 이 아닌 경우 - 완료
        elif self.activation is None and self.bias is not None:
            x = x.reshape(n, 1, -1)

            self.node_list = add_node_list
            #print("bias 만 아님")
            return x
        
        # 둘 다 None 인 경우 
        x = x.reshape(n, 1, -1)
        # self.node_list 는 위에 지정되어 있음

        return x

    """
    keras 코드
        def call(self, inputs, training=None):
        x = ops.matmul(inputs, self.kernel)
        if self.bias is not None:
            x = ops.add(x, self.bias)
        if self.activation is not None:
            x = self.activation(x)
        return x
    
    이전 call 코드
    x = operations_matrix.matrix_multiply(inputs, self.weights)
        if self.bias is not None:
            x = operations_matrix.matrix_add(x, np.tile(self.bias, x.shape))
        if self.activation is not None:
            x = self.activation(x)
            x = np.reshape(x,(-1,self.units))
        return x
    """


        