import numpy as np

from dev.layers.layer import Layer
from dev import activations
from dev.node.node import Node
from dev.backend.backend_ops.operaters import operations_matrix
from dev.graph_engine.core_graph import Cal_graph

import os
import sys

from ..tests.test_setup import import_cuda_module
matrix_ops = import_cuda_module()


class Dense(Layer):
    def __init__(self, units, activation=None, name=None, initializer='he', **kwargs):
        super().__init__(name, **kwargs)
        self.units = units
        self.output_shape = (1, units)
        self.trainable = True
        self.node_list = []
        self.layer_name = "dense"
        self.activation = activations.get(activation) if activation else None
        self.initializer = initializer
        self.weights = None
        self.bias = None
        self.graph_builder = Cal_graph()  # ✅ 계산 그래프 객체 생성

    def call(self, input_data):
        print(f"[DEBUG] Dense.call() 진입 - input shape: {input_data.shape}")

        if self.weights is None or self.bias is None:
            raise RuntimeError("Dense layer가 build되지 않았습니다. weights 또는 bias가 None입니다.")

        if input_data.ndim != 2 or input_data.shape[1] != self.input_shape[1]:
            raise ValueError(f"[ERROR] 입력 shape 불일치. 예상: (batch_size, {self.input_shape[1]}), "
                             f"입력: {input_data.shape}")

        if input_data.shape[1] != self.weights.shape[0]:
            raise ValueError(f"[ERROR] 행렬곱 불가능: 입력 차원 {input_data.shape[1]} ≠ 가중치 차원 {self.weights.shape[0]}")

        # ✅ 1. CUDA 행렬곱
        try:
            matmul_result = matrix_ops.matrix_multiply(input_data, self.weights)

            # CUDA 연산 결과가 tuple로 오는 경우 추출
            if isinstance(matmul_result, tuple):
                matmul_result = matmul_result[0]

        except Exception as e:
            print(f"[ERROR] CUDA 행렬곱 실패. fallback to np.dot. 오류: {e}")
            matmul_result = np.dot(input_data, self.weights)

        print(f"[DEBUG] matmul 완료 - result shape: {matmul_result.shape}")

        # ✅ 2. 계산 그래프에 matmul 추가
        self.node_list = self.graph_builder.add_matrix_multiply_graph(
            input_data.tolist(), self.weights.tolist(), matmul_result.tolist()
        )

        # ✅ 3. Bias add (CUDA + graph)
        # ✅ Bias add (CUDA + graph)
        added_result = matmul_result
        if self.bias is not None:
            bias_reshaped = np.tile(self.bias, (input_data.shape[0], 1))

            try:
                added_result = matrix_ops.matrix_add(matmul_result, bias_reshaped)
                if isinstance(added_result, tuple):
                    added_result = added_result[0]
            except Exception as e:
                print(f"[ERROR] CUDA bias add 실패. fallback to numpy add. 오류: {e}")
                added_result = matmul_result + bias_reshaped

            bias_add_nodes = self.graph_builder.add_matrix_add_graph(
                matmul_result.tolist(), bias_reshaped.tolist(), added_result.tolist()
            )

            self.graph_builder.connect_graphs(bias_add_nodes, self.node_list)
            self.node_list = bias_add_nodes
            

        # ✅ 4. Activation 함수 (예: sigmoid) 적용
        if self.activation is not None:
            activated = self.activation(added_result)  # 활성화 함수 실행 (CUDA or numpy)

            act_nodes = self.graph_builder.add_activation_graph(
                added_result.tolist(), activated.tolist(), self.activation.__name__
            )

            self.graph_builder.connect_graphs(act_nodes, self.node_list)
            self.node_list = act_nodes

            print(f"[DEBUG] activation 완료 - shape: {activated.shape}")
            return activated

        print(f"[DEBUG] Dense.call() 완료 - 반환 shape: {added_result.shape}")
        return added_result
    
    def build(self, input_shape):
        """
        입력 형태에 따라 가중치 및 편향 초기화
        Parameters:
            input_shape (tuple): (batch_size, input_dim)
        """
        self.input_shape = input_shape
        input_dim = input_shape[1]

        # ✅ 가중치 초기화
        if self.initializer == 'ones':
            self.weights = np.ones((input_dim, self.units))
        elif self.initializer == 'zeros':
            self.weights = np.zeros((input_dim, self.units))
        elif self.initializer == 'he':
            stddev = np.sqrt(2. / input_dim)
            self.weights = np.random.randn(input_dim, self.units) * stddev
        elif self.initializer == 'xavier':
            stddev = np.sqrt(1. / input_dim)
            self.weights = np.random.randn(input_dim, self.units) * stddev
        else:
            raise ValueError(f"Unknown initializer: {self.initializer}")

        # ✅ bias 초기화 (기본 zeros)
        self.bias = np.zeros((1, self.units))

        print(f"[DEBUG] Dense.build() 완료 - weights shape: {self.weights.shape}, bias shape: {self.bias.shape}")
