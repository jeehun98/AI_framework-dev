import numpy as np

from dev.layers.layer import Layer
from dev import activations
from dev.node.node import Node
from dev.backend.backend_ops.operaters import operations_matrix

import os
import sys

# CUDA 모듈 경로도 명시적으로 등록
sys.path.insert(0, os.path.abspath("C:/Users/owner/Desktop/AI_framework-dev/dev/backend/backend_ops/operaters"))

# AI_framework-dev를 sys.path에 강제로 추가
sys.path.insert(0, os.path.abspath("C:/Users/owner/Desktop/AI_framework-dev"))

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
        if input_data.ndim != 2 or input_data.shape[1] != self.input_shape[1]:
            raise ValueError(f"Invalid input shape. Expected shape (batch_size, {self.input_shape[1]}), "
                             f"but got {input_data.shape}.")

        # ✅ 1. CUDA 행렬곱
        matmul_result = matrix_ops.matrix_multiply(input_data, self.weights)

        # ✅ 2. 계산 그래프에 matmul 추가
        self.node_list = self.graph_builder.add_matrix_multiply_graph(
            input_data.tolist(), self.weights.tolist(), matmul_result.tolist()
        )

        # ✅ 3. Bias add (CUDA + graph)
        if self.bias is not None:
            bias_reshaped = np.tile(self.bias, (input_data.shape[0], 1))
            added_result = matrix_ops.matrix_add(matmul_result, bias_reshaped)

            bias_add_nodes = self.graph_builder.add_matrix_add_graph(
                matmul_result.tolist(), bias_reshaped.tolist(), added_result.tolist()
            )

            # ✅ 4. 연결: bias add → matmul
            self.graph_builder.connect_graphs(bias_add_nodes, self.node_list)
            self.node_list = bias_add_nodes  # 업데이트

        # ✅ 5. Activation 함수 (예: sigmoid) 적용
        if self.activation is not None:
            activated = self.activation(added_result)  # ⚠️ 여기에도 CUDA 사용 가능

            act_nodes = self.graph_builder.add_activation_graph(
                added_result.tolist(), activated.tolist(), self.activation.__name__
            )

            self.graph_builder.connect_graphs(act_nodes, self.node_list)
            self.node_list = act_nodes

            return activated

        return added_result
