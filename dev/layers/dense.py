import numpy as np

from dev.layers.layer import Layer
from dev import activations
from dev.node.node import Node
from dev.backend.backend_ops.operaters import operations_matrix
from dev.graph_engine import graph_utils

import os
import sys

from ..tests.test_setup import import_cuda_module

matrix_ops = import_cuda_module(
    module_name="operations_matrix_cuda",
    build_dir=r"C:\Users\owner\Desktop\AI_framework-dev\dev\backend\backend_ops\operaters\build\lib.win-amd64-cpython-312"
)

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

    def call(self, input_data):
        input_data = np.atleast_2d(input_data).astype(np.float32)

        if self.weights is None or self.weights.shape[0] != input_data.shape[1]:
            print(f"[DEBUG] Dense.call()에서 자동 build 재시도: input_data.shape={input_data.shape}")
            self.build(input_shape=input_data.shape)

        try:
            matmul_result = matrix_ops.matrix_multiply(input_data, self.weights)
            if isinstance(matmul_result, tuple):
                matmul_result = matmul_result[0]
        except Exception as e:
            print(f"[ERROR] CUDA matmul 실패. fallback to np.dot: {e}")
            matmul_result = np.dot(input_data, self.weights)

        bias_reshaped = np.tile(self.bias, (input_data.shape[0], 1))
        try:
            added_result = matrix_ops.matrix_add(matmul_result, bias_reshaped)
            if isinstance(added_result, tuple):
                added_result = added_result[0]
        except Exception as e:
            added_result = matmul_result + bias_reshaped

        if self.activation is not None:
            activated = self.activation(added_result)
            self.output_shape = activated.shape
            self.build_graph(input_data, matmul_result, added_result, activated)
            return activated

        self.output_shape = added_result.shape
        self.build_graph(input_data, matmul_result, added_result, added_result)
        return added_result

    def build_graph(self, input_data, matmul_result, added_result, final_output):
        from dev.graph_engine.core_ops import matrix_multiply_nodes, matrix_add_nodes
        from dev.graph_engine.activations_graph import build_sigmoid_node, build_relu_node, build_tanh_node

        input_data = np.atleast_2d(input_data).astype(np.float32)
        matmul_nodes = matrix_multiply_nodes(input_data.tolist(), self.weights.tolist(), matmul_result.tolist())

        bias_reshaped = np.tile(self.bias, (input_data.shape[0], 1))
        bias_add_nodes = matrix_add_nodes(matmul_result.tolist(), bias_reshaped.tolist(), added_result.tolist())
        bias_add_nodes = graph_utils.connect_graphs(bias_add_nodes, matmul_nodes)

        if self.activation is not None:
            builder_map = {
                "sigmoid": build_sigmoid_node,
                "relu": build_relu_node,
                "tanh": build_tanh_node,
            }
            act_name = self.activation.__name__
            act_builder = builder_map[act_name]
            act_nodes = [act_builder() for _ in range(final_output.size)]

            for node, parent in zip(act_nodes, bias_add_nodes):
                node.add_parent(parent)

            self.node_list = act_nodes
        else:
            self.node_list = bias_add_nodes

    def build(self, input_shape):
        print(f"[DEBUG] Dense.build() 진입 - 받은 input_shape: {input_shape}")
        self.input_shape = input_shape
        input_dim = input_shape[1]

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

        self.bias = np.zeros((1, self.units))
        print(f"[DEBUG] Dense.build() 완료 - weights shape: {self.weights.shape}, bias shape: {self.bias.shape}")