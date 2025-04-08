# dev/layers/core/dense_cuda.py

import os
import sys
import copy
import numpy as np

# ✅ 테스트 환경 및 CUDA 연산 모듈 설정
from ..tests.test_setup import setup_paths, import_cuda_module
setup_paths()
matrix_ops = import_cuda_module()

# ✅ 내부 모듈 import (상대 경로)
from ..layers.layer import Layer
from .. import activations
from ..node.node import Node
from ..graph_engine.core_graph import Cal_graph


class Dense(Layer):
    def __init__(self, units, activation=None, name=None, initializer='he', **kwargs):
        super().__init__(name, **kwargs)
        self.units = units
        self.output_shape = (1, units)
        self.trainable = True
        self.layer_name = "dense"

        self.cal_graph = Cal_graph()  # ✅ 계산 그래프 인스턴스
        self.activation = activations.get(activation) if activation else None
        self.initializer = initializer

        self.weights = None
        self.bias = None

    def initialize_weights(self, input_dim):
        if self.initializer == 'he':
            self.weights = np.random.randn(input_dim, self.units) * np.sqrt(2 / input_dim)
        elif self.initializer == 'xavier':
            self.weights = np.random.randn(input_dim, self.units) * np.sqrt(1 / input_dim)
        elif self.initializer == 'zeros':
            self.weights = np.zeros((input_dim, self.units))
        elif self.initializer == 'ones':
            self.weights = np.ones((input_dim, self.units))
        elif self.initializer == 'random_uniform':
            self.weights = np.random.uniform(-0.05, 0.05, (input_dim, self.units))
        else:
            raise ValueError(f"Unknown initializer: {self.initializer}")

        self.bias = np.zeros((1, self.units))

    def build(self, input_shape):
        if not isinstance(input_shape, (tuple, list)) or len(input_shape) < 2:
            raise ValueError("Invalid input shape. Expected a tuple with at least two dimensions.")

        input_dim = input_shape[1]
        self.input_shape = input_shape
        self.initialize_weights(input_dim)
        super().build()

    def call(self, input_data):
        if input_data.ndim != 2 or input_data.shape[1] != self.input_shape[1]:
            raise ValueError(f"Invalid input shape. Expected shape (batch_size, {self.input_shape[1]}), got {input_data.shape}.")

        batch_size, input_dim = input_data.shape
        output_dim = self.weights.shape[1]

        result, mul_nodes = matrix_ops.matrix_multiply(
            input_data.astype(np.float64),
            self.weights.astype(np.float64)
        )

        # 계산 그래프 구성
        mul_nodes = self.cal_graph.add_matrix_multiply_graph(
            input_data.tolist(),
            self.weights.tolist(),
            result.tolist()
        )

        if self.bias is not None:
            bias_reshaped = np.tile(self.bias, (batch_size, 1)).astype(np.float64)
            mul_result = result.copy()

            result, add_nodes = matrix_ops.matrix_add(result, bias_reshaped)

            add_nodes = self.cal_graph.add_matrix_add_graph(
                mul_result.tolist(),
                bias_reshaped.tolist(),
                result.tolist()
            )
            self.cal_graph.connect_graphs(add_nodes, mul_nodes)

        # ✅ 활성화 함수 적용
        if self.activation is not None:
            result, act_node_list = self.activation(result)

        return result

    def get_config(self):
        base_config = super().get_config()
        config = {
            'class_name': self.__class__.__name__,
            'units': self.units,
            'activation': self.activation.__name__ if self.activation else None,
            'initializer': self.initializer,
            'input_shape': self.input_shape,
            'trainable': self.trainable
        }
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def set_root_node(self):
        if not all(node.is_root() for node in self.node_list):
            self.node_list = [self.find_root(node) for node in self.node_list]