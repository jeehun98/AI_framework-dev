# 백엔드에서는 기본적인 연산만 수행하도록 하고

# 계산 그래프의 구성은 파이썬 내에서 수행

import sys
import os

# 빌드된 모듈 경로 추가
build_path = os.path.abspath("dev/backend/operaters/build/lib.win-amd64-cpython-312")
if os.path.exists(build_path):
    sys.path.append(build_path)
else:
    raise FileNotFoundError(f"Build path does not exist: {build_path}")

# CUDA DLL 경로 추가
cuda_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin"
if os.path.exists(cuda_path):
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(cuda_path)
    else:
        os.environ["PATH"] = cuda_path + os.pathsep + os.environ["PATH"]
else:
    raise FileNotFoundError(f"CUDA path does not exist: {cuda_path}")

try:
    import matrix_ops
except ImportError as e:
    raise ImportError("Failed to import `matrix_ops` module. Ensure it is built and the path is correctly set.") from e


import numpy as np
from dev.layers.layer import Layer
from dev import activations
from dev.node.node import Node
from dev.backend.operaters import operations_matrix
from dev.cal_graph.cal_graph import Cal_graph

class Dense(Layer):
    def __init__(self, units, activation=None, name=None, initializer='he', **kwargs):
        super().__init__(name, **kwargs)
        self.units = units
        self.output_shape = (1, units)
        self.trainable = True
        self.layer_name = "dense"

        # 계산 그래프 인스턴스 생성
        # 상위 클래스에서 정의되어야 하는 것으로 수정 필요??!!
        self.cal_graph = Cal_graph()

        # 활성화 함수 설정
        self.activation = activations.get(activation) if activation else None

        # 가중치 및 편향 초기화 방식
        self.initializer = initializer
        self.weights = None
        self.bias = None

    def initialize_weights(self, input_dim):
        """
        가중치와 편향을 초기화합니다.
        추가된 초기화 방법: zeros, ones, random_uniform
        """
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
        """ 입력 크기에 맞게 가중치 초기화 """
        if not isinstance(input_shape, (tuple, list)) or len(input_shape) < 2:
            raise ValueError("Invalid input shape. Expected a tuple with at least two dimensions.")

        input_dim = input_shape[1]
        self.input_shape = input_shape

        # 가중치 초기화
        self.initialize_weights(input_dim)

        super().build()

    def call(self, input_data):
        """
        Dense 층의 연산을 수행합니다.

        연산 수행 과정을 수정하자.
        """
        print("call 실행 확인")

        if input_data.ndim != 2 or input_data.shape[1] != self.input_shape[1]:
            raise ValueError(f"Invalid input shape. Expected shape (batch_size, {self.input_shape[1]}), "
                            f"but got {input_data.shape}.")

        batch_size, input_dim = input_data.shape
        output_dim = self.weights.shape[1]

        # 행렬 곱셈 연산, 결과를 저장할 result 행렬을 미리 생성한다.
        result = np.zeros((batch_size, output_dim), dtype=np.float32)

        input_data = input_data.astype(np.float32)
        self.weights = self.weights.astype(np.float32)  

        # ✅ 백엔드 연산: 행렬 곱셈 (반환값 사용 X, result에 직접 저장됨)
        matrix_ops.matrix_mul(input_data, self.weights, result)

        # 계산 그래프 구성 : 행렬 곱셈
        mul_nodes = self.cal_graph.matrix_multiply(input_data.tolist(), self.weights.tolist(), result.tolist())
        
        print("계산 그래프 확인")
        self.cal_graph.print_graph()

        # 편향 추가
        if self.bias is not None:
            # 편향 값을 행렬로 변환
            bias_reshaped = np.tile(self.bias, (batch_size, 1))
            matrix_ops.matrix_add(result, bias_reshaped, result)

            # 계산 그래프 구성 : 편향 추가
            add_nodes = self.cal_graph.matrix_add(result.tolist(), bias_reshaped.tolist(), result.tolist())

            self.cal_graph.connect_graphs(add_nodes, mul_nodes)

        # 활성화 함수 적용
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
        """
        모든 노드의 루트를 설정합니다.
        루트 노드가 이미 설정된 경우 중복 작업을 피합니다.
        """
        if not all(node.is_root() for node in self.node_list):
            self.node_list = [self.find_root(node) for node in self.node_list]
