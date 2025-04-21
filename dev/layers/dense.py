import numpy as np

from dev.layers.layer import Layer
from dev import activations
from dev.backend.backend_ops.operaters import operations_matrix
from dev.graph_engine import graph_utils
from dev.graph_engine.core_ops import matrix_multiply_nodes, matrix_add_nodes
from dev.graph_engine.activations_graph import build_sigmoid_node, build_relu_node, build_tanh_node
from dev.graph_engine.graph_utils import connect_graphs

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
        self.root_node_list = []   # ✅ 출력 노드
        self.leaf_node_list = []   # ✅ 연결 대상 리프 노드
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

        # ✅ 행렬 곱 계산 그래프 구성
        mm_root, mm_leaf = matrix_multiply_nodes(
            input_data.tolist(), self.weights.tolist(), matmul_result.tolist()
        )
        result = matmul_result
        current_root = mm_root
        current_leaf = mm_leaf

        # ✅ bias 더하기
        if self.bias is not None:
            bias_reshaped = np.tile(self.bias, (input_data.shape[0], 1))
            try:
                result = matrix_ops.matrix_add(matmul_result, bias_reshaped)
                if isinstance(result, tuple):
                    result = result[0]
            except Exception:
                result = matmul_result + bias_reshaped

            add_root, add_leaf = matrix_add_nodes(
                matmul_result.tolist(), bias_reshaped.tolist(), result.tolist()
            )

            connect_graphs(current_root, add_leaf)
            current_root = add_root

        # ✅ activation 함수 처리
        if self.activation is not None:
            result = self.activation(result)

            builder_map = {
                "sigmoid": build_sigmoid_node,
                "relu": build_relu_node,
                "tanh": build_tanh_node,
            }
            act_name = self.activation.__name__
            act_builder = builder_map[act_name]

            act_root = []
            act_leaf = []

            for _ in range(result.size):
                root, leaf = act_builder()
                act_root.append(root)
                act_leaf.extend(leaf)

            connect_graphs(current_root, act_leaf)
            current_root = act_root


        # ✅ root/leaf 저장
        self.root_node_list = current_root
        self.leaf_node_list = current_leaf
        self.output_shape = result.shape


        return result

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
