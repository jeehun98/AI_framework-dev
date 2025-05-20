# dev/layers/dense.py

import numpy as np
from dev.layers.layer import Layer
from ..tests.test_setup import import_cuda_module

# ✅ CUDA 모듈 로드 (CUDA 활성화 함수 포함)
matrix_ops = import_cuda_module(
    module_name="operations_matrix_cuda",
    build_dir=r"C:\Users\owner\Desktop\AI_framework-dev\dev\backend\backend_ops\operaters\build\lib.win-amd64-cpython-312"
)

class Dense(Layer):
    def __init__(self, units, activation=None, name=None, initializer='he', **kwargs):
        super().__init__(name, **kwargs)
        self.units = units
        self.activation = activation  # CUDA에서 가져온 함수
        self.initializer = initializer
        self.input_shape = None
        self.output_shape = (1, units)

        self.weights = None
        self.bias = None

        # 역전파에 필요한 정보
        self.last_input = None
        self.last_z = None
        self.dW = None
        self.db = None

    def build(self, input_shape):
        print(f"[DEBUG] Dense.build() 진입 - 받은 input_shape: {input_shape}")
        self.input_shape = input_shape
        input_dim = input_shape[1]

        if self.initializer == 'ones':
            self.weights = np.ones((input_dim, self.units)).astype(np.float32)
        elif self.initializer == 'zeros':
            self.weights = np.zeros((input_dim, self.units)).astype(np.float32)
        elif self.initializer == 'he':
            stddev = np.sqrt(2. / input_dim)
            self.weights = (np.random.randn(input_dim, self.units) * stddev).astype(np.float32)
        elif self.initializer == 'xavier':
            stddev = np.sqrt(1. / input_dim)
            self.weights = (np.random.randn(input_dim, self.units) * stddev).astype(np.float32)
        else:
            raise ValueError(f"Unknown initializer: {self.initializer}")

        self.bias = np.zeros((1, self.units), dtype=np.float32)
        print(f"[DEBUG] Dense.build() 완료 - weights shape: {self.weights.shape}, bias shape: {self.bias.shape}")

    def call(self, input_data):
        input_data = np.atleast_2d(input_data).astype(np.float32)

        if self.weights is None:
            self.build(input_data.shape)

        self.last_input = input_data

        try:
            z = matrix_ops.matrix_multiply(input_data, self.weights)
            if isinstance(z, tuple):  # ✅ tuple일 경우 첫 요소 사용
                z = z[0]
        except Exception as e:
            print(f"[ERROR] CUDA matmul 실패. fallback to np.dot: {e}")
            z = np.dot(input_data, self.weights)

        # ✅ numpy float32 array 보장
        z = np.array(z, dtype=np.float32)

        # ✅ bias shape 맞추기
        if self.bias.shape[0] != z.shape[0]:
            bias_reshaped = np.tile(self.bias, (z.shape[0], 1)).astype(np.float32)
        else:
            bias_reshaped = self.bias

        z = z + bias_reshaped
        self.last_z = z.copy()

        if self.activation is not None:
            try:
                z = self.activation(z)
            except Exception as e:
                print(f"[ERROR] CUDA activation 실패. fallback to numpy: {e}")
                z = self._fallback_activation(z)

        return z


    def backward(self, grad_output):
        # ✅ 활성화 함수의 gradient 적용
        if self.activation is not None:
            grad_output = self._apply_activation_grad(grad_output, self.last_z)

        # ✅ 가중치 및 입력에 대한 gradient 계산
        self.dW = np.dot(self.last_input.T, grad_output)
        self.db = np.sum(grad_output, axis=0, keepdims=True)
        dx = np.dot(grad_output, self.weights.T)
        return dx

    def update(self, optimizer):
        self.weights, self.bias = optimizer.update(self.weights, self.dW, self.bias, self.db)

    def _apply_activation_grad(self, grad_output, z):
        act_name = self.activation.__name__ if hasattr(self.activation, '__name__') else str(self.activation)

        if act_name == "sigmoid":
            sig = 1 / (1 + np.exp(-z))
            return grad_output * sig * (1 - sig)
        elif act_name == "relu":
            return grad_output * (z > 0).astype(np.float32)
        elif act_name == "tanh":
            return grad_output * (1 - np.tanh(z) ** 2)
        else:
            return grad_output

    def _fallback_activation(self, z):
        act_name = self.activation.__name__ if hasattr(self.activation, '__name__') else str(self.activation)

        if act_name == "sigmoid":
            return 1 / (1 + np.exp(-z))
        elif act_name == "relu":
            return np.maximum(0, z)
        elif act_name == "tanh":
            return np.tanh(z)
        else:
            raise ValueError(f"지원하지 않는 activation: {act_name}")
