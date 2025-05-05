import numpy as np

from ..tests.test_setup import import_cuda_module
from dev.backend.backend_ops.activations import activations_cuda

matrix_ops = import_cuda_module(
    module_name="operations_matrix_cuda",
    build_dir=r"C:\Users\owner\Desktop\AI_framework-dev\dev\backend\backend_ops\operaters\build\lib.win-amd64-cpython-312"
)

class DenseMat:
    def __init__(self, units, activation=None, initializer='he', input_dim=None):
        self.input_dim = input_dim  # 나중에 build 단계에서 설정 가능
        self.output_dim = units
        self.activation = activation  # 문자열로 저장: 'relu', 'sigmoid', etc.
        self.initializer = initializer
        self.weights = None
        self.bias = None

    def build(self, input_dim):
        self.input_dim = input_dim
        self.weights = self._initialize_weights(input_dim, self.output_dim)
        self.bias = np.zeros((1, self.output_dim), dtype=np.float32)

    def _initialize_weights(self, input_dim, output_dim):
        if self.initializer == 'ones':
            return np.ones((input_dim, output_dim), dtype=np.float32)
        elif self.initializer == 'zeros':
            return np.zeros((input_dim, output_dim), dtype=np.float32)
        elif self.initializer == 'he':
            stddev = np.sqrt(2. / input_dim)
            return (np.random.randn(input_dim, output_dim) * stddev).astype(np.float32)
        elif self.initializer == 'xavier':
            stddev = np.sqrt(1. / input_dim)
            return (np.random.randn(input_dim, output_dim) * stddev).astype(np.float32)
        else:
            raise ValueError(f"Unknown initializer: {self.initializer}")

    def call(self, input_data):
        input_data = np.atleast_2d(input_data).astype(np.float32)

        if self.input_dim is None:
            raise ValueError("input_dim is not set. Please call build(input_dim) first.")

        if input_data.shape[1] != self.input_dim:
            raise ValueError(f"Input shape mismatch: expected {self.input_dim}, got {input_data.shape[1]}")

        # ✅ CUDA 행렬 곱
        try:
            z = matrix_ops.matrix_multiply(input_data, self.weights)
            if isinstance(z, tuple):
                z = z[0]
        except Exception as e:
            print(f"[WARN] CUDA matrix_multiply failed: {e}, falling back to numpy")
            z = np.dot(input_data, self.weights)

        # ✅ bias 덧셈
        bias_tiled = np.tile(self.bias, (input_data.shape[0], 1))
        try:
            z = matrix_ops.matrix_add(z, bias_tiled)
            if isinstance(z, tuple):
                z = z[0]
        except Exception:
            z = z + bias_tiled

        print("[DEBUG] activation 호출 직전: shape =", z.shape, ", activation =", self.activation)

        # ✅ CUDA 기반 활성화 함수 적용
        if self.activation:
            try:
                activations_cuda.apply_activation(z, self.activation)  # in-place 적용
                print("[DEBUG] activation 호출 결과:", z)
            except Exception as e:
                print("[ERROR] activation 함수 호출 실패:", e)
                raise

        return z
