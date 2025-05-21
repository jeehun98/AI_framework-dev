import numpy as np
from dev.layers.layer import Layer
from ..tests.test_setup import import_cuda_module

# ✅ CUDA 모듈 로드
matrix_ops = import_cuda_module(
    module_name="operations_matrix_cuda",
    build_dir=r"C:\Users\owner\Desktop\AI_framework-dev\dev\backend\backend_ops\operaters\build\lib.win-amd64-cpython-312"
)

class Dense(Layer):
    def __init__(self, units, activation=None, name=None, initializer='he', **kwargs):
        super().__init__(name, **kwargs)
        self.units = units
        self.activation = activation
        self.initializer = initializer
        self.input_shape = None
        self.output_shape = (1, units)

        self.weights = None
        self.bias = None
        self.last_input = None
        self.last_z = None
        self.dW = None
        self.db = None

        # ✅ 활성화 이름 추출
        if callable(activation):
            self.activation_name = activation.__name__
        else:
            self.activation_name = str(activation)

    def build(self, input_shape):
        print(f"[DEBUG] Dense.build() 진입 - 받은 input_shape: {input_shape}")
        self.input_shape = input_shape
        input_dim = input_shape[1]

        if self.initializer == 'ones':
            self.weights = np.ones((input_dim, self.units), dtype=np.float32)
        elif self.initializer == 'zeros':
            self.weights = np.zeros((input_dim, self.units), dtype=np.float32)
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
            if isinstance(z, tuple):
                z = z[0]
        except Exception as e:
            print(f"[ERROR] CUDA matmul 실패. fallback to np.dot: {e}")
            z = np.dot(input_data, self.weights)

        # ✅ bias 추가
        if self.bias.shape[0] != z.shape[0]:
            bias_reshaped = np.tile(self.bias, (z.shape[0], 1)).astype(np.float32)
        else:
            bias_reshaped = self.bias

        z = z + bias_reshaped
        self.last_z = z.copy()

        # ✅ 활성화 함수 적용
        if self.activation is not None:
            try:
                z = self.activation(z)
            except Exception as e:
                print(f"[ERROR] CUDA activation 실패. fallback to numpy: {e}")
                z = self._fallback_activation(z)

        return z

    def backward(self, grad_output):
        if isinstance(grad_output, tuple):
            grad_output = grad_output[0]

        grad_output = grad_output.astype(np.float32)

        # ✅ 활성화 미분 적용
        if self.activation_name is not None and self.activation_name.lower() != "none":
            grad_output = self._apply_activation_grad(grad_output, self.last_z)

        # ✅ dW, db, dx 계산
        try:
            self.dW = matrix_ops.matrix_multiply(self.last_input.T, grad_output)
            if isinstance(self.dW, tuple):
                self.dW = self.dW[0]
        except Exception as e:
            print(f"[WARN] CUDA dW 실패, fallback: {e}")
            self.dW = np.dot(self.last_input.T, grad_output)

        self.db = np.sum(grad_output, axis=0, keepdims=True)

        try:
            dx = matrix_ops.matrix_multiply(grad_output, self.weights.T)
            if isinstance(dx, tuple):
                dx = dx[0]
        except Exception as e:
            print(f"[WARN] CUDA dx 실패, fallback: {e}")
            dx = np.dot(grad_output, self.weights.T)

        return dx

    def update(self, optimizer):
        optimizer.update(self.weights, self.dW, self.bias, self.db)

    def _apply_activation_grad(self, grad_output, z):
        if self.activation_name is None:
            return grad_output

        name = self.activation_name.lower()
        if name == "sigmoid":
            sig = 1 / (1 + np.exp(-z))
            return grad_output * sig * (1 - sig)
        elif name == "relu":
            return grad_output * (z > 0).astype(np.float32)
        elif name == "tanh":
            return grad_output * (1 - np.tanh(z) ** 2)
        else:
            raise NotImplementedError(f"[ERROR] '{self.activation_name}' 미분 미지원")

    def _fallback_activation(self, z):
        name = self.activation_name.lower()
        if name == "sigmoid":
            return 1 / (1 + np.exp(-z))
        elif name == "relu":
            return np.maximum(0, z)
        elif name == "tanh":
            return np.tanh(z)
        else:
            raise ValueError(f"지원하지 않는 activation: {self.activation_name}")
