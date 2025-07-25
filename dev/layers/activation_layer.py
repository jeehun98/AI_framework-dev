import cupy as cp
from dev.layers.layer import Layer
from dev.utils.load_cuda import load_activations_cuda

import sys
sys.path.append("C:/Users/owner/Desktop/AI_framework-dev/dev/backend/graph_executor")
import graph_executor as ge  # Pybind11 모듈

# graph_executor에서 정의된 Shape 사용
Shape = ge.Shape

# 연산 코드 매핑
ACTIVATION_OP_TYPES = {
    'sigmoid': 2,
    'relu': 3,
    'tanh': 4
}

class Activation(Layer):
    def __init__(self, activation, name=None, use_backend_init=True, **kwargs):
        super().__init__(name=name, **kwargs)
        self.activation_name = activation.lower()
        self.trainable = False
        self.layer_name = "activation"
        self.last_z = None
        self.input_shape = None
        self.use_backend_init = use_backend_init

        self.name = name or f"activation_{id(self)}"
        self.output_var = f"{self.name}_out"
        self.activations_cuda = load_activations_cuda()

    def call(self, inputs):
        if not isinstance(inputs, cp.ndarray):
            inputs = cp.asarray(inputs, dtype=cp.float32)
        else:
            inputs = inputs.astype(cp.float32, copy=False)

        self.input_shape = inputs.shape
        self.last_z = inputs

        if cp.isnan(inputs).any() or cp.isinf(inputs).any():
            print("[WARNING] 입력에 NaN 또는 inf 포함됨")

        try:
            self.activations_cuda.apply_activation(inputs, self.activation_name)
        except Exception as e:
            raise RuntimeError(f"[ERROR] CUDA 활성화 실패: {e}")
        return inputs

    def backward(self, grad_output):
        if isinstance(grad_output, tuple):
            grad_output = grad_output[0]

        if not isinstance(grad_output, cp.ndarray):
            grad_output = cp.asarray(grad_output, dtype=cp.float32)
        else:
            grad_output = grad_output.astype(cp.float32, copy=False)

        try:
            self.activations_cuda.apply_activation_grad(self.last_z, grad_output, self.activation_name)
        except Exception as e:
            raise RuntimeError(f"[ERROR] CUDA backward 실패: {e}")
        return grad_output

    def compute_output_shape(self, input_shape):
        return input_shape

    def build(self, input_shape):
        self.input_shape = input_shape
        self.output_shape = input_shape
        super().build(input_shape)

    def to_e_matrix(self, input_id):
        output_id = f"{self.name}_out"

        extra = ge.OpExtraParams()

        if self.activation_name not in ACTIVATION_OP_TYPES:
            raise ValueError(f"[ERROR] Unsupported activation: {self.activation_name}")

        op_type = ACTIVATION_OP_TYPES[self.activation_name]

        e_block = [{
            "op_type": op_type,
            "input_id": input_id,
            "param_id": "",  # ❗️ Pybind11은 str 타입만 받음
            "output_id": output_id,
            "extra_params": extra
        }]

        # ✅ input_shape 검증
        if (
            not self.input_shape
            or not isinstance(self.input_shape, (tuple, list))
            or len(self.input_shape) != 2
            or not all(isinstance(v, int) for v in self.input_shape)
        ):
            raise ValueError(f"[ERROR] input_shape가 잘못되었습니다: {self.input_shape}")

        shape_map = {
            input_id: Shape(int(self.input_shape[0]), int(self.input_shape[1])),
            output_id: Shape(int(self.input_shape[0]), int(self.input_shape[1]))
        }

        return e_block, {}, {}, output_id, shape_map
