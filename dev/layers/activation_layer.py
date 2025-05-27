import cupy as cp
from dev.layers.layer import Layer
from dev.utils.load_cuda import load_activations_cuda  # CUDA Pybind11 모듈 로더

class Activation(Layer):
    def __init__(self, activation, **kwargs):
        super().__init__(**kwargs)
        self.activation_name = activation.lower()
        self.trainable = False
        self.layer_name = "activation"
        self.last_z = None
        self.input_shape = None

        # ✅ CUDA 모듈 로딩
        self.activations_cuda = load_activations_cuda()

    def call(self, inputs):
        # ✅ CuPy가 아니면 변환
        if not isinstance(inputs, cp.ndarray):
            inputs = cp.asarray(inputs, dtype=cp.float32)
        else:
            inputs = inputs.astype(cp.float32, copy=False)

        self.input_shape = inputs.shape
        self.last_z = inputs  # CuPy로 유지

        # ✅ 수치 안정성 디버깅 (선택)
        if cp.isnan(inputs).any() or cp.isinf(inputs).any():
            print("[WARNING] 입력에 NaN 또는 inf 포함됨")

        try:
            self.activations_cuda.apply_activation(inputs, self.activation_name)
        except Exception as e:
            raise RuntimeError(f"[ERROR] CUDA 활성화 실패: {e}")
        return inputs  # in-place 연산

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
        return grad_output  # in-place 수정됨

    def compute_output_shape(self, input_shape):
        return input_shape

    def build(self, input_shape):
        self.input_shape = input_shape
        self.output_shape = input_shape
        super().build(input_shape)
