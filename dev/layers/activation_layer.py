import cupy as cp
from dev.layers.layer import Layer
from dev.utils.load_cuda import load_activations_cuda

# ✅ 연산 코드 매핑
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

        # ✅ GraphCompiler 연산 이름 정의
        self.name = name or f"activation_{id(self)}"
        self.input_var = None
        self.output_var = f"{self.name}_out"
        self.input_idx = None
        self.output_idx = None

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

    def compute_output_shape(self, input_shape):
        return input_shape

    # ✅ GraphCompiler용 연산 정보 반환 (Activation Layer)
    def forward_matrix(self, input_name="input"):
        self.input_var = input_name

        return [{
            "input_idx": self.input_idx,     # None이면 GraphCompiler가 채움
            "param_idx": None,               # 매개변수 없음
            "output_idx": self.output_idx,
            "op_type": ACTIVATION_OP_TYPES[self.activation_name],
            "W": None,
            "b": None
        }]
