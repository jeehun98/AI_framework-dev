import sys, os
import cupy as cp
from dev.layers.layer import Layer

# (선택) 예전 CUDA 테스트용 모듈이 필요하면 로드. 사용 안하면 넘어감.
try:
    from dev.utils.load_cuda import load_activations_cuda
except Exception:
    load_activations_cuda = None

# Pybind11 graph_executor
sys.path.append("C:/Users/owner/Desktop/AI_framework-dev/dev/backend/graph_executor")
import graph_executor as ge

Shape = ge.Shape
OpExtraParams = ge.OpExtraParams

# ✅ 지원 활성화 이름 -> ge.OpType 매핑 (숫자 하드코딩 금지)
NAME2OP = {
    "relu":       ge.OpType.RELU,
    "sigmoid":    ge.OpType.SIGMOID,
    "tanh":       ge.OpType.TANH,
    "leaky_relu": ge.OpType.LEAKY_RELU,
    "elu":        ge.OpType.ELU,
    "gelu":       ge.OpType.GELU,
    "silu":       ge.OpType.SILU,
    "softmax":    ge.OpType.SOFTMAX,
}

class Activation(Layer):
    """
    새 활성화(LeakyReLU, ELU, GELU, SiLU, Softmax) 지원.
    - alpha: LeakyReLU/ELU용 기울기/계수 (default 0.01 / 1.0)
    - gelu_tanh: GELU tanh 근사 사용 여부 (1=True)
    - temperature: Softmax 온도 (default 1.0)
    - axis: Softmax 축 (현재 런타임은 (batch, features) 2D에서 features축=1 가정)
    """
    def __init__(
        self,
        activation: str,
        name: str = None,
        use_backend_init: bool = False,   # 예전 activations_cuda 경로 쓰려면 True
        alpha: float = None,              # Leaky/ELU
        gelu_tanh: int = 1,               # 1: tanh 근사
        temperature: float = 1.0,         # softmax
        axis: int = 1,                    # softmax 축 (2D에서 1)
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.activation_name = activation.lower()
        if self.activation_name not in NAME2OP:
            raise ValueError(f"[Activation] Unsupported activation: {self.activation_name}")

        self.op_type = NAME2OP[self.activation_name]
        self.trainable = False
        self.layer_name = "activation"
        self.input_shape = None
        self.output_var = (name or f"activation_{id(self)}") + "_out"
        self.name = name or f"activation_{id(self)}"

        # 하이퍼파라미터 저장
        self.alpha = alpha
        self.gelu_tanh = int(bool(gelu_tanh))
        self.temperature = float(temperature)
        self.axis = int(axis)

        # (옵션) 레거시 커널 경로
        self.use_backend_init = use_backend_init
        self.last_z = None
        self.activations_cuda = None
        if self.use_backend_init and load_activations_cuda is not None:
            try:
                self.activations_cuda = load_activations_cuda()
            except Exception:
                self.activations_cuda = None  # 없으면 무시

    def __call__(self, x):
        if not self.built:
            self.build(x.shape)
        return self.call(x)

    def build(self, input_shape):
        # 내부 그래프는 (batch, features) 2D를 가정하므로, 필요한 경우 상위에서 Flatten
        self.input_shape = input_shape
        self.output_shape = self.compute_output_shape(input_shape)
        self.built = True

    def call(self, inputs):
        """
        주로 그래프 실행기(run_graph)가 실행하므로 이 경로는 테스트/호환용.
        """
        x = cp.asarray(inputs, dtype=cp.float32)
        self.last_z = x
        if self.use_backend_init and self.activations_cuda is not None:
            try:
                # 이름만으로 호출 가능한 경우에 한해 사용
                self.activations_cuda.apply_activation(x, self.activation_name)
            except Exception:
                # 그래프 실행에서 처리되므로 실패해도 치명적 아님
                pass
        return x

    def backward(self, grad_output):
        """
        주로 그래프 실행기에서 처리. 테스트/호환용.
        """
        if isinstance(grad_output, tuple):
            grad_output = grad_output[0]
        g = cp.asarray(grad_output, dtype=cp.float32)
        if self.use_backend_init and self.activations_cuda is not None and self.last_z is not None:
            try:
                self.activations_cuda.apply_activation_grad(self.last_z, g, self.activation_name)
            except Exception:
                pass
        return g

    def compute_output_shape(self, input_shape):
        return input_shape

    def to_e_matrix(self, input_id):
        """
        그래프 노드 생성:
        - output_id: self.output_var
        - extra_params: alpha/gelu_tanh/temperature/axis 등 세팅
        - shape_map: (batch, features) 2D Shape 등록
        """
        if not self.input_shape or len(self.input_shape) != 2:
            raise ValueError(f"[Activation] input_shape must be 2D (batch, features), got {self.input_shape}")

        output_id = self.output_var

        # ✅ OpExtraParams 채우기
        extra = OpExtraParams()
        # LeakyReLU/ELU 기본값
        if self.activation_name == "leaky_relu":
            extra.alpha = self.alpha if (self.alpha is not None) else 0.01
        elif self.activation_name == "elu":
            extra.alpha = self.alpha if (self.alpha is not None) else 1.0
        # GELU
        if self.activation_name == "gelu":
            extra.gelu_tanh = self.gelu_tanh  # 1(default): tanh 근사
        # Softmax
        if self.activation_name == "softmax":
            extra.temperature = self.temperature if self.temperature > 0 else 1.0
            extra.axis = self.axis  # 현재 런타임은 axis=1 가정

        e_block = [{
            "op_type": int(self.op_type),
            "input_id": input_id,
            "param_id": "",                # bias 없음
            "output_id": output_id,
            "extra_params": extra
        }]

        # Shape 등록 (host측 run_graph가 rows, cols로 사용)
        b, f = map(int, self.input_shape)
        shape_map = {
            input_id: Shape(b, f),
            output_id: Shape(b, f)
        }
        return e_block, {}, {}, output_id, shape_map
