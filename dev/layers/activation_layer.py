import sys, os
import cupy as cp
import numpy as np
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
    임의 차원 입력 지원:
      - elementwise 계열: (B, -1) 가상 2D로 매핑하여 e-블록 생성 (실제 텐서 모양은 유지)
      - softmax:
          * 2D: (B, F)에서 axis 사용(기본 1)
          * 3D 이상(NHWC 가정): (B*H*W..., C) 로 가상 2D → 채널축(C) softmax
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
        if input_shape is None:
            raise ValueError("[Activation] build: input_shape is None")
        # 활성화는 모양을 바꾸지 않음
        self.input_shape = tuple(map(int, input_shape))
        self.output_shape = self.input_shape
        self.built = True

    def call(self, inputs):
        """
        주로 그래프 실행기(run_graph)가 실행하므로 이 경로는 테스트/호환용.
        """
        x = cp.asarray(inputs, dtype=cp.float32)
        self.last_z = x
        if self.use_backend_init and self.activations_cuda is not None:
            try:
                self.activations_cuda.apply_activation(x, self.activation_name)
            except Exception:
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
        if input_shape is None:
            raise ValueError("[Activation] compute_output_shape: input_shape is None")
        return tuple(map(int, input_shape))

    # ---------- helpers: 가상 2D 셰이프 계산 ----------
    @staticmethod
    def _virtual_2d_for_elementwise(shape_tuple):
        """
        elementwise 계열 활성화용 (rows=B, cols=∏rest) 반환
        """
        if len(shape_tuple) == 0:
            return 1, 1
        if len(shape_tuple) == 1:
            return 1, int(shape_tuple[0])
        B = int(shape_tuple[0])
        F = int(np.prod(shape_tuple[1:])) if len(shape_tuple) > 1 else 1
        return B, F

    @staticmethod
    def _virtual_2d_for_softmax(shape_tuple):
        """
        softmax 전용 가상 2D:
          - 2D: (B,F) 그대로
          - 3D 이상(NHWC 가정): (B*H*W..., C)  (C=마지막 축)
        """
        if len(shape_tuple) == 2:
            B, F = map(int, shape_tuple)
            return ("2d", B, F, None, None, F)  # tag, rows, cols, H, W, C
        # 3D 이상
        B = int(shape_tuple[0])
        C = int(shape_tuple[-1])
        spatial = int(np.prod(shape_tuple[1:-1])) if len(shape_tuple) > 2 else 1
        rows = int(B * spatial)
        cols = int(max(C, 1))
        if len(shape_tuple) == 4:
            H, W = int(shape_tuple[1]), int(shape_tuple[2])
        else:
            H, W = spatial, 1  # 힌트용(엔진이 필요 없다면 무시)
        return ("nd", rows, cols, H, W, C)

    @staticmethod
    def _virtual_2d_for_elementwise(shape_tuple):
        """
        Elementwise 계열 활성화 가상 2D:
        - 4D(NHWC): (rows=C, cols=H*W)  ← Conv2D와 동일 정렬
        - 3D(NWC 등): (rows=C, cols=H*W) 가정
        - 2D(B, F): (rows=F, cols=1)로 두지 않고, (rows=1, cols=F)보다
                     Conv 규칙과 맞추기 위해 (rows=F, cols=1) 대신 (rows=1, cols=F)도 가능하나,
                     아래 일관성을 위해 (rows=F, cols=rest) 대신 'rows=F, cols=?'가 중요.
        - 1D(F,): (1,F)
        """
        n = len(shape_tuple)
        if n == 4:
            B, H, W, C = map(int, shape_tuple)
            return int(C), int(H * W)      # ✅ Conv2D와 동일 정렬
        if n == 3:
            H, W, C = map(int, shape_tuple)
            return int(C), int(H * W)      # ✅ 동일 정렬
        if n == 2:
            B, F = map(int, shape_tuple)
            return int(F), 1               # ✅ 행=특징수, 열=1 (배치는 엔진이 곱함)
        if n == 1:
            F = int(shape_tuple[0])
            return int(F), 1
        return 1, int(np.prod(shape_tuple))  # 최후방어

    @staticmethod
    def _virtual_2d_for_softmax(shape_tuple):
        """
        Softmax만 4D에서 (B*H*W, C) 유지 (채널축 softmax를 axis=1로 처리).
        2D는 (B,F) 그대로 axis=1.
        """
        if len(shape_tuple) == 2:
            B, F = map(int, shape_tuple)
            return ("2d", B, F, None, None, F)
        # 3D 이상: (B*H*W, C)
        B = int(shape_tuple[0])
        C = int(shape_tuple[-1])
        spatial = int(np.prod(shape_tuple[1:-1])) if len(shape_tuple) > 2 else 1
        rows = int(B * spatial)
        cols = int(max(C, 1))
        return ("nd", rows, cols, None, None, C)



    def to_e_matrix(self, input_id):
        if self.input_shape is None:
            raise ValueError("[Activation] input_shape is None. Did you forget to call build()?")

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
            # axis는 2D에서만 의미, ND에서는 채널축(C)로 강제
            extra.axis = int(self.axis)

        weights = {}
        biases = {}

        # ---------- Softmax 별도 처리 ----------
        if self.op_type == ge.OpType.SOFTMAX:
            tag, rows, cols, H, W, C = self._virtual_2d_for_softmax(self.input_shape)
            shape_map = {
                input_id: Shape(int(rows), int(cols)),
                output_id: Shape(int(rows), int(cols)),
            }
            extra.axis = 1  # 클래스 축
            e_block = [{
                "op_type": int(self.op_type),
                "input_id": input_id,
                "param_id": "",
                "output_id": output_id,
                "extra_params": extra
            }]
            return e_block, weights, biases, output_id, shape_map

        # ✅ elementwise: Conv2D와 정렬된 2D로
        rows, cols = self._virtual_2d_for_elementwise(self.input_shape)
        shape_map = {
            input_id:  Shape(int(rows), int(cols)),
            output_id: Shape(int(rows), int(cols)),
        }
        e_block = [{
            "op_type": int(self.op_type),
            "input_id": input_id,
            "param_id": "",
            "output_id": output_id,
            "extra_params": extra
        }]
        return e_block, weights, biases, output_id, shape_map