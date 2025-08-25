# dev/layers/rnn.py
import cupy as cp
from dev.layers.layer import Layer

import sys
sys.path.append("C:/Users/owner/Desktop/AI_framework-dev/dev/backend/graph_executor")
import graph_executor as ge  # Pybind11 모듈

Shape = ge.Shape
OpExtraParams = ge.OpExtraParams


def _act_name(activation):
    if activation is None:
        return None
    if isinstance(activation, str):
        return activation.lower()
    # callables like cp.tanh
    return getattr(activation, "__name__", str(activation)).lower()


def _activation_op_type(name: str | None) -> int | None:
    if not name or name == "linear":
        return None
    name = name.lower()
    amap = {
        "relu": int(ge.OpType.RELU),
        "sigmoid": int(ge.OpType.SIGMOID),
        "tanh": int(ge.OpType.TANH),
    }
    if name not in amap:
        raise ValueError(f"[RNN] Unsupported activation: {name}")
    return amap[name]


def _init_kernel(shape, kind="xavier"):
    if kind == "zeros":
        return cp.zeros(shape, dtype=cp.float32)
    if kind == "ones":
        return cp.ones(shape, dtype=cp.float32)
    if kind == "uniform":
        limit = 0.05
        return cp.random.uniform(-limit, limit, shape).astype(cp.float32)
    if kind == "normal":
        return cp.random.normal(0.0, 0.05, shape).astype(cp.float32)
    if kind == "xavier":
        fan_in, fan_out = shape[0], shape[1]
        limit = cp.sqrt(6.0 / (fan_in + fan_out))
        return cp.random.uniform(-limit, limit, shape).astype(cp.float32)
    if kind == "he":
        fan_in = shape[0]
        std = cp.sqrt(2.0 / fan_in)
        return cp.random.normal(0.0, std, shape).astype(cp.float32)
    if kind == "orthogonal":
        a = cp.random.normal(0, 1, shape).astype(cp.float32)
        q, r = cp.linalg.qr(a)
        d = cp.diag(r)
        ph = cp.sign(d + (d == 0))
        q = q * ph
        return q.astype(cp.float32)[:shape[0], :shape[1]]
    if kind == "small_uniform":
        return cp.random.uniform(-1e-3, 1e-3, shape).astype(cp.float32)
    raise ValueError(f"[RNN] Unknown initializer: {kind}")


def _init_bias(shape, kind="zeros"):
    if kind == "zeros":
        return cp.zeros(shape, dtype=cp.float32)
    if kind == "ones":
        return cp.ones(shape, dtype=cp.float32)
    if kind == "small_uniform":
        return cp.random.uniform(-1e-3, 1e-3, shape).astype(cp.float32)
    return _init_kernel(shape, kind)


class RNN(Layer):
    """
    Primitive-only RNN (Elman). GE(GraphExecutor)로만 실행할 때 사용.
    - forward/backward는 미구현 (CUDA backend가 수행)
    - to_e_matrix()에서 시간축 언롤을 primitive로 생성
    입력:  (B, T, D)
    출력:  return_sequences=False -> (B, U)
           return_sequences=True  -> (B, T, U)  (엔진에 CONCAT_TIME 필요)
    """

    def __init__(
        self,
        units,
        activation="tanh",
        return_sequences=False,
        input_shape=None,
        name=None,
        kernel_initializer="xavier",        # Wx
        recurrent_initializer="orthogonal", # Wh
        bias_initializer="zeros",
        use_bias=True,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.layer_name = "rnn"
        self.units = int(units)
        self.return_sequences = bool(return_sequences)
        self.use_bias = bool(use_bias)

        self.activation_name = _act_name(activation)

        self.input_shape = input_shape  # (B,T,D) 기대
        self.output_shape = None

        self.kernel_initializer = kernel_initializer
        self.recurrent_initializer = recurrent_initializer
        self.bias_initializer = bias_initializer

        # Parameters
        self.Wx = None  # (D,U)
        self.Wh = None  # (U,U)
        self.b  = None  # (1,U) if use_bias

        # GraphCompiler identifiers
        self.name = name or f"rnn_{id(self)}"
        self.wx_var = f"{self.name}_Wx"
        self.wh_var = f"{self.name}_Wh"
        self.b_var  = f"{self.name}_b"
        self.out_var = f"{self.name}_out"

    # ------------------------------------------------------------------ build
    def build(self, input_shape):
        if input_shape is None or len(input_shape) != 3:
            raise ValueError(f"[RNN] expects input shape (B,T,D), got {input_shape}")
        self.input_shape = tuple(map(int, input_shape))
        _, _, D = self.input_shape
        U = self.units

        self.Wx = _init_kernel((D, U), self.kernel_initializer)
        self.Wh = _init_kernel((U, U), self.recurrent_initializer)
        self.b  = _init_bias((1, U), self.bias_initializer) if self.use_bias else None

        # NaN/Inf guard
        if cp.isnan(self.Wx).any() or cp.isinf(self.Wx).any():
            raise RuntimeError("[RNN] Wx contains NaN/Inf")
        if cp.isnan(self.Wh).any() or cp.isinf(self.Wh).any():
            raise RuntimeError("[RNN] Wh contains NaN/Inf")

        B, T, _ = self.input_shape
        self.output_shape = (B, self.units) if not self.return_sequences else (B, T, U)
        self.built = True

    # ---------------------------------------------------------- not implemented
    def __call__(self, x):
        if not self.built:
            self.build(x.shape)
        return self.call(x)

    def call(self, x):
        raise NotImplementedError("Forward pass is handled by CUDA backend")

    def backward(self, grad_output):
        raise NotImplementedError("Backward pass is handled by CUDA backend")

    def update(self, optimizer):
        raise RuntimeError("RNN.update() should not be called; graph_executor handles updates.")

    # --------------------------------------------------------------- shape util
    def compute_output_shape(self, input_shape):
        if input_shape is None or len(input_shape) != 3:
            raise ValueError(f"[RNN] expects (B,T,D), got {input_shape}")
        B, T, _ = map(int, input_shape)
        return (B, T, self.units) if self.return_sequences else (B, self.units)


    def to_e_matrix(self, input_id: str):
        """
        단일 RNN 오퍼로 그래프 생성.
        inputs: [X]
        params: [Wx, Wh, (b), (h0)]
        extra:  time_steps, hidden_size, input_w, use_bias
        출력 shape: (T, H) if return_sequences else (1, H)
        """
        if self.input_shape is None:
            raise ValueError("[RNN] call/build 먼저 수행 필요")

        B, T, D = map(int, self.input_shape)
        U = int(self.units)                    # <-- hidden size
        H = int(self.units)

        weights = {
            self.wx_var: cp.ascontiguousarray(self.Wx.astype(cp.float32)),
            self.wh_var: cp.ascontiguousarray(self.Wh.astype(cp.float32)),
        }
        biases = {}
        if getattr(self, "use_bias", True):
            biases[self.b_var] = cp.ascontiguousarray(self.b.astype(cp.float32))

        # h0는 옵션: 없으면 엔진 내부에서 0버퍼 사용
        h0_id = f"{self.name}_h0"
        if getattr(self, "h0", None) is not None:
            biases[h0_id] = cp.ascontiguousarray(self.h0.astype(cp.float32))
            has_h0 = True
        else:
            # 안 넣어도 됨(엔진에서 0 버퍼 생성). 넣고 싶다면 주석 해제:
            # biases[h0_id] = cp.zeros((1, H), dtype=cp.float32)
            has_h0 = False

        # shape map (per-sample)
        shape_map = {
            input_id:     Shape(T, D),
            self.wx_var:  Shape(D, H),
            self.wh_var:  Shape(H, H),
        }
        if getattr(self, "use_bias", True):
            shape_map[self.b_var] = Shape(1, H)
        if has_h0:
            shape_map[h0_id] = Shape(1, H)

        # 출력 shape
        out_rows = T if getattr(self, "return_sequences", False) else 1
        shape_map[self.out_var] = Shape(out_rows, H)

        # extra
        # RNN.to_e_matrix (핵심 부분만)
        extra = ge.OpExtraParams()
        extra.batch_size  = B
        extra.time_steps  = T
        extra.input_w     = D      # feature dim
        extra.hidden_size = U
        extra.use_bias    = bool(self.use_bias)
        extra.axis        = 1 if self.activation_name == "tanh" else (2 if self.activation_name == "sigmoid" else 0)

        # 파라미터 텐서 등록
        weights = {
            self.wx_var: cp.ascontiguousarray(self.Wx.astype(cp.float32)),
            self.wh_var: cp.ascontiguousarray(self.Wh.astype(cp.float32)),
        }
        biases = {}
        if self.use_bias:
            biases[self.b_var] = cp.ascontiguousarray(self.b.astype(cp.float32))

        # Shape 등록(샘플당 뷰)
        shape_map = {
            input_id:     ge.Shape(T, D),
            self.wx_var:  ge.Shape(D, U),
            self.wh_var:  ge.Shape(U, U),
            self.out_var: ge.Shape(1, U),       # return_sequences=False
        }
        if self.use_bias:
            shape_map[self.b_var] = ge.Shape(1, U)

        # ✅ 벡터 생성자 사용 (inputs, params, output, extra)
        op = ge.OpStruct(
            ge.OpType.RNN,
            [input_id],                                  # inputs
            [self.wx_var, self.wh_var] + ([self.b_var] if self.use_bias else []),  # params
            self.out_var,
            extra
        )
        e_block = [op]
        return e_block, weights, biases, self.out_var, shape_map
