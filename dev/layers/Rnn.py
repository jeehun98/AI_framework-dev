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

    # -------------------------------------------------------------- E-matrix
    def to_e_matrix(self, input_id: str):
        """
        시간축 언롤을 primitive로 생성:
          for t in [0..T-1]:
            x_t   = SLICE_TIME(input, t)
            z_x   = MATMUL(x_t, Wx)
            z_h   = MATMUL(h_{t-1}, Wh)
            z     = ADD(z_x, z_h)
            z_b   = ADD(z, b)             # if use_bias
            h_t   = ACT(z_b or z)         # if activation
        반환: (e_block, weights, biases, output_id, shape_map)
        """
        if self.input_shape is None:
            raise ValueError("[RNN] build() first")

        B, T, D = map(int, self.input_shape)
        U = int(self.units)

        # 존재 확인 (엔진이 아직 없으면 명확히 안내)
        if not hasattr(ge.OpType, "SLICE_TIME"):
            raise NotImplementedError("Backend is missing OpType.SLICE_TIME for RNN to_e_matrix.")
        need_concat = self.return_sequences
        if need_concat and not hasattr(ge.OpType, "CONCAT_TIME"):
            raise NotImplementedError("Backend is missing OpType.CONCAT_TIME for return_sequences=True.")

        act_op = _activation_op_type(self.activation_name)

        # 파라미터 등록
        weights = { self.wx_var: cp.ascontiguousarray(self.Wx.astype(cp.float32)),
                    self.wh_var: cp.ascontiguousarray(self.Wh.astype(cp.float32)) }
        biases  = {}
        if self.use_bias:
            biases[self.b_var] = cp.ascontiguousarray(self.b.astype(cp.float32))

        # 초기 hidden h_{-1} = 0 을 상수 텐서로 전달 (FILL_ZERO 없이도 가능)
        h0_id = f"{self.name}_h0"
        biases[h0_id] = cp.zeros((1, U), dtype=cp.float32)

        # Shape 매핑 (per-sample 규칙: rows=time, cols=feat)
        shape_map = {
            input_id:      Shape(T, D),
            self.wx_var:   Shape(D, U),
            self.wh_var:   Shape(U, U),
            h0_id:         Shape(1, U),
        }
        if self.use_bias:
            shape_map[self.b_var] = Shape(1, U)

        e_block = []

        h_prev = h0_id
        h_list = []
        for t in range(T):
            xt     = f"{self.name}_x_{t}"
            zx     = f"{self.name}_zx_{t}"
            zh     = f"{self.name}_zh_{t}"
            zsum   = f"{self.name}_z_{t}"
            zbias  = f"{self.name}_zb_{t}"
            ht     = f"{self.name}_h_{t}"

            # shape 등록
            shape_map[xt]   = Shape(1, D)
            shape_map[zx]   = Shape(1, U)
            shape_map[zh]   = Shape(1, U)
            shape_map[zsum] = Shape(1, U)
            shape_map[zbias]= Shape(1, U)
            shape_map[ht]   = Shape(1, U)

            # x_t = SLICE_TIME(input, t)
            extra_t = OpExtraParams()
            extra_t.time_index = t
            extra_t.input_h = T
            extra_t.input_w = D
            e_block.append({
                "op_type": int(getattr(ge.OpType, "SLICE_TIME")),
                "input_id":  input_id,
                "param_id":  "",
                "output_id": xt,
                "extra_params": extra_t
            })

            # z_x = x_t @ Wx
            e_block.append({
                "op_type": int(ge.OpType.MATMUL),
                "input_id":  xt,
                "param_id":  self.wx_var,
                "output_id": zx,
                "extra_params": OpExtraParams()
            })

            # z_h = h_{t-1} @ Wh
            e_block.append({
                "op_type": int(ge.OpType.MATMUL),
                "input_id":  h_prev,
                "param_id":  self.wh_var,
                "output_id": zh,
                "extra_params": OpExtraParams()
            })

            # z = z_x + z_h  (ADD: input_id=z_x, param_id=z_h)
            e_block.append({
                "op_type": int(ge.OpType.ADD),
                "input_id":  zx,
                "param_id":  zh,
                "output_id": zsum,
                "extra_params": OpExtraParams()
            })

            last_id = zsum

            # z + b (same-shape ADD) — use_bias일 때만
            if self.use_bias:
                e_block.append({
                    "op_type": int(ge.OpType.ADD),
                    "input_id":  last_id,
                    "param_id":  self.b_var,
                    "output_id": zbias,
                    "extra_params": OpExtraParams()
                })
                last_id = zbias

            # h_t = activation(last)
            if act_op is not None:
                e_block.append({
                    "op_type": act_op,
                    "input_id":  last_id,
                    "param_id":  "",
                    "output_id": ht,
                    "extra_params": OpExtraParams()
                })
                last_id = ht
            else:
                # 선형이면 last가 곧 h_t
                # 별도 op 없이 alias하려면 마지막 ADD의 output_id를 ht로 바꿔도 됨
                e_block[-1]["output_id"] = ht
                last_id = ht

            h_list.append(ht)
            h_prev = ht

        output_id = self.out_var
        if self.return_sequences:
            # chain CONCAT_TIME: acc = h0; for each next h -> concat
            acc = h_list[0]
            shape_map[acc] = Shape(1, U)
            for i in range(1, T):
                nxt = f"{self.name}_cat_{i}"
                shape_map[nxt] = Shape(i + 1, U)
                e_block.append({
                    "op_type": int(getattr(ge.OpType, "CONCAT_TIME")),
                    "input_id":  acc,
                    "param_id":  h_list[i],
                    "output_id": nxt,
                    "extra_params": OpExtraParams()
                })
                acc = nxt
            # 마지막 concat을 최종 출력으로
            shape_map[output_id] = Shape(T, U)
            e_block[-1]["output_id"] = output_id
        else:
            # 마지막 h_T를 출력으로
            shape_map[output_id] = Shape(1, U)
            e_block[-1]["output_id"] = output_id

        return e_block, weights, biases, output_id, shape_map
