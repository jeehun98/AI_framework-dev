import cupy as cp
from dev.layers.layer import Layer

class RNN(Layer):
    def __init__(self, units, activation=cp.tanh, input_shape=None, name=None, initializer='xavier', use_backend_init=False, **kwargs):
        super().__init__(name, **kwargs)
        self.units = units
        self.activation = activation
        self.initializer = initializer
        self.use_backend_init = use_backend_init

        self.input_shape = input_shape
        self.output_shape = None

        self.Wx = None
        self.Wh = None
        self.b = None

        self.last_inputs = None
        self.last_hs = None

        self.dWx = None
        self.dWh = None
        self.db = None

        # ✅ GraphCompiler용 변수
        self.name = name or f"rnn_{id(self)}"
        self.input_idx = None
        self.output_idx = None

    def build(self, input_shape):
        if len(input_shape) == 2:
            time_steps, input_dim = input_shape
        elif len(input_shape) == 3:
            _, time_steps, input_dim = input_shape
        else:
            raise ValueError(f"RNN input_shape 오류: {input_shape}")

        self.output_shape = (None, self.units)

        if self.use_backend_init:
            # shape 정보만 저장하고 backend 초기화로 대체
            return

        def init(shape):
            if self.initializer == 'xavier':
                stddev = cp.sqrt(1. / shape[0])
                return cp.random.randn(*shape).astype(cp.float32) * stddev
            elif self.initializer == 'he':
                stddev = cp.sqrt(2. / shape[0])
                return cp.random.randn(*shape).astype(cp.float32) * stddev
            elif self.initializer == 'zeros':
                return cp.zeros(shape, dtype=cp.float32)
            elif self.initializer == 'ones':
                return cp.ones(shape, dtype=cp.float32)
            else:
                raise ValueError(f"Unknown initializer: {self.initializer}")

        self.Wx = init((input_dim, self.units))
        self.Wh = init((self.units, self.units))
        self.b = cp.zeros((1, self.units), dtype=cp.float32)

    def call(self, input_seq):
        input_seq = cp.asarray(input_seq, dtype=cp.float32)
        batch, time_steps, input_dim = input_seq.shape

        if self.Wx is None:
            self.build(input_seq.shape)

        h = cp.zeros((batch, self.units), dtype=cp.float32)
        self.last_inputs = []
        self.last_hs = [h]

        for t in range(time_steps):
            x_t = input_seq[:, t, :]
            self.last_inputs.append(x_t)
            h = self.activation(cp.dot(x_t, self.Wx) + cp.dot(h, self.Wh) + self.b)
            self.last_hs.append(h)

        return h

    def backward(self, grad_output):
        dh_next = grad_output
        time_steps = len(self.last_inputs)

        self.dWx = cp.zeros_like(self.Wx)
        self.dWh = cp.zeros_like(self.Wh)
        self.db = cp.zeros_like(self.b)
        dxs = []

        for t in reversed(range(time_steps)):
            h = self.last_hs[t + 1]
            h_prev = self.last_hs[t]
            x_t = self.last_inputs[t]

            if self.activation == cp.tanh:
                dh_raw = dh_next * (1 - h ** 2)
            elif self.activation == cp.relu:
                dh_raw = dh_next * (h > 0)
            else:
                raise ValueError("Only tanh or relu supported in RNN.backward()")

            self.dWx += cp.dot(x_t.T, dh_raw)
            self.dWh += cp.dot(h_prev.T, dh_raw)
            self.db += cp.sum(dh_raw, axis=0, keepdims=True)

            dx = cp.dot(dh_raw, self.Wx.T)
            dh_next = cp.dot(dh_raw, self.Wh.T)
            dxs.insert(0, dx)

        return cp.stack(dxs, axis=1)

    def update(self, optimizer):
        optimizer.update(self.Wx, self.dWx, self.Wh, self.dWh, self.b, self.db)

    # ✅ GraphCompiler용 연산 정보 정의
    def forward_matrix(self, input_name="input"):
        input_dim = self.input_shape[2] if self.input_shape and len(self.input_shape) == 3 else None

        if self.use_backend_init:
            return {
                "input_idx": self.input_idx,
                "output_idx": self.output_idx,
                "op_type": 10,  # 예: RNN 연산 코드
                "W_shape": (input_dim, self.units),
                "Wh_shape": (self.units, self.units),
                "b_shape": (1, self.units),
                "W": None,
                "Wh": None,
                "b": None
            }
        else:
            return {
                "input_idx": self.input_idx,
                "output_idx": self.output_idx,
                "op_type": 10,
                "W": self.Wx,
                "Wh": self.Wh,
                "b": self.b
            }

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)
