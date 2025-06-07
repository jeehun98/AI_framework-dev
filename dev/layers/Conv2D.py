import cupy as cp
from dev.layers.layer import Layer

class Conv2D(Layer):
    def __init__(self, filters, kernel_size, activation=None, input_shape=None, name=None,
                 initializer='he', use_backend_init=False, **kwargs):
        super().__init__(name=name, **kwargs)
        self.filters = filters
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.activation = activation
        self.initializer = initializer
        self.use_backend_init = use_backend_init

        self.input_shape = input_shape
        self.output_shape = None

        self.weights = None
        self.bias = None
        self.last_input = None
        self.last_z = None
        self.dW = None
        self.db = None

        # ✅ GraphCompiler용 인덱스
        self.name = name or f"conv2d_{id(self)}"
        self.input_idx = None
        self.output_idx = None

    def build(self, input_shape):
        if len(input_shape) == 3:
            in_h, in_w, in_channels = input_shape
        elif len(input_shape) == 4:
            _, in_h, in_w, in_channels = input_shape
        else:
            raise ValueError(f"Conv2D input_shape 오류: {input_shape}")

        kh, kw = self.kernel_size
        out_channels = self.filters

        self.output_shape = (None, in_h - kh + 1, in_w - kw + 1, out_channels)

        if self.use_backend_init:
            return  # ✅ backend에서 초기화

        fan_in = kh * kw * in_channels
        if self.initializer == 'he':
            stddev = cp.sqrt(2. / fan_in)
            self.weights = cp.random.randn(kh, kw, in_channels, out_channels).astype(cp.float32) * stddev
        elif self.initializer == 'xavier':
            stddev = cp.sqrt(1. / fan_in)
            self.weights = cp.random.randn(kh, kw, in_channels, out_channels).astype(cp.float32) * stddev
        elif self.initializer == 'zeros':
            self.weights = cp.zeros((kh, kw, in_channels, out_channels), dtype=cp.float32)
        elif self.initializer == 'ones':
            self.weights = cp.ones((kh, kw, in_channels, out_channels), dtype=cp.float32)
        else:
            raise ValueError(f"Unknown initializer: {self.initializer}")

        self.bias = cp.zeros((out_channels,), dtype=cp.float32)

    def call(self, input_data):
        input_data = cp.asarray(input_data, dtype=cp.float32)
        if self.weights is None:
            self.build(input_data.shape)

        self.last_input = input_data
        batch, in_h, in_w, in_channels = input_data.shape
        kh, kw, _, out_channels = self.weights.shape
        out_h = in_h - kh + 1
        out_w = in_w - kw + 1

        output = cp.zeros((batch, out_h, out_w, out_channels), dtype=cp.float32)

        for b in range(batch):
            for h in range(out_h):
                for w in range(out_w):
                    for c in range(out_channels):
                        region = input_data[b, h:h+kh, w:w+kw, :]
                        output[b, h, w, c] = cp.sum(region * self.weights[:, :, :, c]) + self.bias[c]

        self.last_z = output

        if self.activation:
            output = self._apply_activation(output)

        return output

    def backward(self, grad_output):
        raise NotImplementedError("Conv2D.backward()는 아직 구현되지 않았습니다.")

    def update(self, optimizer):
        if self.dW is not None and self.db is not None:
            optimizer.update(self.weights, self.dW, self.bias, self.db)

    def _apply_activation(self, z):
        if self.activation is None:
            return z
        return self.activation(z)

    # ✅ GraphCompiler용 forward_matrix 정의
    def forward_matrix(self, input_name="input"):
        if self.use_backend_init:
            # Conv2D weights shape: (kh, kw, in_channels, out_channels)
            _, in_h, in_w, in_channels = self.input_shape
            kh, kw = self.kernel_size
            return {
                "input_idx": self.input_idx,
                "output_idx": self.output_idx,
                "op_type": 20,  # Conv2D 연산 코드
                "W_shape": (kh, kw, in_channels, self.filters),
                "b_shape": (self.filters,),
                "W": None,
                "b": None
            }
        else:
            return {
                "input_idx": self.input_idx,
                "output_idx": self.output_idx,
                "op_type": 20,  # Conv2D 연산 코드
                "W": self.weights,
                "b": self.bias
            }
        
    def compute_output_shape(self, input_shape):
        _, in_h, in_w, in_channels = input_shape
        kh, kw = self.kernel_size
        out_channels = self.filters
        out_h = in_h - kh + 1
        out_w = in_w - kw + 1
        return (None, out_h, out_w, out_channels)
