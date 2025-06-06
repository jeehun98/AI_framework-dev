from dev.layers.layer import Layer
import cupy as cp

class Dropout(Layer):
    def __init__(self, rate=0.5, name=None, use_backend_init=False, **kwargs):
        super().__init__(name=name, **kwargs)
        self.rate = rate
        self.mask = None
        self.training = True
        self.use_backend_init = use_backend_init

        # ✅ GraphCompiler용
        self.name = name or f"dropout_{id(self)}"
        self.input_idx = None
        self.output_idx = None

    def call(self, input_data):
        input_data = cp.asarray(input_data, dtype=cp.float32)
        if self.training:
            self.mask = cp.random.rand(*input_data.shape) > self.rate
            return input_data * self.mask / (1.0 - self.rate)
        else:
            return input_data

    def backward(self, grad_output):
        if self.mask is None:
            raise RuntimeError("Dropout called without mask. Make sure training is True.")
        return grad_output * self.mask / (1.0 - self.rate)

    # ✅ GraphCompiler 연산 정의
    def forward_matrix(self, input_name="input"):
        return {
            "input_idx": self.input_idx,
            "output_idx": self.output_idx,
            "op_type": 22,   # Dropout 연산 번호
            "dropout_rate": self.rate,
            "W": None,
            "b": None
        }
