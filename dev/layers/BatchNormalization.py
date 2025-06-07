import cupy as cp
from dev.layers.layer import Layer

class BatchNormalization(Layer):
    def __init__(self, momentum=0.9, epsilon=1e-5, name=None, use_backend_init=False, **kwargs):
        super().__init__(name=name, **kwargs)
        self.momentum = momentum
        self.epsilon = epsilon
        self.training = True
        self.use_backend_init = use_backend_init

        self.gamma = None
        self.beta = None
        self.running_mean = None
        self.running_var = None

        # ✅ GraphCompiler용
        self.name = name or f"batchnorm_{id(self)}"
        self.input_idx = None
        self.output_idx = None
        self.input_shape = None

    def build(self, input_shape):
        self.input_shape = input_shape  # ✅ 반드시 저장
        dim = input_shape[-1]

        if self.use_backend_init:
            return  # shape만 넘기고 파라미터는 backend에서 초기화

        self.gamma = cp.ones((1, dim), dtype=cp.float32)
        self.beta = cp.zeros((1, dim), dtype=cp.float32)
        self.running_mean = cp.zeros((1, dim), dtype=cp.float32)
        self.running_var = cp.ones((1, dim), dtype=cp.float32)

    def call(self, input_data):
        input_data = cp.asarray(input_data, dtype=cp.float32)
        if self.gamma is None:
            self.build(input_data.shape)

        if self.training:
            mean = cp.mean(input_data, axis=0, keepdims=True)
            var = cp.var(input_data, axis=0, keepdims=True)

            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var

            norm = (input_data - mean) / cp.sqrt(var + self.epsilon)
        else:
            norm = (input_data - self.running_mean) / cp.sqrt(self.running_var + self.epsilon)

        return self.gamma * norm + self.beta

    def backward(self, grad_output):
        raise NotImplementedError("BatchNormalization.backward() is not implemented.")

    def compute_output_shape(self, input_shape):
        return input_shape  # BatchNorm은 shape을 유지

    def forward_matrix(self, input_name="input"):
        if self.input_shape is None:
            raise ValueError("BatchNormalization.forward_matrix() called before build().")

        if self.use_backend_init:
            return {
                "input_idx": self.input_idx,
                "output_idx": self.output_idx,
                "op_type": 21,
                "gamma_shape": (1, self.input_shape[-1]),
                "beta_shape": (1, self.input_shape[-1]),
                "W": None,
                "b": None
            }
        else:
            return {
                "input_idx": self.input_idx,
                "output_idx": self.output_idx,
                "op_type": 21,
                "gamma": self.gamma,
                "beta": self.beta
            }
