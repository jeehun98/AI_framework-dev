import cupy as cp
from dev.layers.layer import Layer

class Dense(Layer):
    def __init__(self, units, activation=None, input_shape=None, name=None, initializer='he', use_backend_init=False, **kwargs):
        super().__init__(name, **kwargs)
        self.layer_name = "dense"
        self.units = units
        self.activation = activation
        self.initializer = initializer
        self.input_shape = input_shape
        self.output_shape = (1, units)
        self.use_backend_init = use_backend_init

        self.weights = None
        self.bias = None
        self.last_input = None
        self.last_z = None
        self.dW = None
        self.db = None

        # ✅ GraphCompiler 변수
        self.name = name or f"dense_{id(self)}"
        self.input_var = None
        self.weight_var = f"{self.name}_W"
        self.bias_var = f"{self.name}_b"
        self.output_var = f"{self.name}_out"

    def build(self, input_shape):
        self.input_shape = input_shape
        input_dim = input_shape[1]

        if self.use_backend_init:
            # backend에서 초기화할 것이므로 생성 생략
            self.weights = None
            self.bias = None
            return

        if self.initializer == 'ones':
            self.weights = cp.ones((input_dim, self.units), dtype=cp.float32)
        elif self.initializer == 'zeros':
            self.weights = cp.zeros((input_dim, self.units), dtype=cp.float32)
        else:
            limit = cp.sqrt(2 / input_dim)
            self.weights = cp.random.randn(input_dim, self.units).astype(cp.float32) * limit

        self.bias = cp.zeros((1, self.units), dtype=cp.float32)

    def call(self, x):
        self.last_input = x
        z = cp.dot(x, self.weights) + self.bias
        self.last_z = z
        if self.activation:
            return self.activation(z)
        return z

    # ✅ GraphCompiler용 연산 정의 (Dense Layer)
    def forward_matrix(self, input_name="input"):
        self.input_var = input_name

        # GraphCompiler가 연결 정보를 할당할 수 있도록 None으로 초기화
        matmul_op = {
            "input_idx": None,
            "param_idx": None,   # weight
            "output_idx": None,
            "op_type": 0,  # matmul
            "W_shape": (self.input_shape[1], self.units) if self.use_backend_init else None,
            "W": None if self.use_backend_init else self.weights
        }

        add_op = {
            "input_idx": None,
            "param_idx": None,   # bias
            "output_idx": None,
            "op_type": 1,  # add
            "b_shape": (1, self.units) if self.use_backend_init else None,
            "b": None if self.use_backend_init else self.bias
        }

        return [matmul_op, add_op]


    def backward(self, grad_output):
        grad_output = cp.asarray(grad_output, dtype=cp.float32)

        if self.activation:
            grad_output = self._apply_activation_grad(grad_output, self.last_z)

        self.dW = cp.dot(self.last_input.T, grad_output)
        self.db = cp.sum(grad_output, axis=0, keepdims=True)
        dx = cp.dot(grad_output, self.weights.T)
        return dx

    def update(self, optimizer):
        optimizer.update(self.weights, self.dW, self.bias, self.db)

    def _apply_activation(self, z):
        if self.activation is None:
            return z
        return self.activation(z)
    
    def _apply_activation_grad(self, grad_output, z):
        name = self.activation.__name__.lower()
        if name == "sigmoid":
            sig = 1 / (1 + cp.exp(-z))
            return grad_output * sig * (1 - sig)
        elif name == "relu":
            return grad_output * (z > 0)
        elif name == "tanh":
            return grad_output * (1 - cp.tanh(z) ** 2)
        else:
            raise ValueError(f"Unsupported activation grad: {name}")

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)