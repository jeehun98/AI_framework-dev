import cupy as cp
from dev.layers.layer import Layer

class Dense(Layer):
    def __init__(self, units, activation=None, input_shape=None, name=None, initializer='he', **kwargs):
        super().__init__(name, **kwargs)
        self.layer_name = "dense"
        self.units = units
        self.activation = activation
        self.initializer = initializer
        self.input_shape = input_shape
        self.output_shape = (1, units)

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

    def to_e_matrix(self, input_id):
        weight_id = f"{self.name}_W"
        bias_id = f"{self.name}_b"
        linear_out_id = f"{self.name}_linear"
        output_id = f"{self.name}_out"

        e_block = [
            {
                "op_type": 0,  # matmul
                "input_id": input_id,
                "param_id": weight_id,
                "output_id": linear_out_id
            },
            {
                "op_type": 1,  # add
                "input_id": linear_out_id,
                "param_id": bias_id,
                "output_id": output_id if self.activation is None else f"{self.name}_preact"
            }
        ]

        # ✅ 활성화 함수가 있는 경우, 추가 노드 삽입
        if self.activation:
            activation_name = self.activation.__name__.lower()
            activation_map = {
                "relu": 2,
                "sigmoid": 3,
                "tanh": 4
            }
            if activation_name not in activation_map:
                raise ValueError(f"Unsupported activation: {activation_name}")

            e_block.append({
                "op_type": activation_map[activation_name],  # e.g. 2: relu
                "input_id": f"{self.name}_preact",
                "param_id": None,
                "output_id": output_id
            })

        weights = {weight_id: self.weights}
        biases = {bias_id: self.bias}

        return e_block, weights, biases, output_id


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