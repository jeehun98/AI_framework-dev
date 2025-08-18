import cupy as cp
from dev.layers.layer import Layer

import sys
sys.path.append("C:/Users/owner/Desktop/AI_framework-dev/dev/backend/graph_executor/test")
import graph_executor as ge  # Pybind11 모듈

Shape = ge.Shape
OpExtraParams = ge.OpExtraParams

class Dense(Layer):
    def __init__(self, units, activation=None, input_shape=None, name=None, initializer='he', **kwargs):
        super().__init__(name=name, **kwargs)
        self.layer_name = "dense"
        self.units = units
        self.activation = activation
        self.initializer = initializer
        self.input_shape = input_shape

        self.weights = None
        self.bias = None
        self.last_input = None
        self.last_z = None
        self.dW = None
        self.db = None

        # GraphCompiler용 식별자
        self.name = name or f"dense_{id(self)}"
        self.weight_var = f"{self.name}_W"
        self.bias_var = f"{self.name}_b"
        self.output_var = f"{self.name}_out"

        # 활성화 이름 캐싱
        self.activation_name = getattr(self.activation, "__name__", str(self.activation)).lower() if self.activation else None

        # (옵션) 엔진이 bias 브로드캐스트를 못 하면 True로 바꿔 타일링 사용
        self.force_bias_tile = False

    def build(self, input_shape):
        self.input_shape = input_shape
        input_dim = int(input_shape[1])

        if self.initializer == 'zeros':
            self.weights = cp.zeros((input_dim, self.units), dtype=cp.float32)
        elif self.initializer == 'ones':
            self.weights = cp.ones((input_dim, self.units), dtype=cp.float32)
        elif self.initializer == 'uniform':
            limit = 0.05
            self.weights = cp.random.uniform(-limit, limit, (input_dim, self.units)).astype(cp.float32)
        elif self.initializer == 'normal':
            self.weights = cp.random.normal(0.0, 0.05, (input_dim, self.units)).astype(cp.float32)
        elif self.initializer == 'xavier':
            limit = cp.sqrt(6.0 / (input_dim + self.units))
            self.weights = cp.random.uniform(-limit, limit, (input_dim, self.units)).astype(cp.float32)
        elif self.initializer == 'he':
            std = cp.sqrt(2.0 / input_dim)
            self.weights = cp.random.normal(0.0, std, (input_dim, self.units)).astype(cp.float32)
        elif self.initializer == 'lecun':
            std = cp.sqrt(1.0 / input_dim)
            self.weights = cp.random.normal(0.0, std, (input_dim, self.units)).astype(cp.float32)
        elif self.initializer == 'small_uniform':
            self.weights = cp.random.uniform(-1e-3, 1e-3, (input_dim, self.units)).astype(cp.float32)
        else:
            raise ValueError(f"[Dense] Unknown initializer: {self.initializer}")

        # bias는 (1, units)로 둠 (엔진이 브로드캐스트 못 하면 타일링 옵션 사용)
        self.bias = cp.random.uniform(-1e-3, 1e-3, (1, self.units)).astype(cp.float32)

        # NaN/Inf 방어
        if cp.isnan(self.weights).any() or cp.isinf(self.weights).any():
            raise RuntimeError("[Init] Weight contains NaN or Inf")

        self.built = True
        self.output_shape = self.compute_output_shape(input_shape)

    def __call__(self, x):
        if not self.built:
            self.build(x.shape)
        return self.call(x)

    def call(self, x):
        self.last_input = x
        z = cp.dot(x, self.weights) + (self.bias if self.bias.shape[0] == 1 else self.bias)
        self.last_z = z
        return self._apply_activation(z)

    def backward(self, grad_output):
        grad_output = cp.asarray(grad_output, dtype=cp.float32)
        if self.activation_name:
            grad_output = self._apply_activation_grad(grad_output, self.last_z)
        self.dW = cp.dot(self.last_input.T, grad_output)
        self.db = cp.sum(grad_output, axis=0, keepdims=True)
        dx = cp.dot(grad_output, self.weights.T)
        return dx

    def update(self, optimizer):
        reg_term = self.apply_regularizer(self.weights)
        optimizer.update(self.weights, self.dW + reg_term, self.bias, self.db)

    def _apply_activation(self, z):
        if self.activation is None:
            return z
        return self.activation(z)

    def _apply_activation_grad(self, grad_output, z):
        if self.activation_name == "sigmoid":
            sig = 1 / (1 + cp.exp(-z))
            return grad_output * sig * (1 - sig)
        elif self.activation_name == "relu":
            return grad_output * (z > 0)
        elif self.activation_name == "tanh":
            return grad_output * (1 - cp.tanh(z) ** 2)
        else:
            raise ValueError(f"Unsupported activation grad: {self.activation_name}")

    def compute_output_shape(self, input_shape):
        if input_shape is None or len(input_shape) != 2:
            raise ValueError(f"Dense layer expects input shape to be 2D (batch_size, input_dim), got {input_shape}")
        batch_size, _ = input_shape
        return (int(batch_size), int(self.units))

    def to_e_matrix(self, input_id):
        if self.input_shape is None:
            raise ValueError("[Dense] input_shape is None. Did you forget to call build()?")

        batch, input_dim = map(int, self.input_shape)
        units = int(self.units)

        weight_id = self.weight_var
        bias_id = self.bias_var
        linear_out_id = f"{self.name}_linear"
        output_id = self.output_var
        preact_id = f"{self.name}_preact"

        extra = OpExtraParams()

        # ✅ 정수 대신 enum 사용 (가장 중요)
        e_block = [
            {
                "op_type": ge.OpType.MATMUL,
                "input_id": input_id,
                "param_id": weight_id,
                "output_id": linear_out_id,
                "extra_params": extra
            },
            {
                "op_type": ge.OpType.ADD,
                "input_id": linear_out_id,
                "param_id": bias_id,
                "output_id": preact_id if self.activation_name else output_id,
                "extra_params": extra
            }
        ]

        if self.activation_name:
            act_map = {
                "relu": ge.OpType.RELU,
                "sigmoid": ge.OpType.SIGMOID,
                "tanh": ge.OpType.TANH,
            }
            if self.activation_name not in act_map:
                raise ValueError(f"Unsupported activation: {self.activation_name}")
            e_block.append({
                "op_type": act_map[self.activation_name],
                "input_id": preact_id,
                "param_id": "",
                "output_id": output_id,
                "extra_params": extra
            })

        # 파라미터 텐서 반환
        weights = {weight_id: self.weights}
        biases = {bias_id: self.bias if not self.force_bias_tile else cp.tile(self.bias, (batch, 1))}

        # 셰이프 맵
        shape_map = {
            input_id: Shape(batch, input_dim),
            weight_id: Shape(input_dim, units),   # (in, out)
            bias_id: Shape(1 if not self.force_bias_tile else batch, units),
            linear_out_id: Shape(batch, units),
            output_id: Shape(batch, units),
        }
        if self.activation_name:
            shape_map[preact_id] = Shape(batch, units)

        return e_block, weights, biases, output_id, shape_map
