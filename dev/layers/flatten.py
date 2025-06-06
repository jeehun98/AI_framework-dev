import numpy as np
import cupy as cp
from functools import reduce
from operator import mul

from dev.layers.layer import Layer

class Flatten(Layer):
    def __init__(self, input_shape=None, name=None, use_backend_init=False, **kwargs):
        super().__init__(name=name, **kwargs)
        self.input_shape = input_shape
        self.output_shape = input_shape
        self.trainable = False
        self.layer_name = "flatten"
        self.use_backend_init = use_backend_init

        # ✅ GraphCompiler 연산 정보용
        self.name = name or f"flatten_{id(self)}"
        self.input_idx = None
        self.output_idx = None
        self.input_var = None
        self.output_var = f"{self.name}_out"

    def get_config(self):
        base_config = super().get_config()
        config = {
            "class_name": self.__class__.__name__,
            "input_shape": self.input_shape
        }
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def call(self, inputs):
        if isinstance(inputs, cp.ndarray):
            inputs = inputs.astype(cp.float32)
        else:
            inputs = np.asarray(inputs, dtype=np.float32)

        if inputs.ndim == 1:
            inputs = inputs.reshape(1, -1)
        elif inputs.ndim > 2:
            batch_size = inputs.shape[0]
            flattened_dim = int(np.prod(inputs.shape[1:]))
            inputs = inputs.reshape(batch_size, flattened_dim)

        self.output_shape = inputs.shape
        return inputs

    def compute_output_shape(self, input_shape):
        return (input_shape[0], int(np.prod(input_shape[1:])))

    def build(self, input_shape):
        print("flatten 호출 확인")
        flattened_dim = int(np.prod(input_shape[1:])) if len(input_shape) > 1 else input_shape[0]
        self.input_shape = input_shape
        self.output_shape = (1, flattened_dim)
        super().build(input_shape)

    def get_weights(self):
        return []

    def backward(self, grad_output):
        return grad_output.reshape(self.input_shape)

    # ✅ GraphCompiler용 forward_matrix 정의
    def forward_matrix(self, input_name="input"):
        self.input_var = input_name
        return {
            "input_idx": self.input_idx,    # GraphCompiler가 자동으로 채움
            "output_idx": self.output_idx,
            "op_type": 5,                   # 예: Flatten 연산용 사용자 정의 코드
            "W": None,
            "b": None
        }
