import numpy as np
import cupy as cp
from dev.layers.layer import Layer

import sys
sys.path.append("C:/Users/owner/Desktop/AI_framework-dev/dev/backend/graph_executor")
import graph_executor as ge  # Pybind11 모듈

# graph_executor에서 정의된 Shape 사용
Shape = ge.Shape


class Flatten(Layer):
    def __init__(self, input_shape=None, name=None, use_backend_init=False, **kwargs):
        super().__init__(name=name, **kwargs)
        self.input_shape = input_shape
        self.output_shape = input_shape
        self.trainable = False
        self.layer_name = "flatten"
        self.use_backend_init = use_backend_init

        self.name = name or f"flatten_{id(self)}"
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
        flattened_dim = int(np.prod(input_shape[1:])) if len(input_shape) > 1 else input_shape[0]
        self.input_shape = input_shape
        self.output_shape = (1, flattened_dim)
        super().build(input_shape)

    def get_weights(self):
        return []

    def backward(self, grad_output):
        return grad_output.reshape(self.input_shape)

    def to_e_matrix(self, input_id):
        """
        Flatten 연산: 차원만 변형하므로 W, b는 없음
        """
        output_id = self.output_var

        e_block = [{
            "op_type": 5,  # Flatten
            "input_id": input_id,
            "param_id": None,
            "output_id": output_id
        }]

        # ✅ Flatten된 shape 계산
        input_rows = self.input_shape[0]
        input_cols = int(np.prod(self.input_shape[1:]))

        shape_map = {
            input_id: ge.Shape(input_rows, input_cols),
            output_id: ge.Shape(*self.output_shape)
        }

        return e_block, {}, {}, output_id, shape_map
