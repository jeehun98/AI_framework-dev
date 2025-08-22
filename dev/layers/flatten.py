import numpy as np
import cupy as cp
from dev.layers.layer import Layer

import sys
sys.path.append("C:/Users/owner/Desktop/AI_framework-dev/dev/backend/graph_executor/test")
import graph_executor as ge  # Pybind11 모듈

Shape = ge.Shape
OpExtraParams = ge.OpExtraParams

class Flatten(Layer):
    def __init__(self, input_shape=None, name=None, use_backend_init=False, **kwargs):
        super().__init__(name=name, **kwargs)
        self.input_shape = input_shape
        self.output_shape = None
        self.trainable = False
        self.layer_name = "flatten"
        self.use_backend_init = use_backend_init

        self.name = name or f"flatten_{id(self)}"
        self.output_var = f"{self.name}_out"

    def build(self, input_shape):
        self.input_shape = input_shape
        self.output_shape = self.compute_output_shape(input_shape)
        self.built = True

    def call(self, inputs):
        inputs = cp.asarray(inputs, dtype=cp.float32)

        if inputs.ndim == 1:
            inputs = inputs.reshape(1, -1)
        elif inputs.ndim > 2:
            batch_size = inputs.shape[0]
            flattened_dim = int(np.prod(inputs.shape[1:]))
            inputs = inputs.reshape(batch_size, flattened_dim)

        self.output_shape = inputs.shape

        return inputs

    def __call__(self, x):
        if not self.built:
            self.build(x.shape)
        return self.call(x)

    def backward(self, grad_output):
        return grad_output.reshape(self.input_shape)

    def compute_output_shape(self, input_shape):
        if len(input_shape) < 2:
            raise ValueError(f"Flatten layer expects input shape with rank ≥2, got {input_shape}")
        batch = input_shape[0]
        flattened = int(np.prod(input_shape[1:]))
        return (batch, flattened)

    def get_weights(self):
        return []

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

    def to_e_matrix(self, input_id):
        if self.input_shape is None:
            raise ValueError("[Flatten] input_shape is None. Did you forget to call build()?")

        if self.output_shape is None:
            self.output_shape = self.compute_output_shape(self.input_shape)

        output_id = self.output_var
        extra = OpExtraParams()

        e_block = [{
            "op_type": int(ge.OpType.FLATTEN),  # FLATTEN
            "input_id": input_id,
            "param_id": "",
            "output_id": output_id,
            "extra_params": extra
        }]

        # ✅ 입력 shape는 이전 op가 이미 정확히 등록했으므로 건드리지 않는다.
        #    출력만 등록: (rows=1, cols=배치 제외 모든 차원의 곱)
        if len(self.input_shape) >= 2:
            flattened = int(np.prod(self.input_shape[1:]))  # 레이아웃(NCHW/NHWC) 무관
        else:
            raise ValueError(f"Flatten expects rank>=2, got {self.input_shape}")

        shape_map = {
            output_id: Shape(1, flattened),
        }

        # 가중치/바이어스 없음
        return e_block, {}, {}, output_id, shape_map
