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
            "op_type": 5,  # FLATTEN
            "input_id": input_id,
            "param_id": "",
            "output_id": output_id,
            "extra_params": extra
        }]

        # 🔽 입력 2D 해석을 '그대로' 사용: (rows_in, cols_in) = (F, H*W) 또는 이전 레이어의 2D
        # - Conv2D가 위에서 (F, H*W)로 보내므로 자동으로 맞습니다.
        # - 만약 이전 레이어가 이미 (1, K)라면 rows_in=1, cols_in=K가 됩니다.
        # 주의: self.input_shape는 원래 4D(NHWC)일 수 있으므로, 2D Shape는 engine의 shape_map에서 가져오는 게 이상적.
        # 여기서는 간단히, Conv2D 이후라는 가정 아래 rows_in/cols_in을 계산합니다.
        # 안전하게 하려면 Sequential.compile 단계에서 shape_map을 전달받아 참조하세요.
        if len(self.input_shape) == 4:
            _, H, W, C = map(int, self.input_shape)
            rows_in, cols_in = int(C), int(H * W)   # ✅ Conv 정렬과 일치
        else:
            # 2D 이어받는 경우 등
            rows_in, cols_in = 1, int(np.prod(self.input_shape[1:]))

        rows_out, cols_out = 1, rows_in * cols_in
        shape_map = {
            input_id:  Shape(int(rows_in), int(cols_in)),
            output_id: Shape(int(rows_out), int(cols_out)),
        }
        return e_block, {}, {}, output_id, shape_map