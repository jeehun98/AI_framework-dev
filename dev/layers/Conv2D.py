# dev/layers/conv2d.py
import cupy as cp
import numpy as np
from dev.layers.layer import Layer

import sys
sys.path.append("C:/Users/owner/Desktop/AI_framework-dev/dev/backend/graph_executor")
import graph_executor as ge  # Pybind11 모듈

Shape = ge.Shape
OpExtraParams = ge.OpExtraParams


class Conv2D(Layer):
    def __init__(self, filters, kernel_size, strides=(1, 1), padding='valid',
                 activation=None, input_shape=None, name=None, initializer='he',
                 use_bias=True, force_bias_tile=False,  # 엔진이 채널편향 브로드캐스트 못 할 경우
                 **kwargs):
        super().__init__(name=name, **kwargs)
        self.layer_name = "conv2d"
        self.filters = int(filters)
        self.kernel_size = tuple(map(int, kernel_size))
        self.strides = tuple(map(int, strides))
        self.padding = padding.lower()
        self.activation = activation
        self.initializer = initializer
        self.input_shape = input_shape
        self.use_bias = bool(use_bias)
        self.force_bias_tile = bool(force_bias_tile)

        self.name = name or f"conv2d_{id(self)}"
        self.weight_var = f"{self.name}_W"
        self.bias_var = f"{self.name}_b"
        self.output_var = f"{self.name}_out"

        self.weights = None  # (F, Cin, Kh, Kw), contiguous
        self.bias = None     # (F,)

        self.activation_name = (
            getattr(self.activation, "__name__", str(self.activation)).lower()
            if self.activation else None
        )

        # 내부 캐시
        self.out_h = None
        self.out_w = None
        self.pad_h_total = 0
        self.pad_w_total = 0

    def __call__(self, x):
        if not self.built:
            self.build(x.shape)
        return self.call(x)

    def build(self, input_shape):
        """
        기대 입력: (B, H, W, C)
        특별히 (B, H, W)가 오면 채널=1로 보정.
        """
        if len(input_shape) == 3:
            # (B,H,W) -> (B,H,W,1)
            input_shape = (*input_shape, 1)
        if len(input_shape) != 4:
            raise ValueError(f"[Conv2D] build: expected input shape (B, H, W, C), got {input_shape}")

        self.input_shape = tuple(map(int, input_shape))
        b, in_h, in_w, in_c = self.input_shape
        kh, kw = self.kernel_size
        sh, sw = self.strides

        if self.padding not in ("valid", "same"):
            raise ValueError(f"[Conv2D] Unsupported padding: {self.padding}")

        # 출력 크기 & SAME 패딩 계산
        if self.padding == 'valid':
            out_h = (in_h - kh) // sh + 1
            out_w = (in_w - kw) // sw + 1
            pad_h_total = 0
            pad_w_total = 0
        else:  # 'same'
            out_h = int(np.ceil(in_h / sh))
            out_w = int(np.ceil(in_w / sw))
            pad_h_total = max((out_h - 1) * sh + kh - in_h, 0)
            pad_w_total = max((out_w - 1) * sw + kw - in_w, 0)

        if out_h <= 0 or out_w <= 0:
            raise ValueError(f"[Conv2D] Invalid output shape: {out_h}x{out_w}")

        self.out_h, self.out_w = int(out_h), int(out_w)
        self.pad_h_total, self.pad_w_total = int(pad_h_total), int(pad_w_total)
        self.output_shape = (b, self.out_h, self.out_w, self.filters)

        # 가중치 초기화 (He normal 기본)
        fan_in = in_c * kh * kw
        if self.initializer == 'ones':
            W = cp.ones((self.filters, in_c, kh, kw), dtype=cp.float32)
        elif self.initializer == 'zeros':
            W = cp.zeros((self.filters, in_c, kh, kw), dtype=cp.float32)
        elif self.initializer == 'xavier':
            limit = cp.sqrt(6.0 / (fan_in + self.filters))
            W = cp.random.uniform(-limit, limit, (self.filters, in_c, kh, kw)).astype(cp.float32)
        elif self.initializer == 'uniform':
            limit = 0.05
            W = cp.random.uniform(-limit, limit, (self.filters, in_c, kh, kw)).astype(cp.float32)
        elif self.initializer == 'normal':
            W = cp.random.normal(0.0, 0.05, (self.filters, in_c, kh, kw)).astype(cp.float32)
        elif self.initializer == 'lecun':
            std = cp.sqrt(1.0 / fan_in)
            W = cp.random.normal(0.0, std, (self.filters, in_c, kh, kw)).astype(cp.float32)
        else:  # 'he' 및 기본값
            std = cp.sqrt(2.0 / fan_in)
            W = cp.random.normal(0.0, std, (self.filters, in_c, kh, kw)).astype(cp.float32)

        self.weights = cp.ascontiguousarray(W, dtype=cp.float32)
        if self.use_bias:
            self.bias = cp.zeros((self.filters,), dtype=cp.float32)
        else:
            self.bias = None

        if cp.isnan(self.weights).any() or cp.isinf(self.weights).any():
            raise RuntimeError("[Conv2D] Weight contains NaN/Inf")

        self.built = True

    def call(self, x):
        # 실제 forward는 CUDA backend에서 처리 (GraphExecutor 경로 사용)
        raise NotImplementedError("Forward pass is handled by CUDA backend")

    def backward(self, grad_output):
        raise NotImplementedError("Backward pass is handled by CUDA backend")

    def update(self, optimizer):
        raise RuntimeError("Conv2D.update() should not be called; graph_executor handles updates.")

    def compute_output_shape(self, input_shape):
        if len(input_shape) == 3:
            input_shape = (*input_shape, 1)
        b, in_h, in_w, in_c = map(int, input_shape)
        kh, kw = self.kernel_size
        sh, sw = self.strides

        if self.padding == 'valid':
            out_h = (in_h - kh) // sh + 1
            out_w = (in_w - kw) // sw + 1
        elif self.padding == 'same':
            out_h = int(np.ceil(in_h / sh))
            out_w = int(np.ceil(in_w / sw))
        else:
            raise ValueError(f"[Conv2D] Unknown padding: {self.padding}")

        return (b, int(out_h), int(out_w), self.filters)

    def _activation_op_type(self):
        if not self.activation_name:
            return None
        act_map = {
            "relu": int(ge.OpType.RELU),
            "sigmoid": int(ge.OpType.SIGMOID),
            "tanh": int(ge.OpType.TANH),
        }
        if self.activation_name not in act_map:
            raise ValueError(f"[Conv2D] Unsupported activation: {self.activation_name}")
        return act_map[self.activation_name]

    def to_e_matrix(self, input_id):
        """
        Conv2D → (optional) Bias → (optional) Activation
        Shape 규칙: 'per-sample' 행렬화
          - input:  Shape(1, Cin*Hin*Win)   (배치는 extra.batch_size 사용)
          - weight: Shape(F, Cin*Kh*Kw)
          - output: Shape(F, Hout*Wout)
        """
        if self.input_shape is None:
            raise ValueError("[Conv2D] input_shape is None. Did you forget to call build()?")

        b, in_h, in_w, in_c = self.input_shape
        kh, kw = self.kernel_size
        sh, sw = self.strides
        out_h, out_w = self.out_h, self.out_w

        weight_id   = self.weight_var
        bias_id     = self.bias_var
        output_id   = self.output_var
        preact_id   = f"{self.name}_preact"
        # 중간 conv id는 post-op 여부에 따라 생략 가능
        conv_out_id = f"{self.name}_conv"

        # Extra
        extra = OpExtraParams()
        extra.batch_size = int(b)
        extra.input_h = int(in_h)
        extra.input_w = int(in_w)
        extra.input_c = int(in_c)
        extra.output_c = int(self.filters)
        extra.kernel_h = int(kh)
        extra.kernel_w = int(kw)
        extra.stride_h = int(sh)
        extra.stride_w = int(sw)
        extra.padding_h = int(self.pad_h_total // 2)
        extra.padding_w = int(self.pad_w_total // 2)
        extra.use_bias = bool(self.use_bias)

        e_block = []

        # 1) CONV2D
        # bias/activation이 전혀 없으면 conv가 곧 최종 출력이 되도록 output_id를 바로 지정
        conv_write_id = output_id if (not self.use_bias and not self.activation_name) else conv_out_id
        e_block.append({
            "op_type": int(ge.OpType.CONV2D),
            "input_id":  input_id,
            "param_id":  weight_id,
            "output_id": conv_write_id,
            "extra_params": extra
        })

        # 2) Bias (옵션)
        biases = {}
        bias_shape = None
        last_id = conv_write_id
        if self.use_bias:
            add_bias_supported = hasattr(ge.OpType, "ADD_BIAS")
            if add_bias_supported and not self.force_bias_tile:
                # (1, F) 형태로 채널 편향 브로드캐스트
                bvec = self.bias.reshape(1, -1).astype(cp.float32)
                biases[bias_id] = cp.ascontiguousarray(bvec)
                bias_shape = Shape(1, int(self.filters))

                last_id = preact_id if self.activation_name else output_id
                e_block.append({
                    "op_type": int(getattr(ge.OpType, "ADD_BIAS")),
                    "input_id":  conv_out_id,
                    "param_id":  bias_id,
                    "output_id": last_id,
                    "extra_params": extra
                })
            else:
                # (1, F*H*W) 로 타일링 → ADD (동형)
                tile_per_channel = out_h * out_w
                bvec = cp.repeat(self.bias, tile_per_channel).reshape(1, -1).astype(cp.float32)
                biases[bias_id] = cp.ascontiguousarray(bvec)
                bias_shape = Shape(1, int(self.filters * out_h * out_w))

                last_id = preact_id if self.activation_name else output_id
                e_block.append({
                    "op_type": int(ge.OpType.ADD),
                    "input_id":  conv_out_id,
                    "param_id":  bias_id,
                    "output_id": last_id,
                    "extra_params": extra
                })

        # 3) Activation (옵션)
        act_type = self._activation_op_type()
        if act_type is not None:
            e_block.append({
                "op_type": act_type,
                "input_id": last_id,
                "param_id": "",
                "output_id": output_id,
                "extra_params": extra
            })
            last_id = output_id  # not used further

        # Shape 매핑 (per-sample)
        shape_map = {
            # 입력은 per-sample (rows=1, cols=Cin*Hin*Win)
            input_id:    Shape(1, int(in_c * in_h * in_w)),
            # 가중치는 (F, Cin*Kh*Kw)
            weight_id:   Shape(int(self.filters), int(in_c * kh * kw)),
            # conv 출력
            conv_out_id: Shape(int(self.filters), int(out_h * out_w)),
            # 최종 출력
            output_id:   Shape(int(self.filters), int(out_h * out_w)),
        }
        if self.activation_name:
            shape_map[preact_id] = Shape(int(self.filters), int(out_h * out_w))
        if self.use_bias and bias_shape is not None:
            shape_map[bias_id] = bias_shape

        # 파라미터 패킹(연속/float32 보장)
        weights = {
            weight_id: cp.ascontiguousarray(
                self.weights.reshape(self.filters, -1).astype(cp.float32)
            )
        }

        return e_block, weights, biases, output_id, shape_map
