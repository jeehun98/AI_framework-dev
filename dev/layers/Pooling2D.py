# dev/layers/Pooling2D.py
import cupy as cp
from dev.layers.layer import Layer

import sys
sys.path.append("C:/Users/owner/Desktop/AI_framework-dev/dev/backend/graph_executor/test")
import graph_executor as ge  # PyBind11 모듈

Shape = ge.Shape
OpExtraParams = ge.OpExtraParams

def _compute_out_hw(H, W, kH, kW, sH, sW, pH, pW, dH=1, dW=1):
    Hout = (H + 2*pH - dH*(kH - 1) - 1) // sH + 1
    Wout = (W + 2*pW - dW*(kW - 1) - 1) // sW + 1
    if Hout <= 0 or Wout <= 0:
        raise ValueError(f"[Pool2D] invalid output size: Hout={Hout}, Wout={Wout} "
                         f"(H={H},W={W}, k={kH}x{kW}, s={sH}x{sW}, p={pH}x{pW}, d={dH}x{dW})")
    return int(Hout), int(Wout)


class _BasePool2D(Layer):
    def __init__(self,
                 pool_type: str,         # "max" or "avg"
                 kernel_size=(2, 2),
                 strides=(2, 2),
                 padding='valid',        # 'valid' | 'same' | (pH, pW)
                 dilation=(1, 1),
                 count_include_pad=False,  # only for avg
                 input_shape=None,       # NCHW: (N, C, H, W)
                 name=None,
                 **kwargs):
        super().__init__(name=name, **kwargs)
        self.layer_name = f"{pool_type}pool2d"
        self.pool_type = pool_type
        self.kernel_size = tuple(kernel_size)
        self.strides = tuple(strides)
        self.dilation = tuple(dilation)
        self.count_include_pad = bool(count_include_pad)
        self.input_shape = input_shape  # NCHW
        self.output_shape = None
        self.name = name or f"{pool_type}pool2d_{id(self)}"
        self.output_var = f"{self.name}_out"

        # 내부 캐시(선택)
        self.last_input = None

        # padding 전처리
        if isinstance(padding, str):
            padding = padding.lower()
            if padding not in ("valid", "same"):
                raise ValueError("padding must be 'valid' or 'same' or (pH,pW)")
            self.padding_mode = padding
            self.padding = None
        else:
            # tuple 형태
            self.padding_mode = "custom"
            self.padding = tuple(padding)

    def build(self, input_shape):
        """input_shape: NCHW (N, C, H, W)"""
        if len(input_shape) != 4:
            raise ValueError(f"[{self.layer_name}] expects NCHW 4D input, got {input_shape}")
        self.input_shape = tuple(map(int, input_shape))
        N, C, H, W = self.input_shape
        kH, kW = self.kernel_size
        sH, sW = self.strides
        dH, dW = self.dilation

        # padding 계산
        if self.padding_mode == "same":
            # SAME padding (ceil mode 없이 표준 conv 출력공식에 맞춘 패딩)
            # out = ceil(H / s), 여기선 conv와 동일한 공식 사용을 위해 역산
            # 여기 프레임워크의 풀링 출력식과 맞추려면 아래 패딩이 일반적으로 사용됨:
            # total_pad_h = max(0, (H - 1)*sH + dH*(kH - 1) + 1 - H)
            # total_pad_w = max(0, (W - 1)*sW + dW*(kW - 1) + 1 - W)
            total_pad_h = max(0, (H - 1)*sH + dH*(kH - 1) + 1 - H)
            total_pad_w = max(0, (W - 1)*sW + dW*(kW - 1) + 1 - W)
            pH = total_pad_h // 2
            pW = total_pad_w // 2
        elif self.padding_mode == "valid":
            pH = pW = 0
        else:
            pH, pW = self.padding

        Hout, Wout = _compute_out_hw(H, W, kH, kW, sH, sW, pH, pW, dH, dW)
        self.output_shape = (N, C, Hout, Wout)
        self.built = True

    def __call__(self, x):
        if not self.built:
            self.build(x.shape)
        return self.call(x)

    def call(self, x):
        """그래프 실행기를 쓸 땐 호출되지 않도록 하는 게 안전하지만,
        디버깅용으로 로컬 참조 풀링을 넣을 수도 있다. 여기선 패스."""
        self.last_input = x
        return x  # 그래프 경로 사용 시 실제 값은 엔진이 채움

    def compute_output_shape(self, input_shape):
        if input_shape is None or len(input_shape) != 4:
            raise ValueError(f"[{self.layer_name}] expects 4D (N,C,H,W), got {input_shape}")
        N, C, H, W = map(int, input_shape)
        if not self.built:
            # padding 재계산
            kH, kW = self.kernel_size
            sH, sW = self.strides
            dH, dW = self.dilation
            if self.padding_mode == "same":
                total_pad_h = max(0, (H - 1)*sH + dH*(kH - 1) + 1 - H)
                total_pad_w = max(0, (W - 1)*sW + dW*(kW - 1) + 1 - W)
                pH = total_pad_h // 2
                pW = total_pad_w // 2
            elif self.padding_mode == "valid":
                pH = pW = 0
            else:
                pH, pW = self.padding
            Hout, Wout = _compute_out_hw(H, W, kH, kW, sH, sW, pH, pW, dH, dW)
            return (N, C, Hout, Wout)
        return self.output_shape

    def to_e_matrix(self, input_id):
        if self.input_shape is None:
            raise ValueError(f"[{self.layer_name}] input_shape is None. Did you forget to call build()?")

        N, C, H, W = map(int, self.input_shape)
        kH, kW = self.kernel_size
        sH, sW = self.strides
        dH, dW = self.dilation

        # padding 계산(빌드 시점과 동일)
        if self.padding_mode == "same":
            total_pad_h = max(0, (H - 1)*sH + dH*(kH - 1) + 1 - H)
            total_pad_w = max(0, (W - 1)*sW + dW*(kW - 1) + 1 - W)
            pH = total_pad_h // 2
            pW = total_pad_w // 2
        elif self.padding_mode == "valid":
            pH = pW = 0
        else:
            pH, pW = self.padding

        Hout, Wout = _compute_out_hw(H, W, kH, kW, sH, sW, pH, pW, dH, dW)

        # OpExtraParams 채우기 (NCHW 메타)
        ex = OpExtraParams()
        ex.input_c  = C
        ex.input_h  = H
        ex.input_w  = W
        ex.output_c = C
        ex.kernel_h = kH
        ex.kernel_w = kW
        ex.stride_h = sH
        ex.stride_w = sW
        ex.padding_h = pH
        ex.padding_w = pW
        ex.dilation_h = dH
        ex.dilation_w = dW
        ex.count_include_pad = bool(self.count_include_pad)
        ex.batch_size = N

        output_id = self.output_var
        op_type = ge.OpType.POOL_MAX if self.pool_type == "max" else ge.OpType.POOL_AVG
        e_block = [
            {"op_type": int(op_type), "input_id": input_id, "param_id": "",
             "output_id": output_id, "extra_params": ex}
        ]

        # 그래프 실행기는 sample-view로 (rows=C, cols=H*W)를 사용
        shape_map = {
            input_id:  Shape(C, H*W),
            output_id: Shape(C, Hout*Wout),
        }

        weights = {}  # 풀링은 파라미터 없음
        biases  = {}
        return e_block, weights, biases, output_id, shape_map


class MaxPool2D(_BasePool2D):
    def __init__(self, kernel_size=(2,2), strides=(2,2), padding='valid',
                 dilation=(1,1), input_shape=None, name=None, **kwargs):
        super().__init__(pool_type="max",
                         kernel_size=kernel_size,
                         strides=strides,
                         padding=padding,
                         dilation=dilation,
                         count_include_pad=False,  # maxpool엔 의미 없음
                         input_shape=input_shape,
                         name=name,
                         **kwargs)


class AvgPool2D(_BasePool2D):
    def __init__(self, kernel_size=(2,2), strides=(2,2), padding='valid',
                 dilation=(1,1), count_include_pad=False,
                 input_shape=None, name=None, **kwargs):
        super().__init__(pool_type="avg",
                         kernel_size=kernel_size,
                         strides=strides,
                         padding=padding,
                         dilation=dilation,
                         count_include_pad=count_include_pad,
                         input_shape=input_shape,
                         name=name,
                         **kwargs)
