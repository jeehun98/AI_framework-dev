# python/graph_executor_v2/layers/conv2d.py
from __future__ import annotations
from typing import Optional, Tuple, Iterable, Any, Dict
import cupy as cp

from .base import Layer
from ..ops import conv2d as convops

# shape utils -------------------------------------------------
def _out_hw(H: int, W: int, KH: int, KW: int,
            stride: Tuple[int, int],
            padding: Tuple[int, int],
            dilation: Tuple[int, int]) -> Tuple[int, int]:
    sH, sW = map(int, stride)
    pH, pW = map(int, padding)
    dH, dW = map(int, dilation)
    H_out = (H + 2*pH - dH*(KH-1) - 1)//sH + 1
    W_out = (W + 2*pW - dW*(KW-1) - 1)//sW + 1
    return H_out, W_out


class Conv2D(Layer):
    """
    캡처-호환 Conv2D 레이어 (fused bias/activation).
    - 파라미터: W(Cout,Cin,KH,KW), (opt) b(Cout,)
    - 활성화: ops.conv2d에서 제공하는 fused act 문자열("none","relu","gelu","sigmoid","tanh","leakyrelu")
    - capture 경로: forward_into / backward_into 지원 (workspace 외부 주입)

    Args:
      out_channels, kernel_size
      stride=(1,1), padding=(0,0), dilation=(1,1), groups=1
      activation="none", with_bias=True
      initializer=("xavier"|"kaiming"|"zeros"), bias_init="zeros"
      save_z_in_fwd=True  # 학습 시 pre-activation을 저장(역전파용, 비캡처 경로 대비)
    """
    def __init__(
        self,
        out_channels: int,
        kernel_size: Tuple[int, int] | int,
        *,
        stride: Tuple[int, int] | int = (1, 1),
        padding: Tuple[int, int] | int = (0, 0),
        dilation: Tuple[int, int] | int = (1, 1),
        groups: int = 1,
        activation: str = "none",
        with_bias: bool = True,
        initializer: str = "xavier",
        bias_init: str = "zeros",
        name: Optional[str] = None,
        save_z_in_fwd: bool = True,
    ):
        super().__init__(name=name)
        # 하이퍼파라미터
        self.out_channels = int(out_channels)
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):   stride   = (stride, stride)
        if isinstance(padding, int):  padding  = (padding, padding)
        if isinstance(dilation, int): dilation = (dilation, dilation)

        self.kernel_size = tuple(map(int, kernel_size))  # (KH,KW)
        self.stride      = tuple(map(int, stride))
        self.padding     = tuple(map(int, padding))
        self.dilation    = tuple(map(int, dilation))
        self.groups      = int(groups)
        self.activation  = str(activation or "none").lower()
        self.with_bias   = bool(with_bias)
        self.initializer = initializer
        self.bias_init   = bias_init
        self.save_z_in_fwd = bool(save_z_in_fwd)

        # 파라미터/그라드
        self.W: Optional[cp.ndarray] = None
        self.b: Optional[cp.ndarray] = None
        self.dW: Optional[cp.ndarray] = None
        self.db: Optional[cp.ndarray] = None

        # 비캡처 경로에서 사용할 캐시
        self._last_X: Optional[cp.ndarray] = None          # (N,Cin,H,W)
        self._last_Z: Optional[cp.ndarray] = None          # (N,Cout,H_out,W_out) pre-activation
        self._last_Y: Optional[cp.ndarray] = None          # (N,Cout,H_out,W_out)

        # 캡처 경로 캐시(그래프 캡처 동안 유효)
        self._cap_X: Optional[cp.ndarray] = None
        self._cap_Z: Optional[cp.ndarray] = None

        # shape
        self.input_shape: Optional[Tuple[int, ...]] = None
        self.output_shape: Optional[Tuple[int, ...]] = None
        self.built: bool = False

        self.training: bool = True

    # --------- init helpers ----------
    def _init_weights(self, Cin: int):
        KH, KW = self.kernel_size
        Cout = self.out_channels
        W = cp.empty((Cout, Cin, KH, KW), dtype=cp.float32)
        if self.initializer.lower() in ("xavier", "glorot", "xavier_uniform"):
            limit = (6.0 / (Cin*KH*KW + Cout*KH*KW)) ** 0.5
            W[...] = cp.random.uniform(-limit, limit, size=W.shape).astype(cp.float32)
        elif self.initializer.lower() in ("kaiming", "he", "he_normal"):
            std = (2.0 / (Cin*KH*KW)) ** 0.5
            W[...] = std * cp.random.randn(*W.shape).astype(cp.float32)
        elif self.initializer.lower() in ("zeros", "zero"):
            W.fill(0)
        else:
            raise ValueError(f"unknown initializer: {self.initializer}")
        self.W = W

        if self.with_bias:
            b = cp.empty((Cout,), dtype=cp.float32)
            if self.bias_init.lower() in ("zeros", "zero"):
                b.fill(0)
            else:
                b[...] = cp.random.randn(Cout).astype(cp.float32) * 1e-3
            self.b = b
        else:
            self.b = None

        # grad buffers (비캡처 경로에서만 사용)
        self.dW = cp.zeros_like(self.W) if self.W is not None else None
        self.db = cp.zeros_like(self.b) if self.b is not None else None

    # --------- Layer API -------------
    def build(self, input_shape: Tuple[int, ...]) -> None:
        """
        input_shape: (N, Cin, H, W)
        """
        N, Cin, H, W = map(int, input_shape)
        if self.groups < 1 or Cin % self.groups != 0:
            raise ValueError(f"Cin({Cin}) % groups({self.groups}) != 0")
        self._init_weights(Cin)

        H_out, W_out = _out_hw(H, W, self.kernel_size[0], self.kernel_size[1],
                               self.stride, self.padding, self.dilation)
        self.input_shape  = (N, Cin, H, W)
        self.output_shape = (N, self.out_channels, H_out, W_out)
        self.built = True

    def compute_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        _, Cin, H, W = map(int, input_shape)
        if self.groups < 1 or Cin % self.groups != 0:
            raise ValueError(f"Cin({Cin}) % groups({self.groups}) != 0")
        H_out, W_out = _out_hw(H, W, self.kernel_size[0], self.kernel_size[1],
                               self.stride, self.padding, self.dilation)
        return (int(input_shape[0]), self.out_channels, H_out, W_out)

    def parameters(self) -> Iterable[Tuple[Any, Any, str]]:
        if self.W is not None:
            yield (self.W, self.dW, f"{self.name or 'Conv2D'}.W")
        if self.b is not None:
            yield (self.b, self.db, f"{self.name or 'Conv2D'}.b")

    # --------- runtime (eager) ----------
    def call(self, X: cp.ndarray) -> cp.ndarray:
        assert self.W is not None
        Y = convops.forward(
            X, self.W, self.b,
            stride=self.stride, padding=self.padding, dilation=self.dilation,
            groups=self.groups, with_bias=self.with_bias,
            act=self.activation, leaky_slope=0.01,
            save_z=self.training and self.save_z_in_fwd,
        )
        if self.training:
            self._last_X = X
            self._last_Y = Y
            # 비캡처 경로에선 별도 Z 버퍼를 ops에서 외부로 받지 않으므로,
            # 활성화가 'none'이면 out을 Z로 alias 하여 사용.
            self._last_Z = Y if self.activation == "none" else self._last_Z
        return Y

    def backward(self, gY: cp.ndarray) -> cp.ndarray:
        assert self.W is not None and self._last_X is not None, "call() 이후 backward() 호출"
        # 비캡처: ops가 내부에서 WS를 즉시 할당
        Z_needed = self._last_Y if (self.activation == "none") else self._last_Z
        if Z_needed is None:
            # save_z 옵션 없이 비캡처를 쓴다면 활성화 미분을 수동 구현해야 하나,
            # 간이 버전으로 활성화 없음으로 간주
            Z_needed = self._last_Y
        outs = convops.backward(
            self._last_X, self.W, gY, Z_needed,
            stride=self.stride, padding=self.padding, dilation=self.dilation,
            groups=self.groups, with_bias=self.with_bias, act=self.activation, leaky_slope=0.01,
            want_gX=True, want_gW=True, want_gB=self.with_bias
        )
        if self.dW is None: self.dW = cp.zeros_like(self.W)
        self.dW[...] = outs["gW"]
        if self.with_bias:
            if self.db is None: self.db = cp.zeros_like(self.b)
            self.db[...] = outs["gB"]
        return outs["gX"]

    # --------- capture path ----------
    def forward_into(
        self, X: cp.ndarray, *,
        out: cp.ndarray,
        z_out: Optional[cp.ndarray] = None,
        stream: Optional[int] = None,
        work: Optional[convops.Conv2DWorkspaces] = None,
    ) -> None:
        """
        캡처 안전: 모든 출력/워크스페이스는 외부에서 prealloc.
        - out: (N,Cout,H_out,W_out) 미리 할당
        - z_out: pre-activation 버퍼(활성화/역전파용). act='none'이면 out과 alias 가능.
        - work: Conv2DWorkspaces
        """
        assert self.W is not None
        convops.forward_into(
            X, self.W,
            out=out,
            B=self.b,
            stride=self.stride, padding=self.padding, dilation=self.dilation,
            groups=self.groups, with_bias=self.with_bias,
            act=self.activation, leaky_slope=0.01,
            save_z=(z_out is not None),
            Z_saved=z_out,
            stream=stream,
            work=work
        )
        # ✅ 역전파용 캡처 캐시 저장 (z_out이 없으면 out을 Z로 alias)
        self._cap_X = X
        self._cap_Z = z_out if z_out is not None else out

    def backward_into(
        self, gY: cp.ndarray, *,
        gA_out: cp.ndarray,
        gW_out: Optional[cp.ndarray] = None,
        gB_out: Optional[cp.ndarray] = None,
        work_dZ: Optional[Any] = None,          # Dense 인터페이스와 시그니처 호환(미사용)
        lt_workspace: Optional[Any] = None,     # Dense 인터페이스와 시그니처 호환(미사용)
        stream: Optional[int] = None,
        work: Optional[convops.Conv2DWorkspaces] = None,
        # (선택) 캡처 플랜에서 명시 전달 가능; 없으면 forward_into 캐시 사용
        Z_saved: Optional[cp.ndarray] = None,
        X_saved: Optional[cp.ndarray] = None,
        W_ref: Optional[cp.ndarray] = None,
    ) -> None:
        """
        캡처 안전: gA_out/gW_out/gB_out/WS 모두 외부에서 준비.
        - X_saved / Z_saved 는 캡처 플랜이 관리하는 고정 버퍼를 전달하면 우선 사용
          (없으면 forward_into에서 저장한 self._cap_X / self._cap_Z 사용)
        """
        # 우선순위: 명시 전달 > 캡처 캐시 > (비권장) eager 캐시
        X_use = X_saved if X_saved is not None else (self._cap_X if self._cap_X is not None else self._last_X)
        Z_use = Z_saved if Z_saved is not None else (self._cap_Z if self._cap_Z is not None else self._last_Y)
        W_use = W_ref  if W_ref  is not None else self.W

        if X_use is None or Z_use is None or W_use is None:
            raise RuntimeError("[Conv2D.backward_into] missing saved X/Z/W buffers from capture plan")

        convops.backward_into(
            X_use, W_use, gY, Z_use,
            stride=self.stride, padding=self.padding, dilation=self.dilation,
            groups=self.groups, with_bias=self.with_bias,
            act=self.activation, leaky_slope=0.01,
            gX_out=gA_out, gW_out=gW_out, gB_out=gB_out,
            stream=stream, work=work
        )

    # --------- misc ----------
    def zero_grad(self):
        if self.dW is not None: self.dW[...] = 0
        if self.db is not None: self.db[...] = 0

    def state_dict(self) -> Dict[str, Any]:
        return {
            "W": self.W, "b": self.b,
            "stride": self.stride, "padding": self.padding,
            "dilation": self.dilation, "groups": self.groups,
            "activation": self.activation, "with_bias": self.with_bias,
            "kernel_size": self.kernel_size,
        }

    def load_state_dict(self, sd: Dict[str, Any]):
        for k in ("W","b"):
            if k in sd and sd[k] is not None:
                if getattr(self, k) is None:
                    setattr(self, k, sd[k].copy())
                else:
                    getattr(self, k)[...] = sd[k]
        # 하이퍼파라미터 재적용(선택)
        for k in ("stride","padding","dilation","groups","activation","with_bias","kernel_size"):
            if k in sd:
                setattr(self, k, sd[k])
        # grad 버퍼 재생성
        self.dW = cp.zeros_like(self.W) if self.W is not None else None
        self.db = cp.zeros_like(self.b) if self.b is not None else None
        return self
