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
    - 활성화: "none","relu","gelu","sigmoid","tanh","leakyrelu"
    - capture 경로: forward_into / backward_into (workspace 외부 주입)

    Args:
      out_channels, kernel_size
      stride=(1,1), padding=(0,0), dilation=(1,1), groups=1
      activation="none", with_bias=True
      initializer=("xavier"|"kaiming"|"zeros"), bias_init="zeros"
      save_z_in_fwd=True  # 학습 시 pre-activation을 저장(옵션 A 계약 준수)
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
        """
        옵션 A:
          - act=='none'이면 Z는 Y와 alias (옵스가 내부 처리) -> _last_Z = _last_Y 로 기억
          - act!='none'이고 training & save_z_in_fwd=True 이면 Z 버퍼를 우리가 prealloc 하여 전달
        """
        assert self.W is not None

        # 출력 공간/필요 Z 여부 판단
        N, Cin, H, W = map(int, X.shape)
        KH, KW = self.kernel_size
        H_out, W_out = _out_hw(H, W, KH, KW, self.stride, self.padding, self.dilation)
        need_save_z = bool(self.training and (self.save_z_in_fwd or self.activation == "none"))
        act_is_none = (self.activation == "none")

        Z_buf: Optional[cp.ndarray] = None
        if need_save_z and not act_is_none:
            # 역전파용 Z를 우리가 소유해야 하므로 prealloc
            Z_buf = cp.empty((N, self.out_channels, H_out, W_out), dtype=cp.float32)

        Y = convops.forward(
            X, self.W, self.b,
            stride=self.stride, padding=self.padding, dilation=self.dilation,
            groups=self.groups, with_bias=self.with_bias,
            act=self.activation, leaky_slope=0.01,
            save_z=need_save_z,
            Z_saved=Z_buf,          # act==none이면 None 전달 -> 내부 alias
        )

        if self.training:
            self._last_X = X
            self._last_Y = Y
            if act_is_none:
                # 옵션 A: Z==Y alias
                self._last_Z = Y
            else:
                # 우리가 prealloc한 Z_buf가 채워져 있어야 함
                if need_save_z:
                    self._last_Z = Z_buf
                else:
                    self._last_Z = None  # 사용자가 save_z_in_fwd=False면 역전파 시 에러 유도

        return Y

    def backward(self, gY: cp.ndarray) -> cp.ndarray:
        """
        비캡처 경로: 옵션 A에 따라 Z가 항상 존재해야 학습이 안전.
        - act=='none'이면 Z==Y
        - act!='none'이면 call()에서 Z_buf를 prealloc/전달해야 함
        """
        assert self.W is not None and self._last_X is not None, "call() 이후 backward() 호출"

        act_is_none = (self.activation == "none")
        if act_is_none:
            Z_needed = self._last_Y
        else:
            if self._last_Z is None:
                raise RuntimeError(
                    "[Conv2D.backward] activation != 'none' 인데 Z가 저장되지 않았습니다. "
                    "Conv2D(save_z_in_fwd=True)로 호출하거나, capture 경로를 사용하세요."
                )
            Z_needed = self._last_Z

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
        assert self.W is not None

        act_is_none = (self.activation == "none")
        if (not act_is_none) and (z_out is None):
            raise ValueError("[Conv2D.forward_into] activation != 'none' 에서는 z_out 버퍼가 필요합니다.")

        # --- 공통 shape 계산 ---
        N, Cin, H, W = map(int, X.shape)
        KH, KW = self.kernel_size
        H_out, W_out = _out_hw(H, W, KH, KW, self.stride, self.padding, self.dilation)
        HWo = H_out * W_out
        Cout = int(self.out_channels)
        groups = int(self.groups)
        K = (Cin // groups) * KH * KW

        # --- 옵션 A: act==none 이면 내부적으로 save_z가 강제된다고 보고, WS에도 Z_rows 필요 ---
        effective_save_z = bool(act_is_none or (z_out is not None))

        # --- work 보강: 존재하든 말든 필요한 버퍼가 없으면 즉시 채움 ---
        if work is None:
            work = convops.Conv2DWorkspaces()

        def _need_set(attr, shape):
            arr = getattr(work, attr, None)
            return (arr is None) or (arr.dtype != cp.float32) or (tuple(arr.shape) != tuple(shape))

        # forward WS
        if _need_set("dCol",  (HWo, K)):     work.dCol   = cp.empty((HWo, K),    dtype=cp.float32)
        if _need_set("W_KC",  (K,   Cout)):  work.W_KC   = cp.empty((K,   Cout), dtype=cp.float32)
        if _need_set("Y_tmp", (HWo, Cout)):  work.Y_tmp  = cp.empty((HWo, Cout), dtype=cp.float32)
        if effective_save_z:
            if _need_set("Z_rows", (HWo, Cout)):
                work.Z_rows = cp.empty((HWo, Cout), dtype=cp.float32)
        else:
            # save_z=False 경로에서는 Z_rows가 None이어야 함
            if getattr(work, "Z_rows", None) is not None:
                work.Z_rows = None

        # backward 공통/옵션은 여기서 강제할 필요는 없지만, 미리 만들어두면 검증 패스가 깔끔
        if _need_set("dCol_b",  (HWo, K)):     work.dCol_b  = cp.empty((HWo, K),    dtype=cp.float32)
        if _need_set("dTmp",    (max(Cout*K, HWo*K),)): work.dTmp = cp.empty((max(Cout*K, HWo*K),), dtype=cp.float32)
        if _need_set("gy_rows", (Cout, HWo)):  work.gy_rows = cp.empty((Cout, HWo), dtype=cp.float32)
        if _need_set("Z_rows_b",(Cout, HWo)):  work.Z_rows_b= cp.empty((Cout, HWo), dtype=cp.float32)
        if _need_set("W_CK",    (Cout, K)):    work.W_CK    = cp.empty((Cout, K),   dtype=cp.float32)
        if _need_set("dY_HT",   (HWo,  Cout)): work.dY_HT   = cp.empty((HWo,  Cout),dtype=cp.float32)
        if _need_set("dWpack",  (Cout, K)):    work.dWpack  = cp.empty((Cout, K),   dtype=cp.float32)

        # --- 호출 ---
        convops.forward_into(
            X, self.W,
            out=out,
            B=self.b,
            stride=self.stride, padding=self.padding, dilation=self.dilation,
            groups=self.groups, with_bias=self.with_bias,
            act=self.activation, leaky_slope=0.01,
            save_z=(z_out is not None),  # act==none이면 내부 alias
            Z_saved=z_out,
            stream=stream,
            work=work
        )

        # 역전파용 캐시
        self._cap_X = X
        self._cap_Z = (out if act_is_none else z_out)

    def backward_into(
        self, gY: cp.ndarray, *,
        gA_out: cp.ndarray,
        gW_out: Optional[cp.ndarray] = None,
        gB_out: Optional[cp.ndarray] = None,
        work_dZ: Optional[Any] = None,
        lt_workspace: Optional[Any] = None,
        stream: Optional[int] = None,
        work: Optional[convops.Conv2DWorkspaces] = None,
        Z_saved: Optional[cp.ndarray] = None,
        X_saved: Optional[cp.ndarray] = None,
        W_ref: Optional[cp.ndarray] = None,
    ) -> None:
        # 버퍼 선택
        X_use = X_saved if X_saved is not None else (self._cap_X if self._cap_X is not None else self._last_X)
        Z_use = Z_saved if Z_saved is not None else (self._cap_Z if self._cap_Z is not None else self._last_Z)
        W_use = W_ref  if W_ref  is not None else self.W
        if X_use is None or Z_use is None or W_use is None:
            raise RuntimeError("[Conv2D.backward_into] missing saved X/Z/W buffers from capture plan")

        # shapes
        N, Cin, H, W = map(int, X_use.shape)
        Ny, Cout, Hy, Wy = map(int, gY.shape)
        Nz, Coz, Hz, Wz  = map(int, Z_use.shape)
        if (Ny, Cout, Hy, Wy) != (Nz, Coz, Hz, Wz):
            raise ValueError("[Conv2D.backward_into] gY and Z shapes must match")
        _, CinW, KH, KW = map(int, W_use.shape)
        if CinW != Cin or W_use.shape[0] != Cout:
            raise ValueError("[Conv2D.backward_into] W incompatible with X/gY")

        # C-Contiguous 보정
        if not X_use.flags.c_contiguous: X_use = cp.ascontiguousarray(X_use)
        if not Z_use.flags.c_contiguous: Z_use = cp.ascontiguousarray(Z_use)
        if not W_use.flags.c_contiguous: W_use = cp.ascontiguousarray(W_use)

        # WS 보강 (있든 말든 부족한 것만 채움)
        KH, KW = int(KH), int(KW)
        groups = int(self.groups)
        K   = (Cin // groups) * KH * KW
        HWo = Hy * Wy

        if work is None:
            work = convops.Conv2DWorkspaces()

        def _need_set(attr, shape):
            arr = getattr(work, attr, None)
            return (arr is None) or (arr.dtype != cp.float32) or (tuple(arr.shape) != tuple(shape))

        # backward 공통 필수
        if _need_set("dCol_b",  (HWo, K)):     work.dCol_b  = cp.empty((HWo, K),    dtype=cp.float32)
        if _need_set("dTmp",    (max(Cout*K, HWo*K),)): work.dTmp = cp.empty((max(Cout*K, HWo*K),), dtype=cp.float32)
        if _need_set("gy_rows", (Cout, HWo)):  work.gy_rows = cp.empty((Cout, HWo), dtype=cp.float32)
        if _need_set("Z_rows_b",(Cout, HWo)):  work.Z_rows_b= cp.empty((Cout, HWo), dtype=cp.float32)
        # backward 옵션
        if gA_out is not None:
            if _need_set("W_CK",  (Cout, K)):    work.W_CK  = cp.empty((Cout, K),   dtype=cp.float32)
            if _need_set("dY_HT", (HWo,  Cout)): work.dY_HT = cp.empty((HWo, Cout), dtype=cp.float32)
        else:
            work.W_CK = None; work.dY_HT = None
        if gW_out is not None:
            if _need_set("dWpack",(Cout, K)):    work.dWpack= cp.empty((Cout, K),   dtype=cp.float32)
        else:
            work.dWpack = None

        # (forward WS는 여기서 필수는 아니지만, 일부 커널이 validate 할 수 있어 미리 채워 둠)
        if _need_set("dCol",  (HWo, K)):     work.dCol   = cp.empty((HWo, K),    dtype=cp.float32)
        if _need_set("W_KC",  (K,   Cout)):  work.W_KC   = cp.empty((K,   Cout), dtype=cp.float32)
        if _need_set("Y_tmp", (HWo, Cout)):  work.Y_tmp  = cp.empty((HWo, Cout), dtype=cp.float32)
        if _need_set("Z_rows",(HWo, Cout)):  work.Z_rows = cp.empty((HWo, Cout), dtype=cp.float32)

        # 호출
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
