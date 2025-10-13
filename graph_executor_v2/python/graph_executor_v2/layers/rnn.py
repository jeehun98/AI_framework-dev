# python/graph_executor_v2/layers/rnn.py
from __future__ import annotations
from typing import Optional, Tuple, Iterable, Any, Dict
import cupy as cp

from .base import Layer
from ..ops import rnn as rnnops


class RNN(Layer):
    """
    캡처-호환 RNN(Elman) 레이어 (fused bias/activation).
    - 파라미터: Wx(I,H), Wh(H,H), (opt) b(H,)
    - 활성화: "none","relu","gelu","sigmoid","tanh","leakyrelu"
    - capture 경로: forward_into / backward_into (workspace 외부 주입)

    Args:
      hidden_size: H
      activation="tanh", with_bias=True
      initializer=("xavier"|"kaiming"|"zeros"), bias_init="zeros"
      save_z_in_fwd=True  # 학습 시 pre-activation을 저장(옵션 A 계약 준수)
    """
    def __init__(
        self,
        hidden_size: int,
        *,
        activation: str = "tanh",
        with_bias: bool = True,
        initializer: str = "xavier",
        bias_init: str = "zeros",
        name: Optional[str] = None,
        save_z_in_fwd: bool = True,
    ):
        super().__init__(name=name)
        # 하이퍼파라미터
        self.hidden_size  = int(hidden_size)
        self.activation   = str(activation or "tanh").lower()
        self.with_bias    = bool(with_bias)
        self.initializer  = initializer
        self.bias_init    = bias_init
        self.save_z_in_fwd = bool(save_z_in_fwd)

        # 파라미터/그라드
        self.Wx: Optional[cp.ndarray] = None  # (I,H)
        self.Wh: Optional[cp.ndarray] = None  # (H,H)
        self.b:  Optional[cp.ndarray] = None  # (H,)
        self.dWx: Optional[cp.ndarray] = None
        self.dWh: Optional[cp.ndarray] = None
        self.db:  Optional[cp.ndarray] = None

        # 비캡처 경로 캐시
        self._last_X: Optional[cp.ndarray] = None    # (N,T,I)
        self._last_h0: Optional[cp.ndarray] = None   # (N,H)
        self._last_Y: Optional[cp.ndarray] = None    # (N,T,H)
        self._last_Z: Optional[cp.ndarray] = None    # (N,T,H) pre-activation

        # 캡처 경로 캐시
        self._cap_X: Optional[cp.ndarray] = None
        self._cap_h0: Optional[cp.ndarray] = None
        self._cap_Z: Optional[cp.ndarray] = None

        # shape
        self.input_shape: Optional[Tuple[int, ...]] = None   # (N,T,I)
        self.output_shape: Optional[Tuple[int, ...]] = None  # (N,T,H)
        self.built: bool = False
        self.training: bool = True

    # --------- init helpers ----------
    def _init_weights(self, I: int):
        H = self.hidden_size

        def _init_matrix(shape):
            if self.initializer.lower() in ("xavier", "glorot", "xavier_uniform"):
                fan_in, fan_out = shape[0], shape[1]
                limit = (6.0 / (fan_in + fan_out)) ** 0.5
                return cp.random.uniform(-limit, limit, size=shape).astype(cp.float32)
            elif self.initializer.lower() in ("kaiming", "he", "he_normal"):
                std = (2.0 / shape[0]) ** 0.5
                return (std * cp.random.randn(*shape)).astype(cp.float32)
            elif self.initializer.lower() in ("zeros", "zero"):
                return cp.zeros(shape, dtype=cp.float32)
            else:
                raise ValueError(f"unknown initializer: {self.initializer}")

        self.Wx = _init_matrix((I, H))
        self.Wh = _init_matrix((H, H))

        if self.with_bias:
            self.b = cp.empty((H,), dtype=cp.float32)
            if self.bias_init.lower() in ("zeros", "zero"):
                self.b.fill(0)
            else:
                self.b[...] = cp.random.randn(H).astype(cp.float32) * 1e-3
        else:
            self.b = None

        # grad buffers (비캡처 경로에서만 사용)
        self.dWx = cp.zeros_like(self.Wx) if self.Wx is not None else None
        self.dWh = cp.zeros_like(self.Wh) if self.Wh is not None else None
        self.db  = cp.zeros_like(self.b)  if self.b  is not None else None

    # --------- Layer API -------------
    def build(self, input_shape: Tuple[int, ...]) -> None:
        """
        input_shape: (N, T, I)
        """
        N, T, I = map(int, input_shape)
        if I <= 0 or self.hidden_size <= 0:
            raise ValueError("I/H must be > 0")
        self._init_weights(I)

        self.input_shape  = (N, T, I)
        self.output_shape = (N, T, self.hidden_size)
        self.built = True

    def compute_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        N, T, I = map(int, input_shape)
        return (N, T, self.hidden_size)

    def parameters(self) -> Iterable[Tuple[Any, Any, str]]:
        if self.Wx is not None:
            yield (self.Wx, self.dWx, f"{self.name or 'RNN'}.Wx")
        if self.Wh is not None:
            yield (self.Wh, self.dWh, f"{self.name or 'RNN'}.Wh")
        if self.b is not None:
            yield (self.b,  self.db,  f"{self.name or 'RNN'}.b")

    # --------- runtime (eager) ----------
    def call(self, X: cp.ndarray, h0: Optional[cp.ndarray] = None) -> cp.ndarray:
        """
        옵션 A:
          - act=='none'이면 Z는 Y와 alias (ops가 내부 처리) -> _last_Z = _last_Y 로 기억
          - act!='none'이고 training & save_z_in_fwd=True 이면 Z 버퍼를 우리가 prealloc 하여 전달
        """
        assert self.Wx is not None and self.Wh is not None

        N, T, I = map(int, X.shape)
        H = int(self.hidden_size)
        if h0 is None:
            h0 = cp.zeros((N, H), dtype=cp.float32)
        else:
            if h0.dtype != cp.float32 or h0.shape != (N, H):
                raise ValueError(f"h0 must be float32[{(N,H)}]")

        need_save_z = bool(self.training and (self.save_z_in_fwd or self.activation == "none"))
        act_is_none = (self.activation == "none")

        Z_buf: Optional[cp.ndarray] = None
        if need_save_z and not act_is_none:
            Z_buf = cp.empty((N, T, H), dtype=cp.float32)

        Y = rnnops.forward(
            X, self.Wx, self.Wh, h0, self.b,
            with_bias=self.with_bias,
            act=self.activation, leaky_slope=0.01,
            save_z=need_save_z,
            Z_saved=Z_buf,   # act==none이면 None 전달 -> 내부 alias
        )

        if self.training:
            self._last_X  = X
            self._last_h0 = h0
            self._last_Y  = Y
            if act_is_none:
                self._last_Z = Y
            else:
                self._last_Z = Z_buf if need_save_z else None

        return Y

    def backward(
        self,
        dY: cp.ndarray,
        *,
        want_dX: bool = True,
        want_dWx: bool = True,
        want_dWh: bool = True,
        want_dB: bool = False,
        want_dh0: bool = True,
    ) -> Dict[str, cp.ndarray]:
        """
        비캡처 경로: 옵션 A에 따라 Z가 항상 존재해야 학습이 안전.
        - act=='none'이면 Z==Y
        - act!='none'이면 call()에서 Z_buf를 prealloc/전달해야 함
        """
        assert self.Wx is not None and self.Wh is not None
        assert self._last_X is not None and self._last_h0 is not None and self._last_Y is not None, \
            "call() 이후 backward() 호출"

        act_is_none = (self.activation == "none")
        if act_is_none:
            Z_needed = self._last_Y
        else:
            if self._last_Z is None:
                raise RuntimeError(
                    "[RNN.backward] activation != 'none' 인데 Z가 저장되지 않았습니다. "
                    "RNN(save_z_in_fwd=True)로 호출하거나, capture 경로를 사용하세요."
                )
            Z_needed = self._last_Z

        outs = rnnops.backward(
            self._last_X, self.Wx, self.Wh, self._last_h0,
            dY, Z_needed,
            with_bias=self.with_bias, act=self.activation, leaky_slope=0.01,
            want_dX=want_dX, want_dWx=want_dWx, want_dWh=want_dWh,
            want_dB=want_dB, want_dh0=want_dh0
        )

        # grads 채우기 (요청된 항목만)
        if want_dWx:
            if self.dWx is None: self.dWx = cp.zeros_like(self.Wx)
            self.dWx[...] = outs["dWx"]
        if want_dWh:
            if self.dWh is None: self.dWh = cp.zeros_like(self.Wh)
            self.dWh[...] = outs["dWh"]
        if want_dB and self.with_bias:
            if self.db is None: self.db = cp.zeros_like(self.b)
            self.db[...] = outs["dB"]

        ret: Dict[str, cp.ndarray] = {}
        if want_dX:   ret["dX"]   = outs["dX"]
        if want_dh0:  ret["dh0"]  = outs["dh0"]
        if want_dWx:  ret["dWx"]  = outs["dWx"]
        if want_dWh:  ret["dWh"]  = outs["dWh"]
        if want_dB and self.with_bias: ret["dB"] = outs["dB"]
        return ret

    # --------- capture path ----------
    def forward_into(
        self, X: cp.ndarray, *, h0: Optional[cp.ndarray] = None,
        out: cp.ndarray, z_out: Optional[cp.ndarray] = None,
        stream: Optional[int] = None,
        work: Optional[rnnops.RnnWorkspaces] = None,
    ) -> None:
        assert self.Wx is not None and self.Wh is not None

        N, T, I = map(int, X.shape)
        H = int(self.hidden_size)
        if h0 is None:
            h0 = cp.zeros((N, H), dtype=cp.float32)
        else:
            if h0.dtype != cp.float32 or h0.shape != (N, H):
                raise ValueError(f"h0 must be float32[{(N,H)}]")

        if out.dtype != cp.float32 or out.shape != (N, T, H):
            raise ValueError(f"[RNN.forward_into] `out` must be float32[{(N,T,H)}]")

        act_is_none = (self.activation == "none")
        if (not act_is_none) and (z_out is None):
            raise ValueError("[RNN.forward_into] activation != 'none' 에서는 z_out 버퍼가 필요합니다.")

        effective_save_z = bool(act_is_none or (z_out is not None))

        # work 없거나 부족하면 보강
        if work is None:
            work = rnnops.RnnWorkspaces()

        def _need_set(attr, shape):
            arr = getattr(work, attr, None)
            return (arr is None) or (arr.dtype != cp.float32) or (tuple(arr.shape) != tuple(shape))

        if _need_set("XH_cat", (N, I+H)): work.XH_cat = cp.empty((N, I+H), dtype=cp.float32)
        if _need_set("Y_rows", (N, H)):   work.Y_rows = cp.empty((N, H), dtype=cp.float32)
        if _need_set("W_cat",  (I+H, H)): work.W_cat  = cp.empty((I+H, H), dtype=cp.float32)
        if effective_save_z:
            if _need_set("Z_rows_f", (N, H)): work.Z_rows_f = cp.empty((N, H), dtype=cp.float32)
        else:
            work.Z_rows_f = None

        # backward용 공통 버퍼도 미리 채워두면 다음 단계에서 검증 깔끔
        if _need_set("XH_cat_b", (N, I+H)): work.XH_cat_b = cp.empty((N, I+H), dtype=cp.float32)
        if _need_set("G_rows",   (N, H)):   work.G_rows   = cp.empty((N, H),   dtype=cp.float32)
        if _need_set("Z_rows_b", (N, H)):   work.Z_rows_b = cp.empty((N, H),   dtype=cp.float32)
        if _need_set("W_cat_b",  (I+H, H)): work.W_cat_b  = cp.empty((I+H, H), dtype=cp.float32)
        if _need_set("dXH_cat",  (N, I+H)): work.dXH_cat  = cp.empty((N, I+H), dtype=cp.float32)
        if _need_set("dWcat",    (I+H, H)): work.dWcat    = cp.empty((I+H, H), dtype=cp.float32)
        if _need_set("TmpW",     (I+H, H)): work.TmpW     = cp.empty((I+H, H), dtype=cp.float32)

        rnnops.forward_into(
            X, self.Wx, self.Wh, h0,
            out=out,
            B=self.b,
            with_bias=self.with_bias,
            act=self.activation, leaky_slope=0.01,
            save_z=(z_out is not None),  # act==none이면 내부 alias
            Z_saved=z_out,
            stream=stream,
            work=work
        )

        self._cap_X  = X
        self._cap_h0 = h0
        self._cap_Z  = (out if act_is_none else z_out)

    def backward_into(
        self, dY: cp.ndarray, *,
        # === graph_exec 호출과 호환되는 표준 이름들 ===
        gA_out: Optional[cp.ndarray] = None,      # <- dX_out에 해당 (필수일 수도 있음)
        gW_out: Optional[cp.ndarray] = None,      # 선택: (I+H, H) 등으로 받길 원하면 사용
        gB_out: Optional[cp.ndarray] = None,      # 선택: (H,)
        # --- 추가(무시 가능) ---
        work_dZ: Optional[Any] = None,
        lt_workspace: Optional[Any] = None,
        stream: Optional[int] = None,
        work: Optional[rnnops.RnnWorkspaces] = None,
        # 저장 텐서/레퍼런스(있으면 우선 사용)
        Z_saved: Optional[cp.ndarray] = None,
        X_saved: Optional[cp.ndarray] = None,
        h0_saved: Optional[cp.ndarray] = None,
        Wx_ref: Optional[cp.ndarray] = None,
        Wh_ref: Optional[cp.ndarray] = None,
    ) -> None:
        """
        CUDA Graph 캡처 경로 역전파.
        - graph_exec 표준 인자명(gA_out/gW_out/gB_out)에 맞춤
        - 내부 파라미터 grad(self.dWx/self.dWh/self.db)에 기록
        - gW_out이 주어지면 원한다면 dWcat을 결합해 채워줄 수도 있음(현재는 선택적으로 무시)
        """
        assert self.Wx is not None and self.Wh is not None

        # 버퍼 선택
        X_use  = X_saved  if X_saved  is not None else (self._cap_X  if self._cap_X  is not None else self._last_X)
        h0_use = h0_saved if h0_saved is not None else (self._cap_h0 if self._cap_h0 is not None else self._last_h0)
        Z_use  = Z_saved  if Z_saved  is not None else (self._cap_Z  if self._cap_Z  is not None else self._last_Z)
        Wx_use = Wx_ref   if Wx_ref   is not None else self.Wx
        Wh_use = Wh_ref   if Wh_ref   is not None else self.Wh
        if X_use is None or h0_use is None or Z_use is None or Wx_use is None or Wh_use is None:
            raise RuntimeError("[RNN.backward_into] missing saved X/h0/Z/Wx/Wh buffers from capture plan")

        # 출력 버퍼 체크 (dX)
        if gA_out is not None and (gA_out.dtype != cp.float32 or not gA_out.flags.c_contiguous or gA_out.shape != X_use.shape):
            raise ValueError(f"[capture] gA_out must be float32{X_use.shape} and C-contiguous")

        # 선택 출력 버퍼 체크 (gB)
        if gB_out is not None:
            if not self.with_bias:
                raise ValueError("[capture] gB_out must be None when with_bias=False")
            if gB_out.dtype != cp.float32 or gB_out.ndim != 1 or gB_out.size != self.hidden_size:
                raise ValueError(f"[capture] gB_out must be float32[{self.hidden_size}]")

        # work 확보/보강 (필요 시 채움)
        if work is None:
            work = rnnops.RnnWorkspaces()
        N, T, I = map(int, X_use.shape)
        H = int(self.hidden_size)

        def _need_set(arr, shape):
            return (arr is None) or (arr.dtype != cp.float32) or (tuple(arr.shape) != tuple(shape))

        if _need_set(work.XH_cat_b, (N, I+H)): work.XH_cat_b = cp.empty((N, I+H), dtype=cp.float32)
        if _need_set(work.G_rows,   (N, H)):   work.G_rows   = cp.empty((N, H),   dtype=cp.float32)
        if _need_set(work.Z_rows_b, (N, H)):   work.Z_rows_b = cp.empty((N, H),   dtype=cp.float32)
        if _need_set(work.W_cat_b,  (I+H, H)): work.W_cat_b  = cp.empty((I+H, H), dtype=cp.float32)
        if _need_set(work.dXH_cat,  (N, I+H)): work.dXH_cat  = cp.empty((N, I+H), dtype=cp.float32)
        if _need_set(work.dWcat,    (I+H, H)): work.dWcat    = cp.empty((I+H, H), dtype=cp.float32)
        if _need_set(work.TmpW,     (I+H, H)): work.TmpW     = cp.empty((I+H, H), dtype=cp.float32)

        # rnnops 호출 (옵스는 dX_out, dWx_out, dWh_out, dB_out, dh0_out 서명)
        outs = rnnops.backward(
            X_use, Wx_use, Wh_use, h0_use,
            dY, Z_use,
            with_bias=self.with_bias, act=self.activation, leaky_slope=0.01,
            want_dX=(gA_out is not None), want_dWx=True, want_dWh=True,
            want_dB=(gB_out is not None), want_dh0=True,
        )

        # dX (필요 시 출력 버퍼에)
        if gA_out is not None:
            gA_out[...] = outs["dX"]

        # 내부 파라미터 grad 갱신 (옵티마가 self.dWx/self.dWh/self.db를 참조)
        if self.dWx is None: self.dWx = cp.zeros_like(self.Wx)
        if self.dWh is None: self.dWh = cp.zeros_like(self.Wh)
        self.dWx[...] = outs["dWx"]
        self.dWh[...] = outs["dWh"]
        if self.with_bias:
            if self.db is None: self.db = cp.zeros_like(self.b)
            if "dB" in outs and outs["dB"] is not None:
                self.db[...] = outs["dB"]
                if gB_out is not None:
                    gB_out[...] = outs["dB"]

        # (선택) gW_out을 요청받았다면, 결합된 dWcat으로 채워주고 싶으면 아래 주석을 해제하세요.
        # if gW_out is not None:
        #     # dWcat = concat([dWx; dWh], axis=0)  -> shape (I+H, H)
        #     if gW_out.shape != (I+H, H):
        #         raise ValueError(f"[capture] gW_out must be float32[{(I+H, H)}]")
        #     gW_out[:I,   :] = self.dWx
        #     gW_out[I:,   :] = self.dWh

        # dh0는 레이어 외부에서 필요 시 사용할 수 있으므로 보관 (선택)
        self._last_dh0 = outs.get("dh0", None)


    # --------- misc ----------
    def zero_grad(self):
        if self.dWx is not None: self.dWx[...] = 0
        if self.dWh is not None: self.dWh[...] = 0
        if self.db  is not None: self.db[...]  = 0

    def state_dict(self) -> Dict[str, Any]:
        return {
            "Wx": self.Wx, "Wh": self.Wh, "b": self.b,
            "hidden_size": self.hidden_size,
            "activation": self.activation, "with_bias": self.with_bias,
            "initializer": self.initializer, "bias_init": self.bias_init,
        }

    def load_state_dict(self, sd: Dict[str, Any]):
        for k in ("Wx","Wh","b"):
            if k in sd and sd[k] is not None:
                if getattr(self, k) is None:
                    setattr(self, k, sd[k].copy())
                else:
                    getattr(self, k)[...] = sd[k]
        # 하이퍼파라미터 재적용(선택)
        for k in ("hidden_size","activation","with_bias","initializer","bias_init"):
            if k in sd:
                setattr(self, k, sd[k])
        # grad 버퍼 재생성
        self.dWx = cp.zeros_like(self.Wx) if self.Wx is not None else None
        self.dWh = cp.zeros_like(self.Wh) if self.Wh is not None else None
        self.db  = cp.zeros_like(self.b)  if self.b  is not None else None
        return self
