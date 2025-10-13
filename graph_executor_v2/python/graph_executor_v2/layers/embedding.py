# python/graph_executor_v2/layers/embedding.py
from __future__ import annotations
from typing import Iterable, Optional, Tuple, Any, Dict
import cupy as cp

from .base import Layer
from ..ops import embedding as embops


class Embedding(Layer):
    """
    Lookup-table Embedding (capture-safe).

    입력:
      - I: int32, [N, L] 또는 [L] (토큰 인덱스)
    파라미터:
      - W: float32, [V, D]
      - (옵션) padding_idx: 해당 토큰행은 항상 0(또는 학습에서 제외)
    출력:
      - Y: float32, [N, L, D] 또는 [L, D]

    역전파:
      - dW만 존재(인덱스에 대한 미분은 없음). gA_out은 0으로 채워 전달(체인 유지용).

    Capture 경로:
      - forward_into()에서 입력 인덱스 I 를 내부에 저장(_cap_I)
      - backward_into()에서 저장된 _cap_I 를 사용하여 dW 를 계산
    Eager 경로:
      - call()에서 _last_I 저장, backward()에서 이를 사용
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        *,
        padding_idx: int = -1,
        scale_grad_by_freq: bool = False,
        out_scale: float = 1.0,
        initializer: str = "normal",  # "normal" | "xavier" | "zeros"
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        self.num_embeddings = int(num_embeddings)
        self.embedding_dim  = int(embedding_dim)
        self.padding_idx    = int(padding_idx)
        self.scale_grad_by_freq = bool(scale_grad_by_freq)
        self.out_scale = float(out_scale)
        self.initializer = str(initializer).lower()

        # params
        self.W: Optional[cp.ndarray]  = None  # [V, D]
        self.dW: Optional[cp.ndarray] = None  # [V, D]

        # eager/capture caches
        self._last_I: Optional[cp.ndarray] = None
        self._cap_I:  Optional[cp.ndarray] = None

        # shape state
        self.input_shape: Optional[Tuple[int, ...]] = None
        self.output_shape: Optional[Tuple[int, ...]] = None
        self.built: bool = False

    # ------------- build & shapes -------------
    def build(self, input_shape: Tuple[int, ...]) -> None:
        """
        input_shape: (N, L) or (L,)
        """
        if len(input_shape) not in (1, 2):
            raise ValueError(f"Embedding expects 1D/2D indices, got {input_shape}")

        V, D = self.num_embeddings, self.embedding_dim
        W = cp.empty((V, D), dtype=cp.float32)
        if self.initializer in ("normal", "gaussian"):
            # Std ~ 0.02 (일반적인 초기화)
            W[...] = 0.02 * cp.random.randn(V, D).astype(cp.float32)
        elif self.initializer in ("xavier", "glorot"):
            limit = (6.0 / (V + D)) ** 0.5
            W[...] = cp.random.uniform(-limit, limit, size=(V, D)).astype(cp.float32)
        elif self.initializer in ("zeros", "zero"):
            W.fill(0)
        else:
            raise ValueError(f"unknown initializer: {self.initializer}")

        # padding 행을 0으로 고정(선택)
        if 0 <= self.padding_idx < V:
            W[self.padding_idx].fill(0.0)

        self.W  = W
        self.dW = cp.zeros_like(W)

        # output shape 추론: [N,L,D] 또는 [L,D]
        if len(input_shape) == 2:
            N, L = map(int, input_shape)
            self.output_shape = (N, L, D)
        else:
            L = int(input_shape[0])
            self.output_shape = (L, D)

        self.input_shape = tuple(map(int, input_shape))
        self.built = True

    def compute_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        if len(input_shape) == 2:
            N, L = map(int, input_shape)
            return (N, L, self.embedding_dim)
        elif len(input_shape) == 1:
            L = int(input_shape[0])
            return (L, self.embedding_dim)
        raise ValueError("Embedding expects 1D or 2D input shape")

    # ------------- params iterator -------------
    def parameters(self) -> Iterable[Tuple[Any, Any, str]]:
        if self.W is not None:
            if self.dW is None or self.dW.shape != self.W.shape:
                self.dW = cp.zeros_like(self.W)
            yield (self.W, self.dW, f"{self.name or 'Embedding'}.W")

    # ------------- eager path -------------
    def call(self, I: cp.ndarray) -> cp.ndarray:
        assert self.built and self.W is not None
        # 저장(역전파용)
        self._last_I = I
        Y = embops.forward(
            self.W, I,
            padding_idx=self.padding_idx,
            out_scale=self.out_scale,
            stream=None, out=None
        )
        return Y

    def backward(self, gY: cp.ndarray) -> cp.ndarray:
        assert self.built and self.W is not None
        if self._last_I is None:
            raise RuntimeError("Embedding.backward() requires call() first (missing indices).")
        V, D = map(int, self.W.shape)
        # dW 누적(또는 덮어쓰기). 여기서는 덮어쓰기로 가정하여 zero 후 accumulate.
        if self.dW is None or tuple(self.dW.shape) != (V, D):
            self.dW = cp.zeros((V, D), dtype=cp.float32)
        else:
            self.dW.fill(0.0)

        embops.backward(
            self._last_I, gY,
            V=V, D=D,
            padding_idx=self.padding_idx,
            scale_grad_by_freq=self.scale_grad_by_freq,
            stream=None,
            dW_out=self.dW
        )
        # 인덱스 입력의 upstream grad는 존재하지 않으므로 0 텐서를 반환(체인 정합).
        return cp.zeros(self.input_shape, dtype=cp.float32)  # type: ignore[arg-type]

    # ------------- capture-safe path -------------
    def forward_into(
        self,
        I: cp.ndarray,
        *,
        out: cp.ndarray,
        z_out: Optional[cp.ndarray] = None,   # 미사용
        stream: Optional[int] = None,
        work: Optional[Any] = None,           # 시그니처 정합용(미사용)
    ) -> None:
        assert self.built and self.W is not None
        embops.forward(
            self.W, I,
            padding_idx=self.padding_idx,
            out_scale=self.out_scale,
            stream=stream,
            out=out
        )
        # 캡처에서 bwd용 인덱스 저장
        self._cap_I = I

    def backward_into(
        self,
        gY: cp.ndarray,
        *,
        gA_out: cp.ndarray,                  # 인덱스 입력의 grad는 없으므로 0으로 채움
        gW_out: Optional[cp.ndarray] = None, # dW
        gB_out: Optional[cp.ndarray] = None, # 없음(미사용)
        work_dZ: Optional[Any] = None,       # 호환(미사용)
        lt_workspace: Optional[Any] = None,  # 호환(미사용)
        stream: Optional[int] = None,
        work: Optional[Any] = None,          # 호환(미사용)
    ) -> None:
        assert self.built and self.W is not None
        if self._cap_I is None:
            raise RuntimeError("[Embedding.backward_into] missing saved indices (forward_into must run first)")

        # gA_out: 0 채워 upstream 체인 유지
        if gA_out is not None:
            gA_out.fill(0.0)

        if gW_out is None:
            # 플랜에서 gW_out이 반드시 사전할당되어 오도록 하는 게 이상적.
            # 그래도 안전하게 내부 임시 버퍼로 계산해두고 버린다.
            V, D = map(int, self.W.shape)
            tmp = cp.zeros((V, D), dtype=cp.float32)
            embops.backward(
                self._cap_I, gY,
                V=V, D=D,
                padding_idx=self.padding_idx,
                scale_grad_by_freq=self.scale_grad_by_freq,
                stream=stream,
                dW_out=tmp
            )
        else:
            V, D = map(int, gW_out.shape)
            embops.backward(
                self._cap_I, gY,
                V=V, D=D,
                padding_idx=self.padding_idx,
                scale_grad_by_freq=self.scale_grad_by_freq,
                stream=stream,
                dW_out=gW_out
            )

    # ------------- misc -------------
    def zero_grad(self):
        if self.dW is not None:
            self.dW[...] = 0

    def state_dict(self) -> Dict[str, Any]:
        return {
            "num_embeddings": self.num_embeddings,
            "embedding_dim":  self.embedding_dim,
            "padding_idx":    self.padding_idx,
            "scale_grad_by_freq": self.scale_grad_by_freq,
            "out_scale": self.out_scale,
            "initializer": self.initializer,
            "W": self.W,
        }

    def load_state_dict(self, sd: Dict[str, Any]):
        for k in ("num_embeddings", "embedding_dim", "padding_idx",
                  "scale_grad_by_freq", "out_scale", "initializer"):
            if k in sd: setattr(self, k, sd[k])

        if "W" in sd and sd["W"] is not None:
            W = sd["W"]
            if self.W is None or tuple(self.W.shape) != tuple(W.shape):
                self.W = W.copy()
            else:
                self.W[...] = W
        # grad buffer 갱신
        if self.W is not None:
            self.dW = cp.zeros_like(self.W)
        return self
