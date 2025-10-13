# python/graph_executor_v2/layers/dropout.py
from __future__ import annotations
from typing import Iterable, Optional, Tuple, Any, Dict
import cupy as cp

from .base import Layer
from ..ops import dropout as dropops


class Dropout(Layer):
    """
    캡처-세이프 Dropout 레이어.
      - training=True에서만 p 적용, inference에선 항등.
      - capture 시 work에 mask(int32)와 counter_base(옵션)가 있으면 사용.
    """

    def __init__(
        self,
        p: float = 0.1,
        *,
        scale_in_train: bool = True,
        seed: int = 0x1234,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        self.p = float(p)
        self.scale_in_train = bool(scale_in_train)
        self.seed = int(seed)

        self.input_shape: Optional[Tuple[int, ...]] = None
        self.output_shape: Optional[Tuple[int, ...]] = None
        self.built: bool = False

        self.training: bool = True  # Layer 기본 인터페이스 따라감

    # Dropout은 파라미터 없음
    def parameters(self) -> Iterable[Tuple[Any, Any, str]]:
        if False:
            yield from ()  # 빈 이터레이터

    def build(self, input_shape: Tuple[int, ...]) -> None:
        self.input_shape = tuple(map(int, input_shape))
        self.output_shape = tuple(map(int, input_shape))
        self.built = True

    def compute_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        return tuple(map(int, input_shape))

    # -------- eager --------
    def call(self, X: cp.ndarray) -> cp.ndarray:
        assert self.built
        out = dropops.forward(
            X, p=self.p, train=self.training,
            scale_in_train=self.scale_in_train,
            seed=self.seed, counter_base=0,  # eager는 counter_base=0 고정
            out=None, mask_out=None, stream=None
        )["y"]
        return out

    def backward(self, gY: cp.ndarray) -> cp.ndarray:
        # eager backward 경로는 마스크가 필요 → call에서 보관하지 않으므로 비활성화
        raise NotImplementedError("Use capture path or retain mask externally for eager backward.")

    # -------- capture-safe --------
    def forward_into(
        self,
        X: cp.ndarray,
        *,
        out: cp.ndarray,
        z_out: Optional[cp.ndarray] = None,
        stream: Optional[int] = None,
        work: Optional[Any] = None,  # plan에서 제공하는 {mask, counter_base}
    ) -> None:
        assert self.built
        # eval 모드(p=0): 항등 복사로 끝낸다(마스크 불필요)
        p_eff = (self.p if self.training else 0.0)
        if p_eff == 0.0:
            out[...] = X
            return

        mask = getattr(work, "mask", None) if work is not None else None
        counter_base = int(getattr(work, "counter_base", 0)) if work is not None else 0

        dropops.forward_into(
            X, y=out, mask=mask,
            p=p_eff, scale_in_train=self.scale_in_train,
            seed=self.seed, counter_base=counter_base,
            stream=stream
        )

    def backward_into(
        self,
        gY: cp.ndarray,
        *,
        gA_out: cp.ndarray,
        gW_out: Optional[cp.ndarray] = None,
        gB_out: Optional[cp.ndarray] = None,
        work_dZ: Optional[Any] = None,
        lt_workspace: Optional[Any] = None,
        stream: Optional[int] = None,
        work: Optional[Any] = None,
    ) -> None:
        assert self.built
        p_eff = (self.p if self.training else 0.0)

        if p_eff == 0.0:
            # 항등: dX = dY
            gA_out[...] = gY
            return

        mask = getattr(work, "mask", None) if work is not None else None
        if mask is None:
            raise RuntimeError("[Dropout.backward_into] missing mask buffer from capture plan")

        dropops.backward_into(
            gY, mask, dx=gA_out,
            p=p_eff, scale_in_train=self.scale_in_train,
            stream=stream
        )

    # -------- misc --------
    def zero_grad(self):
        return  # 파라미터 없음

    def state_dict(self) -> Dict[str, Any]:
        return {"p": self.p, "scale_in_train": self.scale_in_train, "seed": self.seed}

    def load_state_dict(self, sd: Dict[str, Any]):
        for k in ("p", "scale_in_train", "seed"):
            if k in sd:
                setattr(self, k, sd[k])
        return self
