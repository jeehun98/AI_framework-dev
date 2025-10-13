from __future__ import annotations
from typing import Iterable, Optional, Tuple, Any, Dict
import cupy as cp

from .base import Layer
from ..ops import batchnorm as bnops


class BatchNorm2d(Layer):
    """
    Batch Normalization (NCHW 기본, NHWC 선택) — capture-safe 레이어.

    - 입력:  X: float32 [N, C, H, W]  (channels_last=False)
             X: float32 [N, H, W, C]  (channels_last=True)
    - 파라미터(affine=True일 때):
        gamma: [C], beta: [C]
    - 러닝 통계:
        running_mean: [C], running_var: [C]
      * training=True: 러닝 통계 업데이트 + save_mean/save_invstd 반환해 bwd에 사용
      * training=False: 러닝 통계 사용(inference), save_mean=None, save_invstd 반환

    capture 경로:
      - forward_into(...)가 save_mean/save_invstd를 레이어 내부 캡처 버퍼에 저장
      - backward_into(...)가 해당 저장 버퍼를 사용
      - gW_out(=dgamma), gB_out(=dbeta)를 외부에서 사전 할당하여 전달 (없으면 스킵)

    파라미터 이터레이터(parameters):
      - affine=True이면 (gamma, dgamma, "<name>.gamma"), (beta, dbeta, "<name>.beta")
    """

    def __init__(
        self,
        *,
        channels_last: bool = False,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        self.channels_last = bool(channels_last)
        self.eps = float(eps)
        self.momentum = float(momentum)
        self.affine = bool(affine)

        # params
        self.gamma: Optional[cp.ndarray] = None
        self.beta:  Optional[cp.ndarray] = None
        self.dgamma: Optional[cp.ndarray] = None
        self.dbeta:  Optional[cp.ndarray] = None

        # running stats
        self.running_mean: Optional[cp.ndarray] = None
        self.running_var:  Optional[cp.ndarray] = None

        # saved stats (from forward)
        self._last_save_mean: Optional[cp.ndarray] = None
        self._last_save_invstd: Optional[cp.ndarray] = None

        # capture용 저장 버퍼 (forward_into에서 기록 → backward_into에서 사용)
        self._cap_save_mean: Optional[cp.ndarray] = None
        self._cap_save_invstd: Optional[cp.ndarray] = None

        # shapes
        self.input_shape: Optional[Tuple[int, ...]] = None
        self.output_shape: Optional[Tuple[int, ...]] = None
        self.built: bool = False

        self.training: bool = True

    # ---------------- build / shapes ----------------
    def build(self, input_shape: Tuple[int, ...]) -> None:
        if len(input_shape) != 4:
            raise ValueError(f"BatchNorm2d expects 4D input, got {input_shape}")
        N, d1, d2, d3 = map(int, input_shape)
        C = d3 if self.channels_last else d1

        # 파라미터/러닝 통계 생성
        if self.affine:
            self.gamma = cp.ones((C,), dtype=cp.float32)
            self.beta  = cp.zeros((C,), dtype=cp.float32)
            self.dgamma = cp.zeros_like(self.gamma)
            self.dbeta  = cp.zeros_like(self.beta)
        else:
            self.gamma = None
            self.beta  = None
            self.dgamma = None
            self.dbeta  = None

        self.running_mean = cp.zeros((C,), dtype=cp.float32)
        self.running_var  = cp.ones((C,), dtype=cp.float32)

        # saved buff (추적용)
        self._last_save_mean = None
        self._last_save_invstd = None
        self._cap_save_mean = None
        self._cap_save_invstd = None

        self.input_shape  = tuple(map(int, input_shape))
        self.output_shape = tuple(map(int, input_shape))
        self.built = True

    def compute_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        if len(input_shape) != 4:
            raise ValueError("BatchNorm2d requires 4D input shape")
        return tuple(map(int, input_shape))

    # ---------------- params iterator ----------------
    def parameters(self) -> Iterable[Tuple[Any, Any, str]]:
        if self.affine:
            assert self.gamma is not None and self.beta is not None
            # dgamma/dbeta가 없을 수도 있으니 지연 생성
            if self.dgamma is None: self.dgamma = cp.zeros_like(self.gamma)
            if self.dbeta  is None: self.dbeta  = cp.zeros_like(self.beta)
            yield (self.gamma, self.dgamma, f"{self.name or 'BatchNorm2d'}.gamma")
            yield (self.beta,  self.dbeta,  f"{self.name or 'BatchNorm2d'}.beta")

    # ---------------- eager path ----------------
    def call(self, X: cp.ndarray) -> cp.ndarray:
        assert self.built, "call() before build()"
        assert self.running_mean is not None and self.running_var is not None

        Y, save_mean, save_invstd = bnops.forward(
            X,
            running_mean=self.running_mean,
            running_var=self.running_var,
            gamma=self.gamma if self.affine else None,
            beta=self.beta if self.affine else None,
            channels_last=self.channels_last,
            eps=self.eps,
            momentum=self.momentum,
            training=self.training,
            with_affine=self.affine,
            stream=None,
            out=None,
        )
        # backward용 저장
        self._last_save_mean = save_mean
        self._last_save_invstd = save_invstd
        return Y

    def backward(self, gY: cp.ndarray) -> cp.ndarray:
        assert self.built, "backward() before build()"
        assert self._last_save_invstd is not None, "forward() must be called before backward()"

        # call에서 쓴 X/Y가 레이어 내부에 없으므로, BN backward는 (X, dY, saved stats)가 필요.
        # base.Sequential.backward는 일반적으로 각 레이어가 캐시한 텐서로 bwd를 호출하지만,
        # 여기서는 간단히 gX만 반환하는 인터페이스로 두고, 상위가 dgamma/dbeta를 옵티마이저로 전달.
        # eager 경로에서는 forward에서 나온 X를 별도로 전달받지 않으므로,
        # Layer 기본 구현과 함께 쓰려면 Sequential이 gY만 전달하는 구조여야 한다.
        # 본 프레임워크의 다른 레이어와의 일관성을 위해, 여기서는
        # "gY와 동일 shape X를 요구"하는 ops API를 만족시키기 위해
        # 추정치로 "정방향 입력 X를 call에서 즉시 보관"하도록 하려면 별도 캐시가 필요.
        # 간단화를 위해, BN의 eager.backward는 Base/Sequential에서
        # gY를 X 자리에 넣지 않도록 주의해야 한다.
        raise NotImplementedError(
            "Use capture path or call ops.batchnorm.backward directly with saved X/Y if eager backward is needed."
        )

    # ---------------- capture-safe path ----------------
    def forward_into(
        self,
        X: cp.ndarray,
        *,
        out: cp.ndarray,
        z_out: Optional[cp.ndarray] = None,   # BN은 pre-activation 개념이 없으므로 미사용
        stream: Optional[int] = None,
        work: Optional[Any] = None,           # 호환 키워드 (미사용)
    ) -> None:
        assert self.built, "forward_into() before build()"
        assert self.running_mean is not None and self.running_var is not None

        Y, save_mean, save_invstd = bnops.forward(
            X,
            running_mean=self.running_mean,
            running_var=self.running_var,
            gamma=self.gamma if self.affine else None,
            beta=self.beta if self.affine else None,
            channels_last=self.channels_last,
            eps=self.eps,
            momentum=self.momentum,
            training=self.training,
            with_affine=self.affine,
            stream=stream,
            out=out,
        )
        # capture용 저장 버퍼(외부에서 주입하지 않아도 레이어 내부에 보관)
        self._cap_save_mean = save_mean
        self._cap_save_invstd = save_invstd
        # z_out은 무시 (BN은 pre-activation 없음)

    def backward_into(
        self,
        gY: cp.ndarray,
        *,
        gA_out: cp.ndarray,
        gW_out: Optional[cp.ndarray] = None,  # == dgamma
        gB_out: Optional[cp.ndarray] = None,  # == dbeta
        work_dZ: Optional[Any] = None,        # 호환 (미사용)
        lt_workspace: Optional[Any] = None,   # 호환 (미사용)
        stream: Optional[int] = None,
        work: Optional[Any] = None,           # 호환 (미사용)
        # BN은 X/Y 텐서가 필요하지만, ops 구현은 (dY, X, save_mean, save_invstd) 형태로 동작.
        # 여기서는 "X == gA_out의 shape"만 보장되므로, 정확한 X 텐서가 필요하다.
        # Capture 플랜에서 BN 이전 레이어의 출력(y)을 BN의 입력 X로 전달해 두고
        # 그 버퍼를 여기로 넘겨야 한다. 현재 그래프 실행기에서는 레이어별 y를 알고 있으며,
        # BN 바로 앞 레이어의 y를 X_saved로 받을 수 있게 확장할 수 있다.
        # 임시로 시그니처만 남기고, 꼭 X_saved를 전달하도록 강제한다.
        X_saved: Optional[cp.ndarray] = None,
        Y_saved: Optional[cp.ndarray] = None,  # 미사용 (검증용)
    ) -> None:
        assert self.built, "backward_into() before build()"
        if X_saved is None:
            raise RuntimeError("[BatchNorm2d.backward_into] X_saved must be provided by capture plan")
        if self._cap_save_invstd is None:
            raise RuntimeError("[BatchNorm2d.backward_into] missing saved invstd from forward_into")

        # dX, dgamma, dbeta 요청 플래그
        want_dX = True
        want_dg = self.affine and (gW_out is not None)
        want_db = self.affine and (gB_out is not None)

        outs = bnops.backward(
            dY=gY,
            X=X_saved,
            save_mean=(self._cap_save_mean if self._cap_save_mean is not None
                       else cp.zeros_like(self._cap_save_invstd)),
            save_invstd=self._cap_save_invstd,
            gamma=self.gamma if self.affine else None,
            with_affine=self.affine,
            channels_last=self.channels_last,
            want_dX=want_dX,
            want_dgamma=want_dg,
            want_dbeta=want_db,
            stream=stream,
        )
        # 쓰기
        gA_out[...] = outs["dX"]
        if want_dg and "dgamma" in outs and gW_out is not None:
            gW_out[...] = outs["dgamma"]
        if want_db and "dbeta" in outs and gB_out is not None:
            gB_out[...] = outs["dbeta"]

    # ---------------- misc ----------------
    def zero_grad(self):
        if self.dgamma is not None: self.dgamma[...] = 0
        if self.dbeta  is not None: self.dbeta[...]  = 0

    def state_dict(self) -> Dict[str, Any]:
        return {
            "channels_last": self.channels_last,
            "eps": self.eps,
            "momentum": self.momentum,
            "affine": self.affine,
            "gamma": self.gamma,
            "beta": self.beta,
            "running_mean": self.running_mean,
            "running_var": self.running_var,
        }

    def load_state_dict(self, sd: Dict[str, Any]):
        # 하이퍼파라미터
        for k in ("channels_last", "eps", "momentum", "affine"):
            if k in sd: setattr(self, k, sd[k])

        # 텐서 (있으면 복사/교체)
        def _copy_in(attr: str):
            if attr in sd and sd[attr] is not None:
                val = sd[attr]
                cur = getattr(self, attr)
                if cur is None:
                    setattr(self, attr, val.copy())
                else:
                    cur[...] = val

        _copy_in("gamma"); _copy_in("beta")
        _copy_in("running_mean"); _copy_in("running_var")

        # grad 버퍼 재생성
        if self.affine:
            if self.gamma is not None and (self.dgamma is None or self.dgamma.shape != self.gamma.shape):
                self.dgamma = cp.zeros_like(self.gamma)
            if self.beta is not None and (self.dbeta is None or self.dbeta.shape != self.beta.shape):
                self.dbeta = cp.zeros_like(self.beta)
        else:
            self.dgamma = None
            self.dbeta  = None
        return self
