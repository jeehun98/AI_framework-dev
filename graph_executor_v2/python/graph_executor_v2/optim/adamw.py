# graph_executor_v2/optim/adamw.py
from __future__ import annotations
from typing import Iterable, List, Tuple, Optional, Callable
import cupy as cp

# (param, grad, is_bias_or_exempt)
ParamTriplet = Tuple[cp.ndarray, cp.ndarray, bool]

class AdamWOpt:
    """
    CUDA Graph 친화 AdamW (Decoupled, in-place, pointer-stable).
    - 모든 하이퍼/모멘트/타임스텝을 0-D 디바이스 텐서로 유지 (host round-trip 금지)
    - step() = step_into() = _apply() : 경로별 부호/로직 불일치 제거
    - bias-correction 분모 안정 가드 포함
    """

    def __init__(
        self,
        params: Iterable[ParamTriplet],
        *,
        lr: float = 1e-3,
        wd: float = 1e-4,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ):
        self.groups: List[dict] = []
        for (p, g, is_exempt) in params:
            assert isinstance(p, cp.ndarray), "param must be cupy ndarray"
            assert isinstance(g, cp.ndarray), "grad placeholder must be cupy ndarray"
            assert p.dtype in (cp.float32, cp.float16), "param dtype should be fp"
            self.groups.append({
                "p": p,
                "g": g,              # 캡처 후 rebind_grads로 교체될 수 있음
                "exempt": bool(is_exempt),
                "m": cp.zeros_like(p),
                "v": cp.zeros_like(p),
            })

        # 0-D device scalars
        self.lr   = cp.array(lr,   dtype=cp.float32)
        self.wd   = cp.array(wd,   dtype=cp.float32)
        self.b1   = cp.array(beta1, dtype=cp.float32)
        self.b2   = cp.array(beta2, dtype=cp.float32)
        self.eps  = cp.array(eps,  dtype=cp.float32)
        self.t    = cp.array(0,    dtype=cp.int32)

        self.grad_scale = cp.array(1.0, dtype=cp.float32)

        # bias-correction work buffers (0-D)
        self._b1_pow_t = cp.array(0.0, dtype=cp.float32)
        self._b2_pow_t = cp.array(0.0, dtype=cp.float32)
        self._bc1      = cp.array(1.0, dtype=cp.float32)  # 1 - b1^t
        self._bc2      = cp.array(1.0, dtype=cp.float32)  # 1 - b2^t
        self._inv_bc1  = cp.array(1.0, dtype=cp.float32)
        self._inv_bc2  = cp.array(1.0, dtype=cp.float32)

        self._initialized = False
        self.ensure_initialized()

    # ---------------- core update (공통 진입점) ----------------
    def _apply(self):
        """
        그래프/이저 공통 경로. host로 값을 끌어오지 않는다 (float()/item() 금지).
        """
        # t <- t + 1
        self.t += 1

        # pow/bias-correction (in-place)
        cp.power(self.b1, self.t, out=self._b1_pow_t)
        cp.power(self.b2, self.t, out=self._b2_pow_t)
        self._bc1[...] = 1.0 - self._b1_pow_t
        self._bc2[...] = 1.0 - self._b2_pow_t

        # 안정 가드: 분모 하한
        cp.maximum(self._bc1, cp.float32(1e-12), out=self._bc1)
        cp.maximum(self._bc2, cp.float32(1e-12), out=self._bc2)
        self._inv_bc1[...] = 1.0 / self._bc1
        self._inv_bc2[...] = 1.0 / self._bc2

        # 파라미터 루프
        for slot in self.groups:
            p = slot["p"]; g = slot["g"]; m = slot["m"]; v = slot["v"]
            exempt = slot["exempt"]

            # g_eff = grad_scale * g
            g_eff = self.grad_scale * g

            # 1st/2nd moments (in-place)
            m *= self.b1
            m += (1.0 - self.b1) * g_eff
            v *= self.b2
            v += (1.0 - self.b2) * (g_eff * g_eff)

            # bias-corrected update
            denom = cp.sqrt(v * self._inv_bc2) + self.eps
            upd   = (m * self._inv_bc1) / denom   # \hat{m} / (sqrt(\hat{v}) + eps)

            # decoupled weight decay (exempt 제외)
            if not exempt:
                # p -= lr * wd * p
                p -= (self.lr * self.wd) * p

            # descent
            p -= self.lr * upd

    def step_into(self):  # 그래프 캡처 내부
        self._apply()

    def step(self):       # 그래프 밖
        self._apply()

    # ---------------- utils ----------------
    def set_lr(self, new_lr: float):
        if new_lr <= 0:
            raise ValueError(f"Learning rate must be positive, got {new_lr}")
        self.lr[...] = cp.float32(new_lr)

    def set_wd(self, new_wd: float):
        self.wd[...] = cp.float32(new_wd)

    def set_grad_scale(self, s: float):
        self.grad_scale[...] = cp.float32(s)

    def zero_moments(self):
        for slot in self.groups:
            slot["m"].fill(0)
            slot["v"].fill(0)
        self.t[...] = 0
        self._b1_pow_t[...] = 0.0
        self._b2_pow_t[...] = 0.0
        self._bc1[...] = 1.0
        self._bc2[...] = 1.0
        self._inv_bc1[...] = 1.0
        self._inv_bc2[...] = 1.0

    def ensure_initialized(self):
        for slot in self.groups:
            assert isinstance(slot["p"], cp.ndarray)
            assert isinstance(slot["g"], cp.ndarray)
            assert isinstance(slot["m"], cp.ndarray)
            assert isinstance(slot["v"], cp.ndarray)
        self._initialized = True

    def rebind_grads(self, params: Iterable[ParamTriplet]):
        params = list(params)
        if len(self.groups) == 0:
            for (p_new, g_new, is_exempt_new) in params:
                self.groups.append({
                    "p": p_new,
                    "g": g_new,
                    "exempt": bool(is_exempt_new),
                    "m": cp.zeros_like(p_new),
                    "v": cp.zeros_like(p_new),
                })
            return

        assert len(params) == len(self.groups), \
            f"param count mismatch: opt={len(self.groups)} vs {len(params)}"

        for slot, (p_new, g_new, is_exempt_new) in zip(self.groups, params):
            assert slot["p"] is p_new, "parameter object mismatch on rebind"
            slot["g"] = g_new
            slot["exempt"] = bool(is_exempt_new)
            assert isinstance(slot["g"], cp.ndarray)
            assert slot["g"].dtype == slot["p"].dtype
            assert slot["g"].shape == slot["p"].shape


# --------- 수집 유틸 ---------
def collect_params_from(
    model,
    allow_missing_grad: bool = True,
    decay_exempt_pred: Optional[Callable[[str], bool]] = None,
) -> List[ParamTriplet]:
    if decay_exempt_pred is None:
        def decay_exempt_pred(nm: str) -> bool:
            n = nm.lower()
            if n.endswith(".b") or (".b" in n) or ("bias" in n):
                return True
            return False

    triplets: List[ParamTriplet] = []
    for (p, g, name) in model.parameters():
        nm = str(name)
        is_exempt = bool(decay_exempt_pred(nm))
        if g is None:
            if not allow_missing_grad:
                continue
            g = cp.zeros_like(p)  # placeholder
        triplets.append((p, g, is_exempt))
    return triplets


def collect_params_from_plan(model, capture_plan, decay_exempt_pred: Optional[Callable[[str], bool]] = None) -> List[ParamTriplet]:
    if decay_exempt_pred is None:
        def decay_exempt_pred(nm: str) -> bool:
            n = nm.lower()
            if n.endswith(".b") or (".b" in n) or ("bias" in n):
                return True
            return False

    triplets: List[ParamTriplet] = []
    bwd_bufs = capture_plan["buffers"]["bwd"]
    for idx, lyr in enumerate(model.layers):
        if hasattr(lyr, "W") and hasattr(lyr, "b"):
            b = bwd_bufs[idx]
            gW = b.get("gW", None)
            gB = b.get("gB", None)
            assert gW is not None and gB is not None, f"missing gW/gB for layer {idx}"
            w_exempt = bool(decay_exempt_pred(getattr(lyr, "name", f"layer{idx}") + ".W"))
            b_exempt = True  # bias는 decay 제외
            triplets.append((lyr.W, gW, w_exempt))
            triplets.append((lyr.b, gB, b_exempt))
    return triplets
