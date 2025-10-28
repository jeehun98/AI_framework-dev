# File: graph_executor_v2/optim/adamw.py
from __future__ import annotations
from typing import Iterable, List, Tuple, Optional, Callable
import cupy as cp

"""
AdamWOpt: CUDA Graph 친화 AdamW (Decoupled)
- 모든 하이퍼파라미터/모멘트/타임스텝을 0-D device 텐서로 보관하여 host round-trip 제거
- step() / step_into() 동일 경로(_apply) → 그래프 캡처/이저 공용
- 파라미터가 fp16이어도 grad는 fp32로 들어올 수 있도록 허용 (CapturePlan gW/gB가 fp32인 설계 호환)
- bias-correction 분모 0 가드
"""

# (param, grad, is_exempt_from_weight_decay)
ParamTriplet = Tuple[cp.ndarray, cp.ndarray, bool]

# ---------- Fused Elementwise Kernels (모듈 로드 시 1회 생성: 캡처 전) ----------
# NOTE: 두 커널 모두 grad(g)는 float32를 받는다. (혼합정밀 호환)
_adamw_fused_fp32 = cp.ElementwiseKernel(
    in_params='''
      float32 p, float32 g, float32 m, float32 v,
      float32 lr, float32 b1, float32 b2, float32 eps, float32 wd,
      float32 inv_bc1, float32 inv_bc2, float32 grad_scale, int32 exempt
    ''',
    out_params='float32 p_out, float32 m_out, float32 v_out',
    operation=r'''
      float ge = grad_scale * g;

      float m_new = b1 * m + (1.f - b1) * ge;
      float v_new = b2 * v + (1.f - b2) * ge * ge;

      float m_hat = m_new * inv_bc1;
      float v_hat = v_new * inv_bc2;

      float upd = m_hat / (sqrtf(v_hat) + eps);

      float w = p - lr * upd;
      float decay = (exempt == 0) ? (lr * wd * p) : 0.f;
      w = w - decay;

      p_out = w; m_out = m_new; v_out = v_new;
    ''',
    name='adamw_fused_fp32'
)

_adamw_fused_fp16 = cp.ElementwiseKernel(
    in_params='''
      float16 p, float32 g, float32 m, float32 v,
      float32 lr, float32 b1, float32 b2, float32 eps, float32 wd,
      float32 inv_bc1, float32 inv_bc2, float32 grad_scale, int32 exempt
    ''',
    out_params='float16 p_out, float32 m_out, float32 v_out',
    operation=r'''
      float pf = (float)p;
      float ge = grad_scale * g;

      float m_new = b1 * m + (1.f - b1) * ge;
      float v_new = b2 * v + (1.f - b2) * ge * ge;

      float m_hat = m_new * inv_bc1;
      float v_hat = v_new * inv_bc2;

      float upd = m_hat / (sqrtf(v_hat) + eps);

      float w = pf - lr * upd;
      float decay = (exempt == 0) ? (lr * wd * pf) : 0.f;
      w = w - decay;

      p_out = (float16)w; m_out = m_new; v_out = v_new;
    ''',
    name='adamw_fused_fp16'
)

def _adamw_fused_update(p, g, m, v, lr, b1, b2, eps, wd, inv_bc1, inv_bc2, grad_scale, exempt):
    """텐서 1개 업데이트. param fp16/fp32 모두 지원, grad는 float32 입력."""
    ex = cp.int32(1 if exempt else 0)
    if p.dtype == cp.float16:
        p_out, m_out, v_out = _adamw_fused_fp16(
            p, g.astype(cp.float32, copy=False), m, v,
            lr, b1, b2, eps, wd,
            inv_bc1, inv_bc2, grad_scale, ex
        )
    elif p.dtype == cp.float32:
        # g는 이미 float32여야 함
        p_out, m_out, v_out = _adamw_fused_fp32(
            p, g.astype(cp.float32, copy=False), m, v,
            lr, b1, b2, eps, wd,
            inv_bc1, inv_bc2, grad_scale, ex
        )
    else:
        raise TypeError(f"Unsupported dtype for param: {p.dtype}")
    p[...] = p_out; m[...] = m_out; v[...] = v_out


class AdamWOpt:
    """
    CUDA Graph 친화 AdamW (Decoupled, in-place, pointer-stable).

    groups: List[dict] = [{
        "p": param (fp16|fp32),
        "g": grad  (fp32 권장; fp16도 허용),
        "m": moment1 (fp32),
        "v": moment2 (fp32),
        "exempt": bool (WD 제외 여부)
    }, ...]
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
            assert isinstance(g, cp.ndarray), "grad must be cupy ndarray"
            assert p.dtype in (cp.float32, cp.float16), "param dtype should be fp16/fp32"
            # grad는 param과 같거나 fp32여야 함 (CapturePlan gW/gB=fp32 호환)
            assert g.dtype in (p.dtype, cp.float32), f"grad dtype {g.dtype} not compatible with param {p.dtype}"
            self.groups.append({
                "p": p,
                "g": g,
                "exempt": bool(is_exempt),
                "m": cp.zeros(p.shape, dtype=cp.float32),
                "v": cp.zeros(p.shape, dtype=cp.float32),
            })

        # 0-D device scalars (host round-trip 금지)
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
        """그래프/이저 공통 경로. host로 값을 끌어오지 않는다 (float()/item() 금지)."""
        cp.add(self.t, 1, out=self.t)

        cp.power(self.b1, self.t, out=self._b1_pow_t)   # b1^t
        cp.power(self.b2, self.t, out=self._b2_pow_t)   # b2^t
        self._bc1[...] = 1.0 - self._b1_pow_t
        self._bc2[...] = 1.0 - self._b2_pow_t

        # 분모 안정 가드
        cp.maximum(self._bc1, cp.float32(1e-12), out=self._bc1)
        cp.maximum(self._bc2, cp.float32(1e-12), out=self._bc2)

        self._inv_bc1[...] = 1.0 / self._bc1
        self._inv_bc2[...] = 1.0 / self._bc2

        for slot in self.groups:
            p = slot["p"]; g = slot["g"]; m = slot["m"]; v = slot["v"]
            _adamw_fused_update(
                p, g, m, v,
                self.lr, self.b1, self.b2, self.eps, self.wd,
                self._inv_bc1, self._inv_bc2, self.grad_scale,
                slot["exempt"]
            )

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
            assert isinstance(slot["m"], cp.ndarray) and slot["m"].dtype == cp.float32
            assert isinstance(slot["v"], cp.ndarray) and slot["v"].dtype == cp.float32
        self._initialized = True

    def rebind_grads(self, params: Iterable[ParamTriplet]):
        """
        그래프 캡처 전 호출 권장.
        파라미터 객체 동일성(pointer-stable) 검증 + grad 텐서만 교체.
        grad dtype은 {param.dtype, fp32} 허용.
        """
        params = list(params)
        if len(self.groups) == 0:
            for (p_new, g_new, is_exempt_new) in params:
                assert g_new.dtype in (p_new.dtype, cp.float32)
                self.groups.append({
                    "p": p_new,
                    "g": g_new,
                    "exempt": bool(is_exempt_new),
                    "m": cp.zeros(p_new.shape, dtype=cp.float32),
                    "v": cp.zeros(p_new.shape, dtype=cp.float32),
                })
            return

        assert len(params) == len(self.groups), \
            f"param count mismatch: opt={len(self.groups)} vs {len(params)}"

        for slot, (p_new, g_new, is_exempt_new) in zip(self.groups, params):
            assert slot["p"] is p_new, "parameter object mismatch on rebind"
            assert g_new.shape == p_new.shape
            assert g_new.dtype in (p_new.dtype, cp.float32)
            slot["g"] = g_new
            slot["exempt"] = bool(is_exempt_new)


# --------- 파라미터 수집 유틸 ---------
def collect_params_from(
    model,
    allow_missing_grad: bool = True,
    decay_exempt_pred: Optional[Callable[[str], bool]] = None,
) -> List[ParamTriplet]:
    """
    (레거시) model.parameters()에서 수집.
    프로젝트 내 CapturePlan 기반 경로보다 거칠지만, 빠른 프로토타입에 유용.
    """
    if decay_exempt_pred is None:
        def decay_exempt_pred(nm: str) -> bool:
            n = nm.lower()
            return (n.endswith(".b") or (".b" in n) or ("bias" in n))

    triplets: List[ParamTriplet] = []
    for (p, g, name) in model.parameters():
        nm = str(name)
        is_exempt = bool(decay_exempt_pred(nm))
        if g is None:
            if not allow_missing_grad:
                continue
            g = cp.zeros_like(p)  # placeholder
        # grad dtype은 {param.dtype, fp32} 허용
        assert g.dtype in (p.dtype, cp.float32), f"grad dtype must be {p.dtype} or fp32, got {g.dtype}"
        triplets.append((p, g, is_exempt))
    return triplets


def collect_params_from_plan(
    model,
    plan,
    decay_exempt_pred: Optional[Callable[[str], bool]] = None
) -> List[ParamTriplet]:
    """
    CapturePlan(per_layer) 기반 수집기.
    - Dense/Conv: (W, gW), (b/bias/B, gB)
    - BN/LN: (gamma, gW), (beta, gB)  (shape 매칭으로 판별)
    - RNN(Elman): (Wx, gWx), (Wh, gWh), (b, gB), (h0, dh0?)  (존재하는 경우)
    - Embedding: (W|weight, gW)
    """
    if decay_exempt_pred is None:
        def decay_exempt_pred(nm: str) -> bool:
            n = nm.lower()
            return (n.endswith(".b") or (".b" in n) or ("bias" in n))

    def _has(x): return (x is not None) and hasattr(x, "shape")

    triplets: List[ParamTriplet] = []
    WEIGHT_NAMES = ("W", "weight")
    BIAS_NAMES   = ("b", "bias", "B")
    GAMMA_NAMES  = ("gamma", "weight")
    BETA_NAMES   = ("beta", "bias")

    for i, (lyr, per) in enumerate(zip(getattr(model, "layers", []), plan.per_layer)):
        # Dense/Conv 공통
        if _has(per.gW):
            pW = next((getattr(lyr, nm) for nm in WEIGHT_NAMES if _has(getattr(lyr, nm, None))), None)
            if _has(pW):
                gW = per.gW
                # grad dtype 허용: {p.dtype, fp32}
                assert gW.dtype in (pW.dtype, cp.float32), f"[L{i}] gW dtype {gW.dtype} not compatible with {pW.dtype}"
                triplets.append((pW, gW, bool(decay_exempt_pred(getattr(lyr, "name", f"layer{i}") + ".W"))))
        if _has(per.gB):
            pb = next((getattr(lyr, nm) for nm in BIAS_NAMES if _has(getattr(lyr, nm, None))), None)
            if _has(pb):
                gB = per.gB
                assert gB.dtype in (pb.dtype, cp.float32), f"[L{i}] gB dtype {gB.dtype} not compatible with {pb.dtype}"
                triplets.append((pb, gB, True))  # bias는 WD 제외 기본

        # BN/LN: gamma/beta
        if _has(per.gW):
            pgamma = next((getattr(lyr, nm) for nm in GAMMA_NAMES
                           if _has(getattr(lyr, nm, None)) and getattr(lyr, nm).shape == per.gW.shape), None)
            if _has(pgamma):
                triplets.append((pgamma, per.gW, True))
        if _has(per.gB):
            pbeta = next((getattr(lyr, nm) for nm in BETA_NAMES
                          if _has(getattr(lyr, nm, None)) and getattr(lyr, nm).shape == per.gB.shape), None)
            if _has(pbeta):
                triplets.append((pbeta, per.gB, True))

        # RNN(Elman)
        if _has(getattr(per, "gWx", None)) and _has(getattr(lyr, "Wx", None)):
            triplets.append((lyr.Wx, per.gWx, False))
        if _has(getattr(per, "gWh", None)) and _has(getattr(lyr, "Wh", None)):
            triplets.append((lyr.Wh, per.gWh, False))
        if _has(getattr(per, "gB", None)) and _has(getattr(lyr, "b", None)):
            triplets.append((lyr.b, per.gB, True))
        if _has(getattr(per, "dh0", None)) and _has(getattr(lyr, "h0", None)):
            triplets.append((lyr.h0, per.dh0, True))

        # Embedding
        if _has(per.gW):
            pE = getattr(lyr, "W", None) or getattr(lyr, "weight", None)
            if _has(pE) and pE.shape == per.gW.shape:
                # WD 적용 여부는 정책에 따라. 여기서는 기본 False(적용)로 두고 옵티마이저가 exempt 해석 가능.
                triplets.append((pE, per.gW, False))

    return triplets
