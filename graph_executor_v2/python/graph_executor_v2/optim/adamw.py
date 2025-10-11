# graph_executor_v2/optim/adamw.py
from __future__ import annotations
from typing import Iterable, List, Tuple, Optional, Callable
import cupy as cp

# (param, grad, is_bias_or_exempt)
ParamTriplet = Tuple[cp.ndarray, cp.ndarray, bool]

# ---------- Fused Elementwise Kernels (모듈 로드 시 1회 생성: 캡처 전) ----------
_adamw_fused_fp32 = cp.ElementwiseKernel(
    in_params='''
      float32 p, float32 g, float32 m, float32 v,
      float32 lr, float32 b1, float32 b2, float32 eps, float32 wd,
      float32 inv_bc1, float32 inv_bc2, float32 grad_scale, int32 exempt
    ''',
    out_params='float32 p_out, float32 m_out, float32 v_out',
    operation=r'''
      // grad scaling
      float ge = grad_scale * g;

      // moments
      float m_new = b1 * m + (1.f - b1) * ge;
      float v_new = b2 * v + (1.f - b2) * ge * ge;

      // bias correction (분모 0 가드는 inv_bc* 계산 시 보장)
      float m_hat = m_new * inv_bc1;
      float v_hat = v_new * inv_bc2;

      // update
      float upd = m_hat / (sqrtf(v_hat) + eps);

      // decoupled weight decay (branchless)
      float w = p - lr * upd;
      float decay = (exempt == 0) ? (lr * wd * p) : 0.f;
      w = w - decay;

      p_out = w; m_out = m_new; v_out = v_new;
    ''',
    name='adamw_fused_fp32'
)

_adamw_fused_fp16 = cp.ElementwiseKernel(
    in_params='''
      float16 p, float16 g, float32 m, float32 v,
      float32 lr, float32 b1, float32 b2, float32 eps, float32 wd,
      float32 inv_bc1, float32 inv_bc2, float32 grad_scale, int32 exempt
    ''',
    out_params='float16 p_out, float32 m_out, float32 v_out',
    operation=r'''
      // promote to fp32 for math
      float pf = (float)p;
      float gf = (float)g;

      float ge = grad_scale * gf;

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
    """텐서 1개에 대해 커널 1회로 AdamW 업데이트."""
    ex = cp.int32(1 if exempt else 0)
    # m, v는 항상 float32 누적(안정성)
    if p.dtype == cp.float16:
        p_out, m_out, v_out = _adamw_fused_fp16(
            p, g, m, v, lr, b1, b2, eps, wd, inv_bc1, inv_bc2, grad_scale, ex
        )
    elif p.dtype == cp.float32:
        p_out, m_out, v_out = _adamw_fused_fp32(
            p, g, m, v, lr, b1, b2, eps, wd, inv_bc1, inv_bc2, grad_scale, ex
        )
    else:
        raise TypeError(f"Unsupported dtype for param: {p.dtype}")
    # in-place 반영
    p[...] = p_out; m[...] = m_out; v[...] = v_out


class AdamWOpt:
    """
    CUDA Graph 친화 AdamW (Decoupled, in-place, pointer-stable).
    - 모든 하이퍼/모멘트/타임스텝을 0-D 디바이스 텐서로 유지 (host round-trip 금지)
    - step() = step_into() = _apply() : 경로별 부호/로직 불일치 제거
    - bias-correction 분모 안정 가드 포함
    - fp16 파라미터도 모멘트는 fp32 누적
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
            assert p.dtype in (cp.float32, cp.float16), "param dtype should be fp16/fp32"
            assert g.dtype == p.dtype, "grad dtype must match param dtype"
            # 모멘트는 항상 fp32로 누적(수치 안정성)
            self.groups.append({
                "p": p,
                "g": g,              # 캡처 후 rebind_grads로 교체될 수 있음
                "exempt": bool(is_exempt),
                "m": cp.zeros(p.shape, dtype=cp.float32),
                "v": cp.zeros(p.shape, dtype=cp.float32),
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
        - 텐서당 커널 1회로 업데이트
        """
        # t <- t + 1 (디바이스 in-place)
        cp.add(self.t, 1, out=self.t)

        # pow/bias-correction (in-place, 0-D)
        cp.power(self.b1, self.t, out=self._b1_pow_t)   # b1^t
        cp.power(self.b2, self.t, out=self._b2_pow_t)   # b2^t
        self._bc1[...] = 1.0 - self._b1_pow_t
        self._bc2[...] = 1.0 - self._b2_pow_t

        # 안정 가드: 분모 하한
        cp.maximum(self._bc1, cp.float32(1e-12), out=self._bc1)
        cp.maximum(self._bc2, cp.float32(1e-12), out=self._bc2)

        # 역수(0-D)
        self._inv_bc1[...] = 1.0 / self._bc1
        self._inv_bc2[...] = 1.0 / self._bc2

        # 파라미터 루프(텐서당 1커널)
        for slot in self.groups:
            p = slot["p"]; g = slot["g"]; m = slot["m"]; v = slot["v"]
            exempt = slot["exempt"]
            _adamw_fused_update(
                p, g, m, v,
                self.lr, self.b1, self.b2, self.eps, self.wd,
                self._inv_bc1, self._inv_bc2, self.grad_scale,
                exempt
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
            assert isinstance(slot["m"], cp.ndarray)
            assert isinstance(slot["v"], cp.ndarray)
            # m,v는 float32 누적
            assert slot["m"].dtype == cp.float32 and slot["v"].dtype == cp.float32
        self._initialized = True

    def rebind_grads(self, params: Iterable[ParamTriplet]):
        """
        그래프 캡처 전 호출 권장.
        파라미터 객체 동일성(pointer-stable) 검증 + grad 텐서만 교체.
        """
        params = list(params)
        if len(self.groups) == 0:
            for (p_new, g_new, is_exempt_new) in params:
                assert g_new.dtype == p_new.dtype
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
            assert g_new.dtype == p_new.dtype
            assert g_new.shape == p_new.shape
            slot["g"] = g_new
            slot["exempt"] = bool(is_exempt_new)


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
        # dtype 정합성 보증
        assert g.dtype == p.dtype, f"grad dtype must match param dtype: {g.dtype} vs {p.dtype}"
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
            # dtype 정합성
            assert gW.dtype == lyr.W.dtype and gB.dtype == lyr.b.dtype, \
                f"dtype mismatch in grads for layer {idx}"
            w_exempt = bool(decay_exempt_pred(getattr(lyr, "name", f"layer{idx}") + ".W"))
            b_exempt = True  # bias는 decay 제외
            triplets.append((lyr.W, gW, w_exempt))
            triplets.append((lyr.b, gB, b_exempt))
    return triplets
