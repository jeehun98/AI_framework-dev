# File: graph_executor_v2/optim/sgd.py
from __future__ import annotations
from typing import Iterable, Tuple, List
import cupy as cp

"""
CUDA Graph 친화 SGD (momentum/Nesterov, decoupled weight decay).
- 모든 하이퍼/모멘텀/타임스텝을 0-D device 텐서로 유지 (host round-trip 없음)
- grad 는 fp32 허용 (CapturePlan gW/gB가 fp32인 설계와 호환)
- step() / step_into() 동일 경로(_apply) → 캡처/이저 공용
"""

ParamTriplet = Tuple[cp.ndarray, cp.ndarray, bool]  # (param, grad, exempt_from_wd)

# ---- Fused Kernels (grad=g 는 float32로 받는다: 혼합정밀 호환) -----------------
_sgd_fp32 = cp.ElementwiseKernel(
    in_params='''
      float32 p, float32 g, float32 v,
      float32 lr, float32 momentum, float32 damp, int32 nesterov,
      float32 wd, int32 exempt, float32 grad_scale
    ''',
    out_params='float32 p_out, float32 v_out',
    operation=r'''
      float ge = grad_scale * g;
      float v_new = momentum * v + (1.f - damp) * ge;
      float step_vec = nesterov ? (momentum * v_new + (1.f - damp) * ge) : v_new;
      float decay = (exempt==0) ? (lr * wd * p) : 0.f;  // decoupled WD
      p_out = p - lr * step_vec - decay;
      v_out = v_new;
    ''',
    name='sgd_fused_fp32'
)

_sgd_fp16 = cp.ElementwiseKernel(
    in_params='''
      float16 p, float32 g, float32 v,
      float32 lr, float32 momentum, float32 damp, int32 nesterov,
      float32 wd, int32 exempt, float32 grad_scale
    ''',
    out_params='float16 p_out, float32 v_out',
    operation=r'''
      float pf = (float)p;
      float ge = grad_scale * g;
      float v_new = momentum * v + (1.f - damp) * ge;
      float step_vec = nesterov ? (momentum * v_new + (1.f - damp) * ge) : v_new;
      float decay = (exempt==0) ? (lr * wd * pf) : 0.f; // decoupled WD
      float w = pf - lr * step_vec - decay;
      p_out = (float16)w; v_out = v_new;
    ''',
    name='sgd_fused_fp16'
)

def _sgd_update(p, g, v, *, lr, momentum, damp, nesterov, wd, exempt, grad_scale):
    """텐서 1개 업데이트. param fp16/fp32 지원, grad는 float32 입력(필요시 astype)."""
    ex = cp.int32(1 if exempt else 0)
    ne = cp.int32(1 if nesterov else 0)
    g32 = g.astype(cp.float32, copy=False)
    if p.dtype == cp.float16:
        p_out, v_out = _sgd_fp16(p, g32, v, lr, momentum, damp, ne, wd, ex, grad_scale)
    elif p.dtype == cp.float32:
        p_out, v_out = _sgd_fp32(p, g32, v, lr, momentum, damp, ne, wd, ex, grad_scale)
    else:
        raise TypeError(f"Unsupported dtype for param: {p.dtype}")
    p[...] = p_out; v[...] = v_out


class SGDOpt:
    """
    Nesterov momentum + decoupled WD, pointer-stable SGD.

    groups[i] = {
        "p": param (fp16|fp32),
        "g": grad  (fp32 권장; fp16도 허용),
        "v": velocity (fp32),
        "exempt": bool (WD 제외 여부)
    }
    """
    def __init__(
        self,
        params: Iterable[ParamTriplet],
        *,
        lr: float = 1e-2,
        momentum: float = 0.9,
        nesterov: bool = True,
        damp: float = 0.0,
        wd: float = 0.0,
    ):
        self.groups: List[dict] = []
        for (p, g, ex) in params:
            assert isinstance(p, cp.ndarray) and isinstance(g, cp.ndarray)
            assert p.dtype in (cp.float16, cp.float32)
            # grad는 {param.dtype, fp32} 허용
            assert g.dtype in (p.dtype, cp.float32), f"grad dtype {g.dtype} not compatible with {p.dtype}"
            self.groups.append({
                "p": p,
                "g": g,
                "exempt": bool(ex),
                "v": cp.zeros(p.shape, dtype=cp.float32),
            })

        # 0-D device scalars (캡처-세이프)
        self.lr       = cp.array(lr,       dtype=cp.float32)
        self.momentum = cp.array(momentum, dtype=cp.float32)
        self.nesterov = bool(nesterov)
        self.damp     = cp.array(damp,     dtype=cp.float32)
        self.wd       = cp.array(wd,       dtype=cp.float32)
        self.grad_scale = cp.array(1.0,    dtype=cp.float32)
        self.t = cp.array(0, dtype=cp.int32)

        self._initialized = False
        self.ensure_initialized()

    # ---- core update (그래프/이저 공용 진입점) ----------------
    def _apply(self):
        cp.add(self.t, 1, out=self.t)
        for s in self.groups:
            _sgd_update(
                s["p"], s["g"], s["v"],
                lr=self.lr, momentum=self.momentum, damp=self.damp,
                nesterov=self.nesterov, wd=self.wd, exempt=s["exempt"],
                grad_scale=self.grad_scale
            )

    step = step_into = _apply  # 동일 경로 → CUDA Graph 캡처 친화

    # ---- utilities ----------------------------------------------------------
    def set_lr(self, new_lr: float):
        if new_lr <= 0:
            raise ValueError(f"Learning rate must be positive, got {new_lr}")
        self.lr[...] = cp.float32(new_lr)

    def set_wd(self, new_wd: float):
        self.wd[...] = cp.float32(new_wd)

    def set_momentum(self, m: float):
        self.momentum[...] = cp.float32(m)

    def set_grad_scale(self, s: float):
        self.grad_scale[...] = cp.float32(s)

    def zero_moments(self):
        for s in self.groups:
            s["v"].fill(0)
        self.t[...] = 0

    def ensure_initialized(self):
        for s in self.groups:
            assert isinstance(s["p"], cp.ndarray)
            assert isinstance(s["g"], cp.ndarray)
            assert isinstance(s["v"], cp.ndarray) and s["v"].dtype == cp.float32
        self._initialized = True

    def rebind_grads(self, params: Iterable[ParamTriplet]):
        """
        그래프 캡처 전 권장. 파라미터 객체 동일성(pointer-stable) 검증 + grad만 교체.
        grad dtype은 {param.dtype, fp32} 허용.
        """
        params = list(params)
        if len(self.groups) == 0:
            for (p, g, ex) in params:
                assert g.dtype in (p.dtype, cp.float32)
                self.groups.append({
                    "p": p, "g": g, "exempt": bool(ex),
                    "v": cp.zeros(p.shape, dtype=cp.float32),
                })
            return

        assert len(params) == len(self.groups), \
            f"param count mismatch: opt={len(self.groups)} vs {len(params)}"

        for s, (p, g, ex) in zip(self.groups, params):
            assert s["p"] is p, "parameter object mismatch on rebind"
            assert g.shape == p.shape and g.dtype in (p.dtype, cp.float32)
            s["g"] = g
            s["exempt"] = bool(ex)
