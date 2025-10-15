from __future__ import annotations
from typing import Iterable, Tuple, List
import cupy as cp
ParamTriplet = Tuple[cp.ndarray, cp.ndarray, bool]

# Lion: m = β1*m + (1-β1)*g; p -= lr * sign(m) + decoupled_wd
_lion_fp32 = cp.ElementwiseKernel(
    in_params='''
      float32 p, float32 g, float32 m,
      float32 lr, float32 b1, float32 wd, float32 grad_scale, int32 exempt
    ''',
    out_params='float32 p_out, float32 m_out',
    operation=r'''
      float ge = grad_scale * g;
      float m_new = b1 * m + (1.f - b1) * ge;
      float step = copysignf(1.f, m_new); // sign(m_new)
      float decay = (exempt==0) ? (lr * wd * p) : 0.f;
      p_out = p - lr * step - decay;
      m_out = m_new;
    ''', name='lion_fused_fp32'
)

_lion_fp16 = cp.ElementwiseKernel(
    in_params='''
      float16 p, float16 g, float32 m,
      float32 lr, float32 b1, float32 wd, float32 grad_scale, int32 exempt
    ''',
    out_params='float16 p_out, float32 m_out',
    operation=r'''
      float pf=(float)p, gf=(float)g;
      float ge = grad_scale * gf;
      float m_new = b1 * m + (1.f - b1) * ge;
      float step = (m_new>0.f?1.f:-1.f);
      float decay = (exempt==0) ? (lr * wd * pf) : 0.f;
      float w = pf - lr * step - decay;
      p_out=(float16)w; m_out=m_new;
    ''', name='lion_fused_fp16'
)

def _lion_update(p,g,m,*,lr,b1,wd,grad_scale,exempt):
    ex=cp.int32(1 if exempt else 0)
    if p.dtype==cp.float16:
        p_out,m_out=_lion_fp16(p,g,m,lr,b1,wd,grad_scale,ex)
    else:
        p_out,m_out=_lion_fp32(p,g,m,lr,b1,wd,grad_scale,ex)
    p[...] = p_out; m[...] = m_out

class LionOpt:
    def __init__(self, params: Iterable[ParamTriplet], *, lr=1e-4, beta1=0.9, wd=1e-2):
        self.groups=[]
        for (p,g,ex) in params:
            assert g.dtype==p.dtype
            self.groups.append({"p":p,"g":g,"exempt":bool(ex),"m":cp.zeros(p.shape,cp.float32)})
        self.lr=cp.array(lr,cp.float32); self.b1=cp.array(beta1,cp.float32)
        self.wd=cp.array(wd,cp.float32); self.grad_scale=cp.array(1.0,cp.float32)
        self.t=cp.array(0,cp.int32)

    def _apply(self):
        cp.add(self.t,1,out=self.t)
        for s in self.groups:
            _lion_update(s["p"],s["g"],s["m"], lr=self.lr, b1=self.b1, wd=self.wd,
                         grad_scale=self.grad_scale, exempt=s["exempt"])
    step=step_into=_apply


    # ... 상단 동일 ...

    def rebind_grads(self, params):
        params = list(params)
        if len(self.groups) == 0:
            # 빈 상태 → 첫 rebind 시 전체 초기화
            for (p, g, ex) in params:
                assert isinstance(p, cp.ndarray) and isinstance(g, cp.ndarray)
                assert g.shape == p.shape and g.dtype == p.dtype
                self.groups.append({
                    "p": p,
                    "g": g,
                    "exempt": bool(ex),
                    "m": cp.zeros(p.shape, dtype=cp.float32),  # ← 상태 텐서 생성
                })
            return
        # 기존 그룹이 있으면 포인터 동일성/정합성만 확인하고 grad만 교체
        assert len(params) == len(self.groups), \
            f"param count mismatch: opt={len(self.groups)} vs {len(params)}"
        for s, (p, g, ex) in zip(self.groups, params):
            assert s["p"] is p
            assert g.shape == p.shape and g.dtype == p.dtype
            s["g"] = g
            s["exempt"] = bool(ex)
