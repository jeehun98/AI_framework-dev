from __future__ import annotations
from typing import Iterable, Tuple, List
import cupy as cp
ParamTriplet = Tuple[cp.ndarray, cp.ndarray, bool]

_ams_fp32 = cp.ElementwiseKernel(
    in_params='''
      float32 p, float32 g, float32 m, float32 v, float32 vhat,
      float32 lr, float32 b1, float32 b2, float32 eps, float32 wd,
      float32 inv_bc1, float32 inv_bc2, float32 grad_scale, int32 exempt
    ''',
    out_params='float32 p_out, float32 m_out, float32 v_out, float32 vhat_out',
    operation=r'''
      float ge = grad_scale * g;
      float m_new = b1 * m + (1.f - b1) * ge;
      float v_new = b2 * v + (1.f - b2) * ge * ge;
      float m_hat = m_new * inv_bc1;
      float v_hat = v_new * inv_bc2;
      float vhat_new = fmaxf(vhat, v_hat);
      float upd = m_hat / (sqrtf(vhat_new) + eps);
      float w = p - lr * upd;
      float decay = (exempt==0) ? (lr * wd * p) : 0.f;
      p_out = w - decay; m_out=m_new; v_out=v_new; vhat_out=vhat_new;
    ''', name='amsgrad_fused_fp32'
)

_ams_fp16 = cp.ElementwiseKernel(
    in_params='''
      float16 p, float16 g, float32 m, float32 v, float32 vhat,
      float32 lr, float32 b1, float32 b2, float32 eps, float32 wd,
      float32 inv_bc1, float32 inv_bc2, float32 grad_scale, int32 exempt
    ''',
    out_params='float16 p_out, float32 m_out, float32 v_out, float32 vhat_out',
    operation=r'''
      float pf=(float)p, gf=(float)g;
      float ge = grad_scale * gf;
      float m_new = b1 * m + (1.f - b1) * ge;
      float v_new = b2 * v + (1.f - b2) * ge * ge;
      float m_hat = m_new * inv_bc1;
      float v_hat = v_new * inv_bc2;
      float vhat_new = fmaxf(vhat, v_hat);
      float upd = m_hat / (sqrtf(vhat_new) + eps);
      float w = pf - lr * upd;
      float decay = (exempt==0) ? (lr * wd * pf) : 0.f;
      p_out=(float16)(w - decay); m_out=m_new; v_out=v_new; vhat_out=vhat_new;
    ''', name='amsgrad_fused_fp16'
)

def _ams_update(p,g,m,v,vhat,*,lr,b1,b2,eps,wd,inv_bc1,inv_bc2,grad_scale,exempt):
    ex=cp.int32(1 if exempt else 0)
    if p.dtype==cp.float16:
        p_out,m_out,v_out,vhat_out=_ams_fp16(p,g,m,v,vhat,lr,b1,b2,eps,wd,inv_bc1,inv_bc2,grad_scale,ex)
    else:
        p_out,m_out,v_out,vhat_out=_ams_fp32(p,g,m,v,vhat,lr,b1,b2,eps,wd,inv_bc1,inv_bc2,grad_scale,ex)
    p[...] = p_out; m[...] = m_out; v[...] = v_out; vhat[...] = vhat_out

class AMSGradOpt:
    def __init__(self, params: Iterable[ParamTriplet], *, lr=1e-3, wd=1e-4, beta1=0.9, beta2=0.999, eps=1e-8):
        self.groups=[]
        for (p,g,ex) in params:
            assert g.dtype==p.dtype
            self.groups.append({"p":p,"g":g,"exempt":bool(ex),
                                "m":cp.zeros(p.shape,cp.float32),
                                "v":cp.zeros(p.shape,cp.float32),
                                "vhat":cp.zeros(p.shape,cp.float32)})
        self.lr=cp.array(lr,cp.float32); self.wd=cp.array(wd,cp.float32)
        self.b1=cp.array(beta1,cp.float32); self.b2=cp.array(beta2,cp.float32)
        self.eps=cp.array(eps,cp.float32); self.t=cp.array(0,cp.int32)
        self.grad_scale=cp.array(1.0,cp.float32)
        self._b1_pow_t=cp.array(0.0,cp.float32); self._b2_pow_t=cp.array(0.0,cp.float32)
        self._inv_bc1=cp.array(1.0,cp.float32); self._inv_bc2=cp.array(1.0,cp.float32)

    def _apply(self):
        cp.add(self.t,1,out=self.t)
        cp.power(self.b1,self.t,out=self._b1_pow_t)
        cp.power(self.b2,self.t,out=self._b2_pow_t)
        bc1=cp.maximum(1.0 - self._b1_pow_t, cp.float32(1e-12))
        bc2=cp.maximum(1.0 - self._b2_pow_t, cp.float32(1e-12))
        self._inv_bc1[...] = 1.0 / bc1; self._inv_bc2[...] = 1.0 / bc2
        for s in self.groups:
            _ams_update(s["p"],s["g"],s["m"],s["v"],s["vhat"],
                        lr=self.lr,b1=self.b1,b2=self.b2,eps=self.eps,wd=self.wd,
                        inv_bc1=self._inv_bc1,inv_bc2=self._inv_bc2,
                        grad_scale=self.grad_scale,exempt=s["exempt"])
    step=step_into=_apply

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
                    "m": cp.zeros(p.shape, dtype=cp.float32),
                    "v": cp.zeros(p.shape, dtype=cp.float32),
                    "vhat": cp.zeros(p.shape, dtype=cp.float32),
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
