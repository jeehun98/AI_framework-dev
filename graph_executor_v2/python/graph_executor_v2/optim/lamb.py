from __future__ import annotations
from typing import Iterable, Tuple, List
import cupy as cp
ParamTriplet = Tuple[cp.ndarray, cp.ndarray, bool]

_lamb_fp32 = cp.ElementwiseKernel(
    in_params='''
      float32 p, float32 g, float32 m, float32 v,
      float32 b1, float32 b2, float32 eps,
      float32 inv_bc1, float32 inv_bc2, float32 grad_scale
    ''',
    out_params='float32 m_out, float32 v_out, float32 r_out',
    operation=r'''
      float ge = grad_scale * g;
      float m_new = b1*m + (1.f-b1)*ge;
      float v_new = b2*v + (1.f-b2)*ge*ge;
      float m_hat = m_new * inv_bc1;
      float v_hat = v_new * inv_bc2;
      float r = m_hat / (sqrtf(v_hat) + eps); // adam-like update vector
      m_out=m_new; v_out=v_new; r_out=r;
    ''', name='lamb_phase1_fp32'
)

_lamb_fp16 = cp.ElementwiseKernel(
    in_params='''
      float16 p, float16 g, float32 m, float32 v,
      float32 b1, float32 b2, float32 eps,
      float32 inv_bc1, float32 inv_bc2, float32 grad_scale
    ''',
    out_params='float32 m_out, float32 v_out, float32 r_out',
    operation=r'''
      float gf=(float)g;
      float ge = grad_scale * gf;
      float m_new = b1*m + (1.f-b1)*ge;
      float v_new = b2*v + (1.f-b2)*ge*ge;
      float m_hat = m_new * inv_bc1;
      float v_hat = v_new * inv_bc2;
      float r = m_hat / (sqrtf(v_hat) + eps);
      m_out=m_new; v_out=v_new; r_out=r;
    ''', name='lamb_phase1_fp16'
)

# Phase2: trust ratio & weight update (elementwise scale only; norms are computed separately)
_lamb_apply_fp32 = cp.ElementwiseKernel(
    in_params='''
      float32 p, float32 r, float32 lr, float32 wd, float32 trust, int32 exempt
    ''',
    out_params='float32 p_out',
    operation=r'''
      float decay = (exempt==0) ? (lr * wd * p) : 0.f;
      p_out = p - lr * trust * r - decay;
    ''', name='lamb_apply_fp32'
)

_lamb_apply_fp16 = cp.ElementwiseKernel(
    in_params='''
      float16 p, float32 r, float32 lr, float32 wd, float32 trust, int32 exempt
    ''',
    out_params='float16 p_out',
    operation=r'''
      float pf=(float)p;
      float decay = (exempt==0) ? (lr * wd * pf) : 0.f;
      float w = pf - lr * trust * r - decay;
      p_out=(float16)w;
    ''', name='lamb_apply_fp16'
)

def _phase1(p,g,m,v,*,b1,b2,eps,inv_bc1,inv_bc2,grad_scale):
    if p.dtype==cp.float16:
        m_out,v_out,r = _lamb_fp16(p,g,m,v,b1,b2,eps,inv_bc1,inv_bc2,grad_scale)
    else:
        m_out,v_out,r = _lamb_fp32(p,g,m,v,b1,b2,eps,inv_bc1,inv_bc2,grad_scale)
    m[...] = m_out; v[...] = v_out
    return r  # update vector

def _apply_with_trust(p,r,*,lr,wd,trust,exempt):
    ex=cp.int32(1 if exempt else 0)
    if p.dtype==cp.float16:
        p_out = _lamb_apply_fp16(p,r,lr,wd,trust,ex)
    else:
        p_out = _lamb_apply_fp32(p,r,lr,wd,trust,ex)
    p[...] = p_out

class LAMBOpt:
    def __init__(self, params: Iterable[ParamTriplet], *, lr=1e-3, wd=0.01, beta1=0.9, beta2=0.999, eps=1e-6, trust_clip=(0.0, 10.0)):
        self.groups=[]
        for (p,g,ex) in params:
            assert g.dtype==p.dtype
            self.groups.append({"p":p,"g":g,"exempt":bool(ex),
                                "m":cp.zeros(p.shape,cp.float32),
                                "v":cp.zeros(p.shape,cp.float32)})
        self.lr=cp.array(lr,cp.float32); self.wd=cp.array(wd,cp.float32)
        self.b1=cp.array(beta1,cp.float32); self.b2=cp.array(beta2,cp.float32)
        self.eps=cp.array(eps,cp.float32); self.t=cp.array(0,cp.int32)
        self.grad_scale=cp.array(1.0,cp.float32)
        self._b1_pow_t=cp.array(0.0,cp.float32); self._b2_pow_t=cp.array(0.0,cp.float32)
        self._inv_bc1=cp.array(1.0,cp.float32); self._inv_bc2=cp.array(1.0,cp.float32)
        self.trust_min=cp.array(float(trust_clip[0]),cp.float32)
        self.trust_max=cp.array(float(trust_clip[1]),cp.float32)

    def _apply(self):
        cp.add(self.t,1,out=self.t)
        cp.power(self.b1,self.t,out=self._b1_pow_t)
        cp.power(self.b2,self.t,out=self._b2_pow_t)
        bc1=cp.maximum(1.0 - self._b1_pow_t, cp.float32(1e-12))
        bc2=cp.maximum(1.0 - self._b2_pow_t, cp.float32(1e-12))
        self._inv_bc1[...] = 1.0 / bc1; self._inv_bc2[...] = 1.0 / bc2

        for s in self.groups:
            # Phase 1: adam-like update vector
            r = _phase1(s["p"], s["g"], s["m"], s["v"],
                        b1=self.b1,b2=self.b2,eps=self.eps,
                        inv_bc1=self._inv_bc1,inv_bc2=self._inv_bc2,
                        grad_scale=self.grad_scale)
            # Norms(리덕션): 캡처-세이프, 0-D 텐서로 계산
            p_norm = cp.linalg.norm(s["p"].astype(cp.float32)).astype(cp.float32)
            r_norm = cp.linalg.norm(r.astype(cp.float32)).astype(cp.float32)
            # trust ratio = ||p|| / ||r|| (0 가드)
            trust = cp.where(r_norm > 0, p_norm / r_norm, cp.float32(1.0))
            # optional clip
            trust = cp.clip(trust, self.trust_min, self.trust_max)
            # Phase 2: apply with trust ratio + decoupled wd
            _apply_with_trust(s["p"], r, lr=self.lr, wd=self.wd, trust=trust, exempt=s["exempt"])

    step=step_into=_apply

    def rebind_grads(self, params):
      params = list(params)
      if len(self.groups) == 0:
          for (p, g, ex) in params:
              assert g.shape == p.shape and g.dtype == p.dtype
              self.groups.append({
                  "p": p,
                  "g": g,
                  "exempt": bool(ex),
                  "m": cp.zeros(p.shape, dtype=cp.float32),
                  "v": cp.zeros(p.shape, dtype=cp.float32),
              })
          return
      assert len(params) == len(self.groups), \
          f"param count mismatch: opt={len(self.groups)} vs {len(params)}"
      for s, (p, g, ex) in zip(self.groups, params):
          assert s["p"] is p
          assert g.shape == p.shape and g.dtype == p.dtype
          s["g"] = g
          s["exempt"] = bool(ex)
