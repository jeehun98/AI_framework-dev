from __future__ import annotations
from typing import Iterable, Tuple, List
import cupy as cp
ParamTriplet = Tuple[cp.ndarray, cp.ndarray, bool]

_rms_fp32 = cp.ElementwiseKernel(
    in_params='''
      float32 p, float32 g, float32 sq,
      float32 lr, float32 alpha, float32 eps, float32 wd,
      float32 grad_scale, int32 exempt
    ''',
    out_params='float32 p_out, float32 sq_out',
    operation=r'''
      float ge = grad_scale * g;
      float sq_new = alpha * sq + (1.f - alpha) * ge * ge;
      float upd = ge / (sqrtf(sq_new) + eps);
      float decay = (exempt==0) ? (lr * wd * p) : 0.f;
      p_out = p - lr * upd - decay;
      sq_out = sq_new;
    ''', name='rmsprop_fused_fp32'
)

_rms_fp16 = cp.ElementwiseKernel(
    in_params='''
      float16 p, float16 g, float32 sq,
      float32 lr, float32 alpha, float32 eps, float32 wd,
      float32 grad_scale, int32 exempt
    ''',
    out_params='float16 p_out, float32 sq_out',
    operation=r'''
      float pf=(float)p, gf=(float)g;
      float ge = grad_scale * gf;
      float sq_new = alpha * sq + (1.f - alpha) * ge * ge;
      float upd = ge / (sqrtf(sq_new) + eps);
      float decay = (exempt==0) ? (lr * wd * pf) : 0.f;
      float w = pf - lr * upd - decay;
      p_out=(float16)w; sq_out=sq_new;
    ''', name='rmsprop_fused_fp16'
)

def _rms_update(p,g,sq,*,lr,alpha,eps,wd,grad_scale,exempt):
    ex=cp.int32(1 if exempt else 0)
    if p.dtype==cp.float16:
        p_out,sq_out=_rms_fp16(p,g,sq,lr,alpha,eps,wd,grad_scale,ex)
    else:
        p_out,sq_out=_rms_fp32(p,g,sq,lr,alpha,eps,wd,grad_scale,ex)
    p[...] = p_out; sq[...] = sq_out

class RMSpropOpt:
    def __init__(self, params: Iterable[ParamTriplet], *, lr=1e-3, alpha=0.99, eps=1e-8, wd=0.0):
        self.groups=[]; 
        for (p,g,ex) in params:
            assert g.dtype==p.dtype
            self.groups.append({"p":p,"g":g,"exempt":bool(ex),"sq":cp.zeros(p.shape,cp.float32)})
        self.lr=cp.array(lr,cp.float32); self.alpha=cp.array(alpha,cp.float32)
        self.eps=cp.array(eps,cp.float32); self.wd=cp.array(wd,cp.float32)
        self.grad_scale=cp.array(1.0,cp.float32); self.t=cp.array(0,cp.int32)

    def _apply(self):
        cp.add(self.t,1,out=self.t)
        for s in self.groups:
            _rms_update(s["p"],s["g"],s["sq"], lr=self.lr, alpha=self.alpha, eps=self.eps,
                        wd=self.wd, grad_scale=self.grad_scale, exempt=s["exempt"])
    step=step_into=_apply
        
    def rebind_grads(self, params):
        params = list(params)
        if len(self.groups) == 0:
            # 빈 상태 → 캡처 플랜 기준으로 최초 초기화
            for (p, g, ex) in params:
                assert isinstance(p, cp.ndarray) and isinstance(g, cp.ndarray)
                assert g.shape == p.shape and g.dtype == p.dtype
                self.groups.append({
                    "p": p,
                    "g": g,
                    "exempt": bool(ex),
                    "sq": cp.zeros(p.shape, dtype=cp.float32),  # ← 상태 텐서 생성
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
