from __future__ import annotations
from typing import Iterable, Tuple, List
import cupy as cp

ParamTriplet = Tuple[cp.ndarray, cp.ndarray, bool]

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
      float step_vec = (nesterov ? (momentum * v_new + (1.f - damp) * ge) : v_new);
      float decay = (exempt==0) ? (lr * wd * p) : 0.f;
      p_out = p - lr * step_vec - decay;
      v_out = v_new;
    ''',
    name='sgd_fused_fp32'
)

_sgd_fp16 = cp.ElementwiseKernel(
    in_params='''
      float16 p, float16 g, float32 v,
      float32 lr, float32 momentum, float32 damp, int32 nesterov,
      float32 wd, int32 exempt, float32 grad_scale
    ''',
    out_params='float16 p_out, float32 v_out',
    operation=r'''
      float pf = (float)p, gf=(float)g;
      float ge = grad_scale * gf;
      float v_new = momentum * v + (1.f - damp) * ge;
      float step_vec = (nesterov ? (momentum * v_new + (1.f - damp) * ge) : v_new);
      float decay = (exempt==0) ? (lr * wd * pf) : 0.f;
      float w = pf - lr * step_vec - decay;
      p_out=(float16)w; v_out=v_new;
    ''',
    name='sgd_fused_fp16'
)

def _sgd_update(p,g,v,*,lr,momentum,damp,nesterov,wd,exempt,grad_scale):
    ex = cp.int32(1 if exempt else 0)
    ne = cp.int32(1 if nesterov else 0)
    if p.dtype == cp.float16:
        p_out, v_out = _sgd_fp16(p,g,v,lr,momentum,damp,ne,wd,ex,grad_scale)
    else:
        p_out, v_out = _sgd_fp32(p,g,v,lr,momentum,damp,ne,wd,ex,grad_scale)
    p[...] = p_out; v[...] = v_out

class SGDOpt:
    def __init__(self, params: Iterable[ParamTriplet], *, lr=1e-2, momentum=0.9, nesterov=True, damp=0.0, wd=0.0):
        self.groups: List[dict] = []
        for (p,g,ex) in params:
            assert g.dtype == p.dtype
            self.groups.append({"p":p,"g":g,"exempt":bool(ex),"v":cp.zeros(p.shape,cp.float32)})
        self.lr=cp.array(lr,cp.float32); self.momentum=cp.array(momentum,cp.float32)
        self.nesterov=bool(nesterov); self.damp=cp.array(damp,cp.float32)
        self.wd=cp.array(wd,cp.float32); self.grad_scale=cp.array(1.0,cp.float32)
        self.t=cp.array(0,cp.int32)

    def _apply(self):
        cp.add(self.t,1,out=self.t)
        for s in self.groups:
            _sgd_update(s["p"],s["g"],s["v"], lr=self.lr, momentum=self.momentum, damp=self.damp,
                        nesterov=self.nesterov, wd=self.wd, exempt=s["exempt"], grad_scale=self.grad_scale)
    step = step_into = _apply
    
    def rebind_grads(self, params):
        params = list(params)
        if len(self.groups) == 0:
            for (p,g,ex) in params:
                assert g.dtype == p.dtype
                self.groups.append({"p":p,"g":g,"exempt":bool(ex),
                                    "v":cp.zeros(p.shape, cp.float32)})
            return
        assert len(params) == len(self.groups)
        for s,(p,g,ex) in zip(self.groups, params):
            assert s["p"] is p
            assert g.shape == p.shape and g.dtype == p.dtype
            s["g"] = g; s["exempt"] = bool(ex)

    