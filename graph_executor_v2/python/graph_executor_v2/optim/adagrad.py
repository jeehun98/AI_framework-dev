from __future__ import annotations
from typing import Iterable, Tuple, List
import cupy as cp
ParamTriplet = Tuple[cp.ndarray, cp.ndarray, bool]

_adagrad_fp32 = cp.ElementwiseKernel(
    in_params='''
      float32 p, float32 g, float32 acc,
      float32 lr, float32 eps, float32 wd, float32 grad_scale, int32 exempt
    ''',
    out_params='float32 p_out, float32 acc_out',
    operation=r'''
      float ge = grad_scale * g;
      float acc_new = acc + ge * ge;
      float upd = ge / (sqrtf(acc_new) + eps);
      float decay = (exempt==0) ? (lr * wd * p) : 0.f;
      p_out = p - lr * upd - decay;
      acc_out = acc_new;
    ''', name='adagrad_fused_fp32'
)

_adagrad_fp16 = cp.ElementwiseKernel(
    in_params='''
      float16 p, float16 g, float32 acc,
      float32 lr, float32 eps, float32 wd, float32 grad_scale, int32 exempt
    ''',
    out_params='float16 p_out, float32 acc_out',
    operation=r'''
      float pf=(float)p, gf=(float)g;
      float ge = grad_scale * gf;
      float acc_new = acc + ge * ge;
      float upd = ge / (sqrtf(acc_new) + eps);
      float decay = (exempt==0) ? (lr * wd * pf) : 0.f;
      float w = pf - lr * upd - decay;
      p_out=(float16)w; acc_out=acc_new;
    ''', name='adagrad_fused_fp16'
)

def _adagrad_update(p,g,acc,*,lr,eps,wd,grad_scale,exempt):
    ex=cp.int32(1 if exempt else 0)
    if p.dtype==cp.float16:
        p_out,acc_out=_adagrad_fp16(p,g,acc,lr,eps,wd,grad_scale,ex)
    else:
        p_out,acc_out=_adagrad_fp32(p,g,acc,lr,eps,wd,grad_scale,ex)
    p[...] = p_out; acc[...] = acc_out

class AdagradOpt:
    def __init__(self, params: Iterable[ParamTriplet], *, lr=1e-2, eps=1e-10, wd=0.0):
        self.groups=[]
        for (p,g,ex) in params:
            assert g.dtype==p.dtype
            self.groups.append({"p":p,"g":g,"exempt":bool(ex),"acc":cp.zeros(p.shape,cp.float32)})
        self.lr=cp.array(lr,cp.float32); self.eps=cp.array(eps,cp.float32)
        self.wd=cp.array(wd,cp.float32); self.grad_scale=cp.array(1.0,cp.float32)
        self.t=cp.array(0,cp.int32)

    def _apply(self):
        cp.add(self.t,1,out=self.t)
        for s in self.groups:
            _adagrad_update(s["p"],s["g"],s["acc"], lr=self.lr, eps=self.eps, wd=self.wd,
                            grad_scale=self.grad_scale, exempt=s["exempt"])
    step=step_into=_apply

    def rebind_grads(self, params):
        params = list(params)
        if len(self.groups) == 0:
            # 빈 상태 → 첫 rebind 시 전체 초기화
            for (p, g, ex) in params:
                assert g.shape == p.shape and g.dtype == p.dtype
                self.groups.append({
                    "p": p,
                    "g": g,
                    "exempt": bool(ex),
                    "acc": cp.zeros(p.shape, dtype=cp.float32),  # ← 상태 텐서 생성
                })
            return
        # 기존 그룹이 있다면 포인터 동일성/정합성 확인 후 grad만 교체
        assert len(params) == len(self.groups), \
            f"param count mismatch: opt={len(self.groups)} vs {len(params)}"
        for s, (p, g, ex) in zip(self.groups, params):
            assert s["p"] is p
            assert g.shape == p.shape and g.dtype == p.dtype
            s["g"] = g
            s["exempt"] = bool(ex)