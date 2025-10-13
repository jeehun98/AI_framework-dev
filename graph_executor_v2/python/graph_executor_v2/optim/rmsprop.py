# graph_executor_v2/optim/rmsprop.py
import cupy as cp
ParamTriplet = tuple[cp.ndarray, cp.ndarray, bool]

_rms_fused_fp32 = cp.ElementwiseKernel(
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
      float w = p - lr * upd - decay;
      p_out = w; sq_out = sq_new;
    ''',
    name='rmsprop_fused_fp32'
)

_rms_fused_fp16 = cp.ElementwiseKernel(
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
    ''',
    name='rmsprop_fused_fp16'
)

def _rms_update(p,g,sq,*,lr,alpha,eps,wd,grad_scale,exempt):
    ex = cp.int32(1 if exempt else 0)
    if p.dtype==cp.float16:
        p_out,sq_out = _rms_fused_fp16(p,g,sq,lr,alpha,eps,wd,grad_scale,ex)
    else:
        p_out,sq_out = _rms_fused_fp32(p,g,sq,lr,alpha,eps,wd,grad_scale,ex)
    p[...] = p_out; sq[...] = sq_out

class RMSpropOpt:
    def __init__(self, params, *, lr=1e-3, alpha=0.99, eps=1e-8, wd=0.0):
        self.groups=[]
        for (p,g,ex) in params:
            assert g.dtype==p.dtype
            self.groups.append({"p":p,"g":g,"exempt":bool(ex),"sq":cp.zeros(p.shape, cp.float32)})
        self.lr=cp.array(lr,cp.float32)
        self.alpha=cp.array(alpha,cp.float32)
        self.eps=cp.array(eps,cp.float32)
        self.wd=cp.array(wd,cp.float32)
        self.grad_scale=cp.array(1.0,cp.float32)
        self.t=cp.array(0,cp.int32)

    def _apply(self):
        cp.add(self.t,1,out=self.t)
        for s in self.groups:
            _rms_update(s["p"], s["g"], s["sq"],
                        lr=self.lr, alpha=self.alpha, eps=self.eps, wd=self.wd,
                        grad_scale=self.grad_scale, exempt=s["exempt"])
    step = step_into = _apply
    def rebind_grads(self,params):
        assert len(params)==len(self.groups)
        for s,(p,g,ex) in zip(self.groups,params):
            assert s["p"] is p; assert g.shape==p.shape and g.dtype==p.dtype
            s["g"]=g; s["exempt"]=bool(ex)
