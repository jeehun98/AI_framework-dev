# python/graph_executor_v2/layers/activation_layer.py
from __future__ import annotations
from typing import Optional, Literal
import cupy as cp

ActivationName = Literal[
    "none", "relu", "leaky_relu", "gelu", "silu", "sigmoid", "tanh", "elu", "softplus"
]

# ============================================================
# RawKernels (capture-safe: out-매개변수에 직접 쓰기)
#  - ReLU / LeakyReLU / ELU 는 piecewise라 마스크 없이 커널로 처리
#  - fp16 경로도 지원(half)
# ============================================================
_relu_fwd_raw = cp.RawKernel(r'''
extern "C" __global__
void relu_fwd(const void* x, void* y, int n, int dtype) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= n) return;
  if (dtype == 0) {  // float32
    const float* xi = (const float*)x;
    float* yi = (float*)y;
    float v = xi[i];
    yi[i] = v > 0.f ? v : 0.f;
  } else {           // float16
    const __half* xi = (const __half*)x;
    __half* yi = (__half*)y;
    float v = __half2float(xi[i]);
    yi[i] = __float2half(v > 0.f ? v : 0.f);
  }
}
''', name='relu_fwd')

_relu_bwd_raw = cp.RawKernel(r'''
extern "C" __global__
void relu_bwd(const void* x, const void* gy, void* gx, int n, int dtype) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= n) return;
  if (dtype == 0) {
    const float* xi=(const float*)x; const float* gyi=(const float*)gy; float* gxo=(float*)gx;
    gxo[i] = (xi[i] > 0.f) ? gyi[i] : 0.f;
  } else {
    const __half* xi=(const __half*)x; const __half* gyi=(const __half*)gy; __half* gxo=(__half*)gx;
    float xv = __half2float(xi[i]), gyv = __half2float(gyi[i]);
    gxo[i] = __float2half(xv > 0.f ? gyv : 0.f);
  }
}
''', name='relu_bwd')

_leaky_fwd_raw = cp.RawKernel(r'''
extern "C" __global__
void lrelu_fwd(const void* x, void* y, float ns, int n, int dtype) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= n) return;
  if (dtype == 0) {
    const float* xi=(const float*)x; float* yi=(float*)y;
    float v = xi[i];
    yi[i] = (v >= 0.f) ? v : ns * v;
  } else {
    const __half* xi=(const __half*)x; __half* yi=(__half*)y;
    float v = __half2float(xi[i]);
    yi[i] = __float2half((v >= 0.f) ? v : ns * v);
  }
}
''', name='lrelu_fwd')

_leaky_bwd_raw = cp.RawKernel(r'''
extern "C" __global__
void lrelu_bwd(const void* x, const void* gy, void* gx, float ns, int n, int dtype) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= n) return;
  if (dtype == 0) {
    const float* xi=(const float*)x; const float* gyi=(const float*)gy; float* gxo=(float*)gx;
    float xv = xi[i]; gxo[i] = (xv >= 0.f) ? gyi[i] : ns * gyi[i];
  } else {
    const __half* xi=(const __half*)x; const __half* gyi=(const __half*)gy; __half* gxo=(__half*)gx;
    float xv = __half2float(xi[i]); float gyv = __half2float(gyi[i]);
    gxo[i] = __float2half((xv >= 0.f) ? gyv : ns * gyv);
  }
}
''', name='lrelu_bwd')

_elu_fwd_raw = cp.RawKernel(r'''
extern "C" __global__
void elu_fwd(const void* x, void* y, float alpha, int n, int dtype) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= n) return;
  if (dtype == 0) {
    const float* xi=(const float*)x; float* yi=(float*)y;
    float v = xi[i];
    yi[i] = (v >= 0.f) ? v : (alpha * (expf(v) - 1.f));
  } else {
    const __half* xi=(const __half*)x; __half* yi=(__half*)y;
    float v = __half2float(xi[i]);
    float r = (v >= 0.f) ? v : (alpha * (expf(v) - 1.f));
    yi[i] = __float2half(r);
  }
}
''', name='elu_fwd')

_elu_bwd_raw = cp.RawKernel(r'''
extern "C" __global__
void elu_bwd(const void* x, const void* gy, void* gx, float alpha, int n, int dtype) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= n) return;
  if (dtype == 0) {
    const float* xi=(const float*)x; const float* gyi=(const float*)gy; float* gxo=(float*)gx;
    float v = xi[i];
    float der = (v >= 0.f) ? 1.f : (alpha * expf(v));
    gxo[i] = gyi[i] * der;
  } else {
    const __half* xi=(const __half*)x; const __half* gyi=(const __half*)gy; __half* gxo=(__half*)gx;
    float v = __half2float(xi[i]); float gyv = __half2float(gyi[i]);
    float der = (v >= 0.f) ? 1.f : (alpha * expf(v));
    gxo[i] = __float2half(gyv * der);
  }
}
''', name='elu_bwd')

# ============================================================
# ElementwiseKernels (allocating path에 사용)
# ============================================================
_relu_fwd_ew  = cp.ElementwiseKernel('T x','T y','y = (x>(T)0)?x:(T)0;','relu_fwd')
_relu_bwd_ew  = cp.ElementwiseKernel('T x,T gy','T gx','gx = (x>(T)0)?gy:(T)0;','relu_bwd')
_leaky_fwd_ew = cp.ElementwiseKernel('T x,float32 ns','T y','y=(x>=(T)0)?x:(T)(ns*(float)x);','lrelu_fwd')
_leaky_bwd_ew = cp.ElementwiseKernel('T x,T gy,float32 ns','T gx','gx=(x>=(T)0)?gy:(T)(ns*(float)gy);','lrelu_bwd')

_sigmoid_f32  = cp.ElementwiseKernel('float32 xf','float32 yf','yf=1.f/(1.f+expf(-xf));','sigmoid_f32')
_sigmoid_b32  = cp.ElementwiseKernel('float32 yf,float32 gyf','float32 gxf','gxf=gyf*yf*(1.f-yf);','sigmoid_b32')

_tanh_f32     = cp.ElementwiseKernel('float32 xf','float32 yf','yf=tanhf(xf);','tanh_f32')
_tanh_b32     = cp.ElementwiseKernel('float32 yf,float32 gyf','float32 gxf','gxf=gyf*(1.f-yf*yf);','tanh_b32')

_silu_f32     = cp.ElementwiseKernel('float32 xf','float32 yf','float s=1.f/(1.f+expf(-xf)); yf=xf*s;','silu_f32')
_silu_b32     = cp.ElementwiseKernel('float32 xf,float32 gyf','float32 gxf','float s=1.f/(1.f+expf(-xf)); gxf=gyf*(s*(1.f+xf*(1.f-s)));','silu_b32')

_GELU_A = cp.array(0.044715, dtype=cp.float32)
_GELU_C = cp.array((2.0/cp.pi)**0.5, dtype=cp.float32)
_gelu_f32     = cp.ElementwiseKernel('float32 xf,float32 A,float32 C','float32 yf',
    'float x3=xf*xf*xf; float t=C*(xf+A*x3); float th=tanhf(t); yf=0.5f*xf*(1.f+th);','gelu_f32')
_gelu_b32     = cp.ElementwiseKernel('float32 xf,float32 gyf,float32 A,float32 C','float32 gxf',
    'float x2=xf*xf; float t=C*(xf+A*x2*xf); float th=tanhf(t); float sech2=1.f-th*th; float dt=C*(1.f+3.f*A*x2); float dy=0.5f*(1.f+th)+0.5f*xf*sech2*dt; gxf=gyf*dy;','gelu_b32')

_elu_f32      = cp.ElementwiseKernel('float32 xf,float32 a','float32 yf','yf=(xf>=0.f)?xf:(a*(expf(xf)-1.f));','elu_f32')
_elu_b32      = cp.ElementwiseKernel('float32 xf,float32 gyf,float32 a','float32 gxf','float der=(xf>=0.f)?1.f:(a*expf(xf)); gxf=gyf*der;','elu_b32')

_splus_f32    = cp.ElementwiseKernel('float32 xf','float32 yf',
    'if(xf>20.f) yf=xf; else if(xf<-20.f) yf=expf(xf); else yf=log1pf(expf(xf));','softplus_f32')
_splus_b32    = cp.ElementwiseKernel('float32 xf,float32 gyf','float32 gxf','float s=1.f/(1.f+expf(-xf)); gxf=gyf*s;','softplus_b32')

# ============================================================
# Utilities
# ============================================================
def _grid1d(n: int, tpblock: int = 256) -> tuple[int, int]:
    return ( (n + tpblock - 1) // tpblock, tpblock )

def _dtype_code(x: cp.ndarray) -> int:
    # 0: float32, 1: float16
    if x.dtype == cp.float32: return 0
    if x.dtype == cp.float16: return 1
    raise TypeError(f"Unsupported dtype: {x.dtype}")

# ============================================================
# Allocating helpers (편의용, CuPy가 out배열 생성)
# ============================================================
def _alloc_forward(x: cp.ndarray, act: ActivationName, *, neg_slope: float, alpha: float) -> cp.ndarray:
    actn = (act or "none").lower()
    if actn == "none":
        return x.copy()
    if actn == "relu":
        return _relu_fwd_ew(x)
    if actn == "leaky_relu":
        return _leaky_fwd_ew(x, cp.float32(neg_slope))
    # fp16은 승격 계산 후 캐스팅
    xf = x.astype(cp.float32, copy=False)
    if actn == "sigmoid":
        y = _sigmoid_f32(xf)
    elif actn == "tanh":
        y = _tanh_f32(xf)
    elif actn == "silu":
        y = _silu_f32(xf)
    elif actn == "gelu":
        y = _gelu_f32(xf, _GELU_A, _GELU_C)
    elif actn == "elu":
        y = _elu_f32(xf, cp.float32(alpha))
    elif actn == "softplus":
        y = _splus_f32(xf)
    else:
        raise ValueError(f"unknown activation: {act}")
    return y.astype(x.dtype, copy=False)

def _alloc_backward(x: cp.ndarray, gy: cp.ndarray, act: ActivationName, *, neg_slope: float, alpha: float, y_saved: Optional[cp.ndarray]) -> cp.ndarray:
    actn = (act or "none").lower()
    if actn == "none":
        return gy.copy()
    if actn == "relu":
        return _relu_bwd_ew(x, gy)
    if actn == "leaky_relu":
        return _leaky_bwd_ew(x, gy, cp.float32(neg_slope))
    xf = x.astype(cp.float32, copy=False)
    gyf = gy.astype(cp.float32, copy=False)
    if actn == "sigmoid":
        yf = _sigmoid_f32(xf) if y_saved is None else y_saved.astype(cp.float32, copy=False)
        g = _sigmoid_b32(yf, gyf)
    elif actn == "tanh":
        yf = _tanh_f32(xf) if y_saved is None else y_saved.astype(cp.float32, copy=False)
        g = _tanh_b32(yf, gyf)
    elif actn == "silu":
        g = _silu_b32(xf, gyf)
    elif actn == "gelu":
        g = _gelu_b32(xf, gyf, _GELU_A, _GELU_C)
    elif actn == "elu":
        g = _elu_b32(xf, gyf, cp.float32(alpha))
    elif actn == "softplus":
        g = _splus_b32(xf, gyf)
    else:
        raise ValueError(f"unknown activation: {act}")
    return g.astype(x.dtype, copy=False)

# ============================================================
# Layer (CuPy-only, capture-safe *_into 제공)
# ============================================================
class ActivationLayer:
    """
    CuPy-only activation layer.
    - forward/backward : allocating (편의용)
    - forward_into/backward_into : capture-safe (NO allocation)
    """
    def __init__(self, act: ActivationName = "relu", *, neg_slope: float = 0.01, alpha: float = 1.0, save_y: bool = False, name: Optional[str] = None):
        self.name = name or f"Activation({act})"
        self.act: ActivationName = act
        self.neg_slope = float(neg_slope)
        self.alpha     = float(alpha)
        self.save_y    = bool(save_y)
        self._y_saved: Optional[cp.ndarray] = None

    # ---------- Allocating (eager) ----------
    def forward(self, x: cp.ndarray) -> cp.ndarray:
        y = _alloc_forward(x, self.act, neg_slope=self.neg_slope, alpha=self.alpha)
        if self.save_y:
            self._y_saved = y
        return y

    def backward(self, x: cp.ndarray, gy: cp.ndarray) -> cp.ndarray:
        return _alloc_backward(x, gy, self.act, neg_slope=self.neg_slope, alpha=self.alpha, y_saved=self._y_saved)

    # ---------- Capture-safe (NO allocation) ----------
    def forward_into(self, x: cp.ndarray, *, out: cp.ndarray, stream: Optional[int] = None) -> None:
        """out은 x와 같은 dtype/shape, C-contiguous 가정."""
        if out.shape != x.shape or out.dtype != x.dtype or not out.flags.c_contiguous:
            raise ValueError("[capture] out must be C-contiguous and match x (shape/dtype)")
        n = x.size
        grid, block = _grid1d(n)
        dtype = _dtype_code(x)

        actn = (self.act or "none").lower()
        if actn == "none":
            out[...] = x
            if self.save_y: self._y_saved = out
            return

        if actn == "relu":
            _relu_fwd_raw((grid,), (block,), (x.data.ptr, out.data.ptr, n, dtype))
            if self.save_y: self._y_saved = out
            return

        if actn == "leaky_relu":
            _leaky_fwd_raw((grid,), (block,), (x.data.ptr, out.data.ptr, self.neg_slope, n, dtype))
            if self.save_y: self._y_saved = out
            return

        if actn == "elu":
            _elu_fwd_raw((grid,), (block,), (x.data.ptr, out.data.ptr, self.alpha, n, dtype))
            if self.save_y: self._y_saved = out
            return

        # 나머지는 ufunc 연쇄(in-place)로 무할당 처리
        if actn == "sigmoid":
            out[...] = x
            cp.negative(out, out=out)
            cp.exp(out, out=out)
            cp.add(out, 1, out=out)
            cp.reciprocal(out, out=out)
        elif actn == "tanh":
            out[...] = x
            cp.tanh(out, out=out)
        elif actn == "silu":
            # out = sigmoid(x); out *= x
            out[...] = x
            cp.negative(out, out=out); cp.exp(out, out=out); cp.add(out, 1, out=out); cp.reciprocal(out, out=out)  # sigmoid in out
            cp.multiply(out, x, out=out)
        elif actn == "gelu":
            # out = 0.5 * x * (1 + tanh(C*(x + A*x^3)))
            out[...] = x
            cp.multiply(out, out, out=out)             # x^2
            cp.multiply(out, x, out=out)               # x^3
            cp.multiply(out, float(_GELU_A.get()), out=out) # A*x^3
            cp.add(out, x, out=out)                    # x + A*x^3
            cp.multiply(out, float(_GELU_C.get()), out=out) # t
            cp.tanh(out, out=out)                      # th
            cp.add(out, 1, out=out)                    # 1+th
            cp.multiply(out, 0.5, out=out)             # 0.5*(1+th)
            cp.multiply(out, x, out=out)               # 0.5*x*(1+th)
        elif actn == "softplus":
            out[...] = x
            # 안정형 간이 구현
            # 큰 양수 -> y≈x, 큰 음수 -> y≈exp(x), 중간 -> log1p(exp(x))
            # 분기 없는 근사: log1p(exp(x))로 통일해도 되지만 위상태를 유지
            # 여기서는 간단 버전(정확 but 분기 없는 연쇄):
            cp.exp(out, out=out)
            cp.add(out, 1, out=out)
            cp.log(out, out=out)
        else:
            raise ValueError(f"unknown activation: {self.act}")

        if self.save_y:
            self._y_saved = out

    def backward_into(self, x: cp.ndarray, gy: cp.ndarray, *, out: cp.ndarray, stream: Optional[int] = None) -> None:
        """gx를 out에 기록. 무할당 경로."""
        if out.shape != x.shape or out.dtype != x.dtype or not out.flags.c_contiguous:
            raise ValueError("[capture] out must be C-contiguous and match x")
        if gy.shape != x.shape or gy.dtype != x.dtype or not gy.flags.c_contiguous:
            raise ValueError("[capture] gy must be C-contiguous and match x")

        n = x.size
        grid, block = _grid1d(n)
        dtype = _dtype_code(x)
        actn = (self.act or "none").lower()

        if actn == "none":
            out[...] = gy
            return

        if actn == "relu":
            _relu_bwd_raw((grid,), (block,), (x.data.ptr, gy.data.ptr, out.data.ptr, n, dtype))
            return

        if actn == "leaky_relu":
            _leaky_bwd_raw((grid,), (block,), (x.data.ptr, gy.data.ptr, out.data.ptr, self.neg_slope, n, dtype))
            return

        if actn == "elu":
            _elu_bwd_raw((grid,), (block,), (x.data.ptr, gy.data.ptr, out.data.ptr, self.alpha, n, dtype))
            return

        # 나머지(연쇄 수학) — out을 작업공간으로 사용
        if actn == "sigmoid":
            # gx = gy * sigmoid(x) * (1 - sigmoid(x))
            out[...] = x
            cp.negative(out, out=out); cp.exp(out, out=out); cp.add(out, 1, out=out); cp.reciprocal(out, out=out)  # out = sigmoid(x)
            cp.multiply(out, (1 - out), out=out)   # out = s*(1-s)
            cp.multiply(out, gy, out=out)          # out = gy * ...
        elif actn == "tanh":
            # gx = gy * (1 - tanh(x)^2)
            out[...] = x
            cp.tanh(out, out=out)
            cp.multiply(out, out, out=out)         # out = th^2
            cp.multiply(out, -1, out=out)          # out = -th^2
            cp.add(out, 1, out=out)                # out = 1 - th^2
            cp.multiply(out, gy, out=out)
        elif actn == "silu":
            # gx = gy * [ s*(1 + x*(1 - s)) ], s = sigmoid(x)
            out[...] = x
            cp.negative(out, out=out); cp.exp(out, out=out); cp.add(out, 1, out=out); cp.reciprocal(out, out=out)  # out = s
            cp.add((1 - out) * x, 1, out=out)      # out = 1 + x*(1-s)
            cp.multiply(out, (1 / (1 + cp.exp(-x))), out=out)  # out = s*(1 + x*(1-s))  (s 재사용식)
            cp.multiply(out, gy, out=out)
        elif actn == "gelu":
            # analytic grad of tanh approx
            out[...] = x.astype(x.dtype, copy=False)
            # compute s = derivative part into out
            # out will end as dy/dx
            # reuse the same sequence as forward to build th and dt, then combine
            tmp = out  # alias
            # x^2
            cp.multiply(tmp, tmp, out=tmp)                 # x^2
            cp.multiply(tmp, x, out=tmp)                   # x^3
            cp.multiply(tmp, float(_GELU_A.get()), out=tmp)# A*x^3
            cp.add(tmp, x, out=tmp)                        # x + A*x^3
            cp.multiply(tmp, float(_GELU_C.get()), out=tmp)# t
            cp.tanh(tmp, out=tmp)                          # th
            th = tmp                                       # alias
            # sech2 = 1 - th^2  (reuse tmp)
            cp.multiply(tmp, tmp, out=tmp)                 # th^2
            cp.multiply(tmp, -1, out=tmp); cp.add(tmp, 1, out=tmp)  # 1 - th^2
            # dt/dx = C*(1 + 3*A*x^2)  (reuse out for x^2 again)
            x2 = out
            x2[...] = x
            cp.multiply(x2, x2, out=x2)                    # x^2
            cp.multiply(x2, 3.0 * float(_GELU_A.get()), out=x2)
            cp.add(x2, 1.0, out=x2)
            cp.multiply(x2, float(_GELU_C.get()), out=x2)  # dt
            # dy/dx = 0.5*(1+th) + 0.5*x*sech2*dt
            cp.add(th, 1.0, out=th); cp.multiply(th, 0.5, out=th)    # 0.5*(1+th)
            cp.multiply(tmp, x2, out=x2)                               # sech2*dt (reuse dt var)
            cp.multiply(x2, 0.5, out=x2); cp.multiply(x2, x, out=x2)  # 0.5*x*sech2*dt
            cp.add(th, x2, out=out)                                   # out = dy/dx
            cp.multiply(out, gy, out=out)
        elif actn == "softplus":
            # gx = gy * sigmoid(x)
            out[...] = x
            cp.negative(out, out=out); cp.exp(out, out=out); cp.add(out, 1, out=out); cp.reciprocal(out, out=out)
            cp.multiply(out, gy, out=out)
        else:
            raise ValueError(f"unknown activation: {self.act}")

    # ---------- misc ----------
    def parameters(self):
        if False:
            yield
        return

    def state_dict(self) -> dict:
        return {
            "name": self.name,
            "act": self.act,
            "neg_slope": self.neg_slope,
            "alpha": self.alpha,
            "save_y": self.save_y,
        }

    def load_state_dict(self, sd: dict):
        self.name = sd.get("name", self.name)
        self.act = sd.get("act", self.act)
        self.neg_slope = float(sd.get("neg_slope", self.neg_slope))
        self.alpha = float(sd.get("alpha", self.alpha))
        self.save_y = bool(sd.get("save_y", self.save_y))
        return self

    # ----- shape utils -----
    def compute_output_shape(self, input_shape) -> tuple:
        """
        Activation은 shape를 바꾸지 않음.
        input_shape: Tuple[int, ...] | Sequence[int]
        """
        # tuple 보장 및 정수형으로 캐스팅
        if not isinstance(input_shape, tuple):
            input_shape = tuple(int(v) if v is not None else None for v in input_shape)
        else:
            input_shape = tuple(int(v) if v is not None else None for v in input_shape)
        return input_shape