# python/graph_executor_v2/layers/activation_layer.py
from __future__ import annotations
from typing import Optional, Literal
import math
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
#include <cuda_fp16.h>
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
''', name='relu_fwd', options=('-std=c++14',))

_relu_bwd_raw = cp.RawKernel(r'''
#include <cuda_fp16.h>
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
''', name='relu_bwd', options=('-std=c++14',))

_leaky_fwd_raw = cp.RawKernel(r'''
#include <cuda_fp16.h>
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
''', name='lrelu_fwd', options=('-std=c++14',))

_leaky_bwd_raw = cp.RawKernel(r'''
#include <cuda_fp16.h>
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
''', name='lrelu_bwd', options=('-std=c++14',))

_elu_fwd_raw = cp.RawKernel(r'''
#include <cuda_fp16.h>
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
''', name='elu_fwd', options=('-std=c++14',))

_elu_bwd_raw = cp.RawKernel(r'''
#include <cuda_fp16.h>
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
''', name='elu_bwd', options=('-std=c++14',))

_relu_bwd_fromy_raw = cp.RawKernel(r'''
#include <cuda_fp16.h>
extern "C" __global__
void relu_bwd_fromy(const void* y, const void* gy, void* gx, int n, int dtype) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= n) return;
  if (dtype == 0) {
    const float* yi=(const float*)y; const float* gyi=(const float*)gy; float* gxo=(float*)gx;
    gxo[i] = (yi[i] > 0.f) ? gyi[i] : 0.f;
  } else {
    const __half* yi=(const __half*)y; const __half* gyi=(const __half*)gy; __half* gxo=(__half*)gx;
    float yv = __half2float(yi[i]), gyv = __half2float(gyi[i]);
    gxo[i] = __float2half(yv > 0.f ? gyv : 0.f);
  }
}
''', name='relu_bwd_fromy', options=('-std=c++14',))

_leaky_bwd_fromy_raw = cp.RawKernel(r'''
#include <cuda_fp16.h>
extern "C" __global__
void lrelu_bwd_fromy(const void* y, const void* gy, void* gx, float ns, int n, int dtype) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= n) return;
  if (dtype == 0) {
    const float* yi=(const float*)y; const float* gyi=(const float*)gy; float* gxo=(float*)gx;
    float yv = yi[i]; gxo[i] = (yv >= 0.f) ? gyi[i] : ns * gyi[i];
  } else {
    const __half* yi=(const __half*)y; const __half* gyi=(const __half*)gy; __half* gxo=(__half*)gx;
    float yv = __half2float(yi[i]); float gyv = __half2float(gyi[i]);
    gxo[i] = __float2half((yv >= 0.f) ? gyv : ns * gyv);
  }
}
''', name='lrelu_bwd_fromy', options=('-std=c++14',))

_elu_bwd_fromy_raw = cp.RawKernel(r'''
#include <cuda_fp16.h>
extern "C" __global__
void elu_bwd_fromy(const void* y, const void* gy, void* gx, float alpha, int n, int dtype) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= n) return;
  if (dtype == 0) {
    const float* yi=(const float*)y; const float* gyi=(const float*)gy; float* gxo=(float*)gx;
    float yv = yi[i];
    float der = (yv >= 0.f) ? 1.f : (yv + alpha);  // alpha*exp(x) = y + alpha
    gxo[i] = gyi[i] * der;
  } else {
    const __half* yi=(const __half*)y; const __half* gyi=(const __half*)gy; __half* gxo=(__half*)gx;
    float yv = __half2float(yi[i]); float gyv = __half2float(gyi[i]);
    float der = (yv >= 0.f) ? 1.f : (yv + alpha);
    gxo[i] = __float2half(gyv * der);
  }
}
''', name='elu_bwd_fromy', options=('-std=c++14',))


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

# ---- GELU 상수: Host-side (D2H 없이 사용) ----
_GELU_A = 0.044715
_GELU_C = math.sqrt(2.0 / math.pi)

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
    return ((n + tpblock - 1) // tpblock, tpblock)

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
        y = _gelu_f32(xf, cp.float32(_GELU_A), cp.float32(_GELU_C))
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
        g = _gelu_b32(xf, gyf, cp.float32(_GELU_A), cp.float32(_GELU_C))
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
        self._z_saved: Optional[cp.ndarray] = None  # pre-activation stash (for bwd fallback)

    # ---------- convenience ----------
    def __call__(self, x: cp.ndarray):
        """Sequential 등에서 layer(x) 형태로 호출될 때 allocating 경로로 수행"""
        return self.forward(x)

    # ---------- Allocating (eager) ----------
    def forward(self, x: cp.ndarray) -> cp.ndarray:
        y = _alloc_forward(x, self.act, neg_slope=self.neg_slope, alpha=self.alpha)
        if self.save_y:
            self._y_saved = y
        return y

    def backward(self, x: cp.ndarray, gy: cp.ndarray) -> cp.ndarray:
        return _alloc_backward(x, gy, self.act, neg_slope=self.neg_slope, alpha=self.alpha, y_saved=self._y_saved)

    # ---------- Capture-safe (NO allocation) ----------
    def forward_into(
        self,
        x: cp.ndarray,
        *,
        out: cp.ndarray,
        z_out: Optional[cp.ndarray] = None,
        stream: Optional[int] = None,
        **kwargs,
    ) -> None:
        """out은 x와 같은 dtype/shape, C-contiguous 가정."""
        if out.shape != x.shape or out.dtype != x.dtype or not out.flags.c_contiguous:
            raise ValueError("[capture] out must be C-contiguous and match x (shape/dtype)")
        # pre-activation 보존:
        # 1) z_out이 주어지면 D2D 복사해서 안전하게 저장
        # 2) 아니면, 추가 할당 없이 입력 x 텐서를 **참조로** stash (캡처-세이프)
        if z_out is not None:
            if z_out.shape != x.shape or z_out.dtype != x.dtype or not z_out.flags.c_contiguous:
                raise ValueError("[capture] z_out must be C-contiguous and match x (shape/dtype)")
            z_out[...] = x
            self._z_saved = z_out
        else:
            self._z_saved = x  # alias only (no copy, no alloc)

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
            cp.negative(out, out=out); cp.exp(out, out=out); cp.add(out, 1, out=out); cp.reciprocal(out, out=out)
        elif actn == "tanh":
            out[...] = x
            cp.tanh(out, out=out)
        elif actn == "silu":
            out[...] = x
            cp.negative(out, out=out); cp.exp(out, out=out); cp.add(out, 1, out=out); cp.reciprocal(out, out=out)  # sigmoid
            cp.multiply(out, x, out=out)
        elif actn == "gelu":
            out[...] = x
            cp.multiply(out, out, out=out)      # x^2
            cp.multiply(out, x, out=out)        # x^3
            cp.multiply(out, _GELU_A, out=out)  # A*x^3
            cp.add(out, x, out=out)             # x + A*x^3
            cp.multiply(out, _GELU_C, out=out)  # t
            cp.tanh(out, out=out)               # th
            cp.add(out, 1, out=out)             # 1+th
            cp.multiply(out, 0.5, out=out)
            cp.multiply(out, x, out=out)
        elif actn == "softplus":
            out[...] = x
            cp.exp(out, out=out); cp.add(out, 1, out=out); cp.log(out, out=out)
        else:
            raise ValueError(f"unknown activation: {self.act}")

        if self.save_y:
            self._y_saved = out

    def backward_into(
        self,
        gy: cp.ndarray,
        *,
        out: Optional[cp.ndarray] = None,
        x_in: Optional[cp.ndarray] = None,
        z_in: Optional[cp.ndarray] = None,
        stream: Optional[int] = None,
        gA_out: Optional[cp.ndarray] = None,
        gB_out: Optional[cp.ndarray] = None,
        **kwargs,
    ) -> None:
        """
        gx를 out(or gy)에 기록. 무할당 경로.
        프레임워크 호환:
          - 호출 형태: backward_into(g_in, gA_out=..., ...)
          - x(또는 z)가 필요하면 z_in 또는 forward에서 stash한 self._z_saved 사용
        """
        # out 미지정이면 gy에 in-place로 계산
        if out is None:
            out = gy
        if out.shape != gy.shape or out.dtype != gy.dtype or not out.flags.c_contiguous:
            raise ValueError("[capture] out must be C-contiguous and match gy")
        if not gy.flags.c_contiguous:
            raise ValueError("[capture] gy must be C-contiguous")

        # x(또는 z) 입력 확보
        x = x_in if x_in is not None else (z_in if z_in is not None else self._z_saved)

        n = gy.size
        grid, block = _grid1d(n)
        actn = (self.act or "none").lower()

        if actn == "none":
            if out is not gy:
                out[...] = gy
            return

        # ReLU/Leaky/ELU는 x 필요 (raw kernel 사용)
        if actn in ("relu", "leaky_relu", "elu"):
            if x is None:
                # fall back to saved y
                y = self._y_saved
                if y is None:
                    raise RuntimeError(f"[capture] {self.act} backward needs x/z or saved y.")
                if y.shape != gy.shape or y.dtype != gy.dtype or not y.flags.c_contiguous:
                    raise RuntimeError("[capture] saved y must be C-contiguous and match gy")
                dtype = _dtype_code(y)
                if actn == "relu":
                    _relu_bwd_fromy_raw((grid,), (block,), (y.data.ptr, gy.data.ptr, out.data.ptr, n, dtype))
                elif actn == "leaky_relu":
                    _leaky_bwd_fromy_raw((grid,), (block,), (y.data.ptr, gy.data.ptr, out.data.ptr, self.neg_slope, n, dtype))
                else:  # "elu"
                    _elu_bwd_fromy_raw((grid,), (block,), (y.data.ptr, gy.data.ptr, out.data.ptr, self.alpha, n, dtype))
                return
            # 정상 경로(x 있음)
            dtype = _dtype_code(x)
            if actn == "relu":
                _relu_bwd_raw((grid,), (block,), (x.data.ptr, gy.data.ptr, out.data.ptr, n, dtype))
            elif actn == "leaky_relu":
                _leaky_bwd_raw((grid,), (block,), (x.data.ptr, gy.data.ptr, out.data.ptr, self.neg_slope, n, dtype))
            else:  # "elu"
                _elu_bwd_raw((grid,), (block,), (x.data.ptr, gy.data.ptr, out.data.ptr, self.alpha, n, dtype))
            return

        # 나머지는 수학 연쇄 — out을 작업공간으로 사용
        if actn == "sigmoid":
            # gx = gy * s * (1 - s), s = sigmoid(x)
            if x is None and self._y_saved is not None:
                # y가 저장되어 있으면 s = y/x 가 아니라 y/x는 아님. Activation(sigmoid)의 y == s.
                out[...] = self._y_saved
            else:
                if x is None:
                    raise RuntimeError("[capture] sigmoid backward needs x or saved y")
                out[...] = x
                cp.negative(out, out=out); cp.exp(out, out=out); cp.add(out, 1, out=out); cp.reciprocal(out, out=out)
            cp.multiply(out, (1 - out), out=out)
            cp.multiply(out, gy, out=out)

        elif actn == "tanh":
            # gx = gy * (1 - tanh(x)^2)
            if x is None and self._y_saved is not None:
                # y = tanh(x) 가 저장되어 있으면 바로 사용
                out[...] = self._y_saved
            else:
                if x is None:
                    raise RuntimeError("[capture] tanh backward needs x or saved y")
                out[...] = x
                cp.tanh(out, out=out)
            cp.multiply(out, out, out=out)  # th^2
            cp.multiply(out, -1, out=out); cp.add(out, 1, out=out)
            cp.multiply(out, gy, out=out)

        elif actn == "silu":
            # gx = gy * [ s*(1 + x*(1 - s)) ], s = sigmoid(x)
            if x is None:
                raise RuntimeError("[capture] silu backward needs x (no saved-y formula used)")
            out[...] = x
            cp.negative(out, out=out); cp.exp(out, out=out); cp.add(out, 1, out=out); cp.reciprocal(out, out=out)  # s
            cp.add((1 - out) * x, 1, out=out)                  # 1 + x*(1-s)
            cp.multiply(out, (1 / (1 + cp.exp(-x))), out=out)  # s*(1 + x*(1-s))
            cp.multiply(out, gy, out=out)

        elif actn == "gelu":
            # analytic grad of tanh approx: x 필요
            if x is None:
                raise RuntimeError("[capture] gelu backward needs x/z")
            out[...] = x.astype(x.dtype, copy=False)
            tmp = out
            cp.multiply(tmp, tmp, out=tmp)      # x^2
            cp.multiply(tmp, x, out=tmp)        # x^3
            cp.multiply(tmp, _GELU_A, out=tmp)  # A*x^3
            cp.add(tmp, x, out=tmp)             # x + A*x^3
            cp.multiply(tmp, _GELU_C, out=tmp)  # t
            cp.tanh(tmp, out=tmp)               # th
            th = tmp
            cp.multiply(tmp, tmp, out=tmp); cp.multiply(tmp, -1, out=tmp); cp.add(tmp, 1, out=tmp)  # 1 - th^2
            x2 = out
            x2[...] = x
            cp.multiply(x2, x2, out=x2)         # x^2
            cp.multiply(x2, 3.0 * _GELU_A, out=x2)
            cp.add(x2, 1.0, out=x2)
            cp.multiply(x2, _GELU_C, out=x2)    # dt
            cp.add(th, 1.0, out=th); cp.multiply(th, 0.5, out=th)
            cp.multiply(tmp, x2, out=x2)        # sech2*dt
            cp.multiply(x2, 0.5, out=x2); cp.multiply(x2, x, out=x2)
            cp.add(th, x2, out=out)
            cp.multiply(out, gy, out=out)

        elif actn == "softplus":
            # gx = gy * sigmoid(x)
            if x is None:
                raise RuntimeError("[capture] softplus backward needs x")
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
        if not isinstance(input_shape, tuple):
            input_shape = tuple(int(v) if v is not None else None for v in input_shape)
        else:
            input_shape = tuple(int(v) if v is not None else None for v in input_shape)
        return input_shape
