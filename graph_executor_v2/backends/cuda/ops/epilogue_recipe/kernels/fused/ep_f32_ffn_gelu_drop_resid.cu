#include <cuda_runtime.h>
#include "kernels/epilogue_params.cuh"
#include "kernels/functors/act_gelu.cuh"
#include "kernels/functors/dropout_philox.cuh"

extern "C" __global__
void ep_f32_ffn_gelu_drop_resid(EpParamsF32 P){
  PhiloxState st{P.seed, P.offset};
  int t = blockIdx.x*blockDim.x + threadIdx.x, T=P.M*P.N;
  for(int i=t;i<T;i+=gridDim.x*blockDim.x){
    int m=i/P.N, n=i%P.N;
    int ix=m*P.ld_x+n, iy=m*P.ld_y+n;
    float v = P.x[ix];
    if (P.has_bias) v += P.bias[n];
    v = gelu_f(v);
    if (P.use_dropout) v = apply_dropout<float>(v, st, (unsigned long long)i, P.p_drop, P.keep_scale);
    // y = alpha*v + beta*y + resid
    float out = P.alpha*v + (P.beta!=0.f ? P.beta*P.y[iy] : 0.f);
    if (P.resid) out += P.resid[iy];
    P.y[iy] = out;
  }
}
