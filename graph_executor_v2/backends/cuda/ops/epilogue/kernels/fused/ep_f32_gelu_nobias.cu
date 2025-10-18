#include <cuda_runtime.h>
#include "../epilogue_params.cuh"
#include "../functors/act_gelu.cuh"
#include "../functors/dropout_philox.cuh"
extern "C" __global__
void ep_f32_gelu_nobias(EpParamsF32 P){
  PhiloxState st{P.seed, P.offset};
  int t = blockIdx.x*blockDim.x + threadIdx.x, T=P.M*P.N;
  for(int i=t;i<T;i+=gridDim.x*blockDim.x){
    int m=i/P.N, n=i%P.N;
    int ix=m*P.ld_x+n, iy=m*P.ld_y+n;
    float v = gelu_f(P.x[ix]);
    if (P.use_dropout) v = apply_dropout<float>(v, st, (unsigned long long)i, P.p_drop, P.keep_scale);
    float out = P.alpha*v + (P.beta!=0.f ? P.beta*P.y[iy] : 0.f);
    P.y[iy] = out;
  }
}
