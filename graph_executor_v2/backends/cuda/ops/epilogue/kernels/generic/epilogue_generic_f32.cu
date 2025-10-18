#include <cuda_runtime.h>
#include "../epilogue_params.cuh"
#include "../functors/act_relu.cuh"
#include "../functors/dropout_philox.cuh"

extern "C" __global__
void epilogue_kernel_f32_generic(EpParamsF32 P){
  PhiloxState st{P.seed, P.offset};
  int t = blockIdx.x*blockDim.x + threadIdx.x, T=P.M*P.N;
  for(int i=t; i<T; i += gridDim.x*blockDim.x){
    int m=i/P.N, n=i%P.N;
    int ix=m*P.ld_x+n, iy=m*P.ld_y+n;
    float v = P.x[ix];
    if (P.has_bias) v += P.bias[n];
    if (P.act==1) v = relu_f(v);
    if (P.use_dropout) v = apply_dropout<float>(v, st, (unsigned long long)i, P.p_drop, P.keep_scale);
    float out = P.alpha*v + (P.beta!=0.f ? P.beta*P.y[iy] : 0.f);
    P.y[iy] = out;
  }
}
