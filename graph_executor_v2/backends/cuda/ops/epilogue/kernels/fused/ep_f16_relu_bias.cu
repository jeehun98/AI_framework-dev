#include <cuda_runtime.h>
#include "../epilogue_params.cuh"
#include "../functors/act_relu.cuh"
#include "../functors/dropout_philox.cuh"

extern "C" __global__
void ep_f16_relu_bias(EpParamsF16 P){
  PhiloxState st{P.seed, P.offset};
  int t = blockIdx.x*blockDim.x + threadIdx.x, T=P.M*P.N;
  for(int i=t; i<T; i += gridDim.x*blockDim.x){
    int m=i/P.N, n=i%P.N;
    int ix=m*P.ld_x+n, iy=m*P.ld_y+n;
    half v = relu_h(__hadd(P.x[ix], P.bias[n]));
    if (P.use_dropout) v = apply_dropout<half>(v, st, (unsigned long long)i, P.p_drop, P.keep_scale);
    float out = P.alpha*__half2float(v) + (P.beta!=0.f ? P.beta*__half2float(P.y[iy]) : 0.f);
    P.y[iy] = __float2half(out);
  }
}
