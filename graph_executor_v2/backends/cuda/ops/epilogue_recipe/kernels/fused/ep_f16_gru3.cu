#include <cuda_runtime.h>
#include "kernels/epilogue_params.cuh"
#include "kernels/functors/act_sigmoid_tanh.cuh"

extern "C" __global__
void ep_f16_gru3(EpParamsF16 P){
  int M=P.M, N=P.N;
  int t = blockIdx.x*blockDim.x + threadIdx.x, T=M*N;
  const half* x = P.x; const half* b = P.bias; const half* hprev = P.resid;
  half* y = P.y;
  for (int i=t;i<T;i+=gridDim.x*blockDim.x){
    int m=i/N, n=i%N;
    int row_x = m*P.ld_x;
    int ix_z  = row_x + n;
    int ix_r  = row_x + N + n;
    int ix_n  = row_x + 2*N + n;

    half vz = x[ix_z]; if (b) vz = __hadd(vz, b[n]);
    half vr = x[ix_r]; if (b) vr = __hadd(vr, b[N + n]);
    half vn = x[ix_n]; if (b) vn = __hadd(vn, b[2*N + n]);

    half z = sigmoid_h(vz);
    half r = sigmoid_h(vr);
    half ntilde = tanh_h(vn);

    float hp = hprev ? __half2float(hprev[m*P.ld_y + n]) : 0.f;
    float h  = (1.f - __half2float(z)) * __half2float(ntilde) + __half2float(z) * hp;
    float out = P.alpha*h + (P.beta ? P.beta*__half2float(y[m*P.ld_y+n]) : 0.f);
    y[m*P.ld_y + n] = __float2half(out);
  }
}
