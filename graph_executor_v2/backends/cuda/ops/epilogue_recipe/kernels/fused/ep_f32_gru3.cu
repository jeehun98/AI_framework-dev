#include <cuda_runtime.h>
#include "kernels/epilogue_params.cuh"
#include "kernels/functors/act_sigmoid_tanh.cuh"

// x: [M, 3N] with gates Z|R|N, bias: [3N], resid: h_prev [M,N], y: h_t
extern "C" __global__
void ep_f32_gru3(EpParamsF32 P){
  int M=P.M, N=P.N;
  int t = blockIdx.x*blockDim.x + threadIdx.x, T=M*N;
  const float* x = P.x; const float* b = P.bias; const float* hprev = P.resid;
  float* y = P.y;
  for (int i=t;i<T;i+=gridDim.x*blockDim.x){
    int m=i/N, n=i%N;
    int row_x = m*P.ld_x;         // ld_x is 3N (caller must set)
    int ix_z  = row_x + n;
    int ix_r  = row_x + N + n;
    int ix_n  = row_x + 2*N + n;
    // bias offsets
    float vz = x[ix_z] + (b ? b[n]       : 0.f);
    float vr = x[ix_r] + (b ? b[N + n]   : 0.f);
    float vn = x[ix_n] + (b ? b[2*N + n] : 0.f);

    float z = sigmoid_f(vz);
    float r = sigmoid_f(vr);
    // NOTE: 여기서는 vn이 이미 (Wx*x + (r ⊙ Uh*hprev) + b_n) 형태라고 가정
    // 보통 r 적용은 hidden matmul 경로에서 반영됨. 필요하면 아래처럼 추가:
    // vn += r * hprev[m*P.ld_y + n] * w_hn (요건 상위에서 합성하는 걸 권장)
    float n_tilde = tanh_f(vn);

    float hp = hprev ? hprev[m*P.ld_y + n] : 0.f; // ld_y==N 가정
    float h  = (1.f - z) * n_tilde + z * hp;
    y[m*P.ld_y + n] = P.alpha*h + (P.beta ? P.beta*y[m*P.ld_y+n] : 0.f);
  }
}
