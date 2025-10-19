#include <cuda_runtime.h>
#include "philox.cuh"
#include "epilogue_params.cuh"
#include "policy/ep_apply.cuh"
#include "policy/ep_policy.cuh"

using namespace epi;

// ===== 정책 인스턴스 선언 =====
// 필요한 조합만 인스턴스화 (예시)
using P_F32_ReLU_Bias_Drop    = EpPolicy<float, true,  1, true,  false>;
using P_F32_ReLU_Bias_NoDrop  = EpPolicy<float, true,  1, false, false>;
using P_F16_GELU_NoBias_Drop  = EpPolicy<half,  false, 2, true,  false>;

// 공통 커널 템플릿
template<typename Policy, typename P>
__global__ void ep_kernel_policy(P params){
  PhiloxState st{params.seed, params.offset};
  int M=params.M, N=params.N;
  int t = blockIdx.x*blockDim.x + threadIdx.x;
  int T = M*N;
  for(int i=t;i<T;i+=gridDim.x*blockDim.x){
    int m=i/N, n=i%N;
    int ix=m*params.ld_x+n, iy=m*params.ld_y+n;
    EpApply<Policy>::run(params, m,n, ix,iy, st, (unsigned long long)i);
  }
}

// ===== 엔트리 심볼 (런처에서 호출) =====
extern "C" {

__global__ void ep_f32_relu_bias_drop(EpParamsF32 p){
  ep_kernel_policy<P_F32_ReLU_Bias_Drop, EpParamsF32><<<gridDim,blockDim,0,0>>>(p); // dummy; real grid is set by launcher
}
__global__ void ep_f32_relu_bias_nodrop(EpParamsF32 p){
  ep_kernel_policy<P_F32_ReLU_Bias_NoDrop, EpParamsF32><<<gridDim,blockDim,0,0>>>(p);
}
__global__ void ep_f16_gelu_nobias_drop(EpParamsF16 p){
  ep_kernel_policy<P_F16_GELU_NoBias_Drop, EpParamsF16><<<gridDim,blockDim,0,0>>>(p);
}

} // extern "C"
