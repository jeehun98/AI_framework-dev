#include <cuda_runtime.h>
#include "kernels/philox.cuh"
#include "kernels/epilogue_params.cuh"
#include "kernels/policy/ep_apply.cuh"
#include "kernels/policy/ep_policy.cuh"

using namespace epi;

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

// NOTE: 이 파일은 템플릿 정의 제공용입니다. 실제 커널 런치는 런처에서
// ep_kernel_policy<Policy, EpParamsF32><<<grid,block,0,stream>>>(p);
// 형태로 직접 호출합니다. (여기서 명시적 인스턴스화는 생략)
