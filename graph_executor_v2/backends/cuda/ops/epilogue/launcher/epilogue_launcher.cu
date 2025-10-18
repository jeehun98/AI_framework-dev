// launcher/epilogue_launcher.cu
#include <cuda_runtime.h>
#include "api/epilogue.h"
#include "kernels/epilogue_params.cuh"

// ✨ 선언만 (정의는 kernels/epilogue_kernel.cu 한 곳)
extern "C" __global__ void epilogue_kernel_f32_rowmajor(EpParams);
extern "C" __global__ void epilogue_kernel_f32_rowmajor_relu_bias(EpParams);
extern "C" __global__ void epilogue_kernel_f32_rowmajor_relu_nobias(EpParams);

namespace epi {

Status run(const Plan& plan, const Tensors& ts,
           DType xdt, DType ydt, DType bdt, void* stream){
  // 1) 검증
  if (xdt!=DType::F32 || ydt!=DType::F32 || bdt!=DType::F32)
    return {false, "MVP: FP32 only"};
  if (ts.M<=0 || ts.N<=0) return {false,"M,N must be positive"};
  if (!ts.x || !ts.y)     return {false,"x,y required"};
  if (plan.attrs.bias==BiasKind::PerN && !ts.bias) return {false,"bias==nullptr"};

  // 2) 파라미터 패킹
  EpParams P{};
  P.M = ts.M; P.N = ts.N;
  P.ld_x = ts.ld_x ? ts.ld_x : ts.N;
  P.ld_y = ts.ld_y ? ts.ld_y : ts.N;
  P.x = static_cast<const float*>(ts.x);
  P.y = static_cast<float*>(ts.y);
  P.bias = static_cast<const float*>(ts.bias);
  P.alpha = plan.attrs.alpha;
  P.beta  = plan.attrs.beta;
  P.act = (plan.attrs.act==ActKind::ReLU) ? 1 : 0;
  P.has_bias = (plan.attrs.bias==BiasKind::PerN) ? 1 : 0;

  // 3) 디스패치
  dim3 block(256);
  int total = P.M * P.N;
  dim3 grid((total + block.x - 1)/block.x);
  grid.x = grid.x ? grid.x : 1;
  auto s = reinterpret_cast<cudaStream_t>(stream);

  if (P.act == 1) { // ReLU
    if (P.has_bias)
      epilogue_kernel_f32_rowmajor_relu_bias<<<grid,block,0,s>>>(P);
    else
      epilogue_kernel_f32_rowmajor_relu_nobias<<<grid,block,0,s>>>(P);
  } else {
    // 폴백: Generic
    epilogue_kernel_f32_rowmajor<<<grid,block,0,s>>>(P);
  }
  return {true,""};
}

} // namespace epi
