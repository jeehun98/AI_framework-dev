#include <cuda_runtime.h>
#include "../api/epilogue.h"
#include "epilogue_pack_common.cuh"
#include "epilogue_dispatch_common.cuh"

// 선언
extern "C" __global__ void ep_f16_relu_bias(EpParamsF16);
extern "C" __global__ void ep_f16_relu_nobias(EpParamsF16);
extern "C" __global__ void epilogue_kernel_f16_generic(EpParamsF16);

namespace epi {

Status run_fp16(const Plan& plan, const Tensors& ts, DType bdt, void* stream){
  if (bdt!=DType::F16) return {false,"Bias dtype must be F16 for FP16 path (MVP)"};
  if (const char* err = validate_common(plan, ts)) return {false, err};

  auto P = pack_fp16(plan, ts);
  dim3 block(256), grid = make_grid(P.M*P.N, block.x);
  auto s = reinterpret_cast<cudaStream_t>(stream);

  const bool relu = (plan.attrs.act==ActKind::ReLU);
  const bool bias = (plan.attrs.bias==BiasKind::PerN);

  if (relu) {
    if (bias) ep_f16_relu_bias<<<grid,block,0,s>>>(P);
    else      ep_f16_relu_nobias<<<grid,block,0,s>>>(P);
  } else {
    epilogue_kernel_f16_generic<<<grid,block,0,s>>>(P);
  }
  return {true,""};
}

} // namespace epi
