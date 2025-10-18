#include <cuda_runtime.h>
#include "../api/epilogue.h"
#include "epilogue_pack_common.cuh"
#include "epilogue_dispatch_common.cuh"

// 선언만 (정의는 kernels/ 아래)
extern "C" __global__ void ep_f32_relu_bias(EpParamsF32);
extern "C" __global__ void ep_f32_relu_nobias(EpParamsF32);
extern "C" __global__ void epilogue_kernel_f32_generic(EpParamsF32);
extern "C" __global__ void ep_f32_gelu_bias(EpParamsF32);
extern "C" __global__ void ep_f32_gelu_nobias(EpParamsF32);

namespace epi {

Status run_fp32(const Plan& plan, const Tensors& ts, DType bdt, void* stream){
    if (bdt!=DType::F32) return {false,"Bias dtype must be F32 for FP32 path (MVP)"};
    if (const char* err = validate_common(plan, ts)) return {false, err};

    auto P = pack_fp32(plan, ts);
    dim3 block(256), grid = make_grid(P.M*P.N, block.x);
    auto s = reinterpret_cast<cudaStream_t>(stream);

    const auto act = plan.attrs.act;
    const bool bias = (plan.attrs.bias==BiasKind::PerN);

    if (act==ActKind::ReLU) {
    if (bias) ep_f32_relu_bias<<<grid,block,0,s>>>(P);
    else      ep_f32_relu_nobias<<<grid,block,0,s>>>(P);
    } else if (act==ActKind::GELU) {
    if (bias) ep_f32_gelu_bias<<<grid,block,0,s>>>(P);
    else      ep_f32_gelu_nobias<<<grid,block,0,s>>>(P);
    } else {
    epilogue_kernel_f32_generic<<<grid,block,0,s>>>(P);
    }

  return {true,""};
}

} // namespace epi
