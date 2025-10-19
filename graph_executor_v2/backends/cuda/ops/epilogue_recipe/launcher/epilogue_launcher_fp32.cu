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
extern "C" __global__ void ep_f32_ffn_gelu_drop_resid(EpParamsF32);
extern "C" __global__ void ep_f32_gru3(EpParamsF32);


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

Status run_fp32_recipe(const Plan& plan, const Tensors& ts, DType bdt, void* stream){
  if (plan.recipe == LayerRecipe::FFN_GELU_Dropout_Residual){
    if (bdt!=DType::F32) return {false,"Bias dtype must be F32 for FP32 recipe"};
    if (const char* err = validate_ffn_gelu_dropout_resid(plan, ts)) return {false, err};

    auto P = pack_fp32(plan, ts);
    P.resid = static_cast<const float*>(ts.y); // 일반적 residual: y_old (in-place add)
    dim3 block(256), grid = make_grid(P.M*P.N, block.x);
    auto s = (cudaStream_t)stream;
    ep_f32_ffn_gelu_drop_resid<<<grid,block,0,s>>>(P);
    return {true,""};
  } else if (plan.recipe == LayerRecipe::CNN_Conv_Bias_ReLU) {
    if (bdt!=DType::F32) return {false,"Bias dtype must be F32 for FP32 CNN recipe"};
    if (const char* err = validate_cnn_bias_relu(plan, ts)) return {false, err};

    auto P = pack_fp32(plan, ts);
    // 레시피니까 강제 설정 (act=ReLU, has_bias=1)
    P.act = 1; P.has_bias = 1;
    dim3 block(256), grid = make_grid(P.M*P.N, block.x);
    auto s = (cudaStream_t)stream;
    ep_f32_relu_bias<<<grid,block,0,s>>>(P);
    return {true,""};
  }

  // ... (FFN, CNN 분기)
  if (plan.recipe == LayerRecipe::GRU3_Gates){
    if (bdt!=DType::F32) return {false,"Bias dtype must be F32 for FP32 GRU recipe"};
    if (const char* err = validate_gru3(plan, ts)) return {false, err};
    EpParamsF32 P = pack_fp32(plan, ts);
    // ld_x는 상위에서 3N으로 넘겨주는 걸 권장. 없으면 여기서 강제:
    if (P.ld_x == P.N) P.ld_x = 3*P.N;
    dim3 block(256), grid = make_grid(P.M*P.N, block.x);
    auto s = (cudaStream_t)stream;
    ep_f32_gru3<<<grid,block,0,s>>>(P);
    return {true,""};
  }

  return {false,"unknown recipe for FP32"};
}

} // namespace epi
