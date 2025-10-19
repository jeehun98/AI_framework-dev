#include <cuda_runtime.h>
#include "../api/epilogue.h"
#include "epilogue_pack_common.cuh"
#include "epilogue_dispatch_common.cuh"

// 선언
extern "C" __global__ void ep_f16_relu_bias(EpParamsF16);
extern "C" __global__ void ep_f16_relu_nobias(EpParamsF16);
extern "C" __global__ void epilogue_kernel_f16_generic(EpParamsF16);
extern "C" __global__ void ep_f16_gelu_bias(EpParamsF16);
extern "C" __global__ void ep_f16_gelu_nobias(EpParamsF16);
extern "C" __global__ void ep_f16_ffn_gelu_drop_resid(EpParamsF16);
extern "C" __global__ void ep_f16_relu_bias(EpParamsF16);
extern "C" __global__ void ep_f16_gru3(EpParamsF16);


namespace epi {

Status run_fp16(const Plan& plan, const Tensors& ts, DType bdt, void* stream){
  if (bdt!=DType::F16) return {false,"Bias dtype must be F16 for FP16 path (MVP)"};
  if (const char* err = validate_common(plan, ts)) return {false, err};

  auto P = pack_fp16(plan, ts);
  dim3 block(256), grid = make_grid(P.M*P.N, block.x);
  auto s = reinterpret_cast<cudaStream_t>(stream);

    const auto act = plan.attrs.act;
    const bool bias = (plan.attrs.bias==BiasKind::PerN);

    if (act==ActKind::ReLU) {
    if (bias) ep_f16_relu_bias<<<grid,block,0,s>>>(P);
    else      ep_f16_relu_nobias<<<grid,block,0,s>>>(P);
    } else if (act==ActKind::GELU) {
    if (bias) ep_f16_gelu_bias<<<grid,block,0,s>>>(P);
    else      ep_f16_gelu_nobias<<<grid,block,0,s>>>(P);
    } else {
    epilogue_kernel_f16_generic<<<grid,block,0,s>>>(P);
    }
  return {true,""};
}

Status run_fp16_recipe(const Plan& plan, const Tensors& ts, DType bdt, void* stream){
  if (plan.recipe == LayerRecipe::FFN_GELU_Dropout_Residual){
    if (bdt!=DType::F16) 
      return {false,"Bias dtype must be F16 for FP16 recipe"};
    if (const char* err = validate_ffn_gelu_dropout_resid(plan, ts)) 
      return {false, err};

    auto P = pack_fp16(plan, ts);
    P.resid = static_cast<const half*>(ts.y); // in-place residual add
    dim3 block(256), grid = make_grid(P.M*P.N, block.x);
    auto s = (cudaStream_t)stream;
    ep_f16_ffn_gelu_drop_resid<<<grid,block,0,s>>>(P);
    return {true,""};
  }
  return {false,"unknown recipe for FP16"};
}

Status run_fp16_recipe(const Plan& plan, const Tensors& ts, DType bdt, void* stream){
  if (plan.recipe == LayerRecipe::FFN_GELU_Dropout_Residual) {
    // (기존 FFN 처리)
  } else if (plan.recipe == LayerRecipe::CNN_Conv_Bias_ReLU) {
    if (bdt!=DType::F16) return {false,"Bias dtype must be F16 for FP16 CNN recipe"};
    if (const char* err = validate_cnn_bias_relu(plan, ts)) return {false, err};
    auto P = pack_fp16(plan, ts);
    P.act = 1; P.has_bias = 1;
    dim3 block(256), grid = make_grid(P.M*P.N, block.x);
    auto s = (cudaStream_t)stream;
    ep_f16_relu_bias<<<grid,block,0,s>>>(P);
    return {true,""};
  }
  // ...
  if (plan.recipe == LayerRecipe::GRU3_Gates){
    if (bdt!=DType::F16) return {false,"Bias dtype must be F16 for FP16 GRU recipe"};
    if (const char* err = validate_gru3(plan, ts)) return {false, err};
    EpParamsF16 P = pack_fp16(plan, ts);
    if (P.ld_x == P.N) P.ld_x = 3*P.N;
    dim3 block(256), grid = make_grid(P.M*P.N, block.x);
    auto s = (cudaStream_t)stream;
    ep_f16_gru3<<<grid,block,0,s>>>(P);
    return {true,""};
  }
  return {false,"unknown recipe for FP16"};
}
} // namespace epi
