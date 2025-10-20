#include <cuda_runtime.h>
#include "api/epilogue.h"
#include "kernels/epilogue_params.cuh"
#include "kernels/philox.cuh"
#include "kernels/policy/ep_apply.cuh"
#include "kernels/policy/ep_policy.cuh"
#include "kernels/policy/ep_kernel_policy.cuh"  // ⬅ 추가


namespace epi {

static inline dim3 make_grid(int total, int block=256){
  int g=(total+block-1)/block; return dim3(g>0?g:1);
}

static inline EpParamsF32 pack_f32(const Plan& plan, const Tensors& ts){
  EpParamsF32 p{};
  p.M=ts.M; p.N=ts.N;
  p.ld_x=ts.ld_x?ts.ld_x:ts.N; p.ld_y=ts.ld_y?ts.ld_y:ts.N;
  p.x=(const float*)ts.x; p.y=(float*)ts.y; p.bias=(const float*)ts.bias;
  p.resid=(const float*)ts.resid;
  p.alpha=plan.attrs.alpha; p.beta=plan.attrs.beta;
  p.use_dropout=plan.attrs.dropout?1:0;
  p.p_drop=plan.attrs.p_drop;
  p.keep_scale = (plan.attrs.dropout && plan.attrs.p_drop<1.f)? 1.f/(1.f-plan.attrs.p_drop): 1.f;
  p.seed=ts.rng_seed; p.offset=ts.rng_offset;
  return p;
}
static inline EpParamsF16 pack_f16(const Plan& plan, const Tensors& ts){
  EpParamsF16 p{};
  p.M=ts.M; p.N=ts.N;
  p.ld_x=ts.ld_x?ts.ld_x:ts.N; p.ld_y=ts.ld_y?ts.ld_y:ts.N;
  p.x=(const half*)ts.x; p.y=(half*)ts.y; p.bias=(const half*)ts.bias;
  p.resid=(const half*)ts.resid;
  p.alpha=plan.attrs.alpha; p.beta=plan.attrs.beta;
  p.use_dropout=plan.attrs.dropout?1:0;
  p.p_drop=plan.attrs.p_drop;
  p.keep_scale = (plan.attrs.dropout && plan.attrs.p_drop<1.f)? 1.f/(1.f-plan.attrs.p_drop): 1.f;
  p.seed=ts.rng_seed; p.offset=ts.rng_offset;
  return p;
}

Status run(const Plan& plan, const Tensors& ts,
           DType xdt, DType ydt, DType bdt, void* stream){
  if (ts.M<=0 || ts.N<=0) return {false,"M,N must be positive"};
  if (!ts.x || !ts.y)     return {false,"x,y required"};

  auto s = reinterpret_cast<cudaStream_t>(stream);
  dim3 block(256), grid = make_grid(ts.M*ts.N, block.x);

  // === 예시: FP32 ReLU + PerN-Bias (+/- Dropout) ===
  if (xdt==DType::F32 && ydt==DType::F32 &&
      plan.attrs.act==ActKind::ReLU && plan.attrs.bias==BiasKind::PerN)
  {
    auto p = pack_f32(plan, ts);
    if (plan.attrs.dropout) {
      using Policy = EpPolicy<float, true, 1, true,  false>;
      ep_kernel_policy<Policy, EpParamsF32><<<grid,block,0,s>>>(p);
    } else {
      using Policy = EpPolicy<float, true, 1, false, false>;
      ep_kernel_policy<Policy, EpParamsF32><<<grid,block,0,s>>>(p);
    }
    return {true,""};
  }

  // === 예시: FP16 GELU + no-bias + Dropout ===
  if (xdt==DType::F16 && ydt==DType::F16 &&
      plan.attrs.act==ActKind::GELU && plan.attrs.bias==BiasKind::None &&
      plan.attrs.dropout)
  {
    auto p = pack_f16(plan, ts);
    using Policy = EpPolicy<half, false, 2, true,  false>;
    ep_kernel_policy<Policy, EpParamsF16><<<grid,block,0,s>>>(p);
    return {true,""};
  }

  return {false,"no policy kernel instantiated for given attrs"};
}

} // namespace epi
