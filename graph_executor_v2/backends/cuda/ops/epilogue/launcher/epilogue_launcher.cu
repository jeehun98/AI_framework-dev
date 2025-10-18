// launcher/epilogue_launcher.cu

#include <cuda_runtime.h>
#include "api/epilogue.h"
#include "kernels/epilogue_params.cuh"

extern "C" __global__ void epilogue_kernel_f32_rowmajor(EpParams);


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
  epilogue_kernel_f32_rowmajor<<<grid,block,0,s>>>(P);
  return {true,""};
}

} // namespace epi


/*

#include "api/epilogue.h"
#include "kernels/epilogue_params.cuh"

namespace epi {

static void compute_broadcast_stride(EpParams& P, const Attrs& a) {
  // RowMajor 가정 (ColMajor/Strided은 필요 시 확장)
  P.sx_m = P.ld_x; P.sx_n = 1;
  P.sy_m = P.ld_y; P.sy_n = 1;
  // bias
  if (a.bias == BiasKind::PerN) { P.sb_m = 0; P.sb_n = 1; }
  else if (a.bias == BiasKind::PerM){ P.sb_m = 1; P.sb_n = 0; }
  else if (a.bias == BiasKind::Scalar){ P.sb_m = 0; P.sb_n = 0; }
  // resid, z 동일하게 full [M,N] 가정 (향후 broadcast 필요 시 추가)
  P.sr_m = P.sy_m; P.sr_n = P.sy_n;
  P.sz_m = P.sy_m; P.sz_n = P.sy_n;
}

static uint32_t make_opmask(const Attrs& a){
  uint32_t m=0;
  if (a.bias   != BiasKind::None) m |= (1u<<0);
  if (a.save_z)                    m |= (1u<<1);
  if (a.act   != ActKind::None)    m |= (1u<<2);
  if (a.dropout)                   m |= (1u<<3);
  if (a.resid != ResidKind::None)  m |= (1u<<4);
  if (a.beta != 0.f)               m |= (1u<<5);
  if (a.clamp)                     m |= (1u<<6);
  if (a.quant != QuantKind::FP32)  m |= (1u<<7);
  return m;
}

Status run(const Plan& plan, const Tensors& ts,
           DType xdt, DType ydt, DType bdt, void* stream) {
  // 1) 기본 검증 (독립 모듈이므로 여기서 명확히)
  if (ts.M<=0 || ts.N<=0) return {false, "M,N must be positive"};
  if (!ts.x || !ts.y)     return {false, "x,y pointers required"};
  if (plan.attrs.save_z && !ts.z) return {false,"save_z is set but z==nullptr"};
  if (plan.attrs.dropout && ts.rng_seed==0ULL)
    return {false,"dropout requires non-zero rng_seed for determinism"};

  // 2) 파라미터 패킹
  EpParams P{};
  P.M = ts.M; P.N = ts.N;
  P.ld_x = ts.ld_x? ts.ld_x: ts.N;
  P.ld_y = ts.ld_y? ts.ld_y: ts.N;
  P.x = ts.x; P.y = ts.y;
  P.bias = ts.bias; P.resid = ts.resid; P.z = ts.z; P.mask = ts.mask;
  P.alpha = plan.attrs.alpha; P.beta = plan.attrs.beta;
  P.act_alpha = plan.attrs.act_alpha;
  P.p_drop = plan.attrs.p_drop;
  P.keep_scale = (plan.attrs.dropout && plan.attrs.p_drop<1.f) ?
                  1.f/(1.f-plan.attrs.p_drop) : 1.f;
  P.clamp_min = plan.attrs.clamp_min; P.clamp_max = plan.attrs.clamp_max;
  P.seed = ts.rng_seed; P.offset = ts.rng_offset;
  P.opmask = make_opmask(plan.attrs);
  P.act = (uint8_t)plan.attrs.act;
  P.resid_k = (uint8_t)plan.attrs.resid;
  P.quant = (uint8_t)plan.attrs.quant;
  P.bias_k = (uint8_t)plan.attrs.bias;
  P.x_type = (uint8_t)xdt; P.y_type = (uint8_t)ydt; P.b_type = (uint8_t)bdt;

  compute_broadcast_stride(P, plan.attrs);

  // 3) 디스패치 (MVP: generic만)
  dim3 block(256);
  dim3 grid( (P.M*P.N + block.x-1)/block.x );
  grid.x = min(grid.x, 65535u);
  cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
  epilogue_kernel_generic<<<grid,block,0,s>>>(P);
  return {true, ""};
}

} // namespace epi

*/