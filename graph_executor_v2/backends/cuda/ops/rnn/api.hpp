#pragma once
#ifdef BUILD_STANDALONE_OPS
  #include "backends/cuda/ops/_common/shim/ai_shim.hpp"
#else
  #include "ai/tensor.hpp"
  #include "ai/op_schema.hpp"
  #include "ai/dispatch.hpp"
#endif

namespace ai {

struct RNNAttrs {
  int T{0}, B{0}, I{0}, H{0};
  bool save_z{true};     // true: Z를 [TB,H]로 저장, false: 즉시 tanh(Z)로 H만 저장(필요 시 ws.TMP_Z로 임시)
  // 선택: bidirectional, num_layers 등 확장 여지 (미사용이면 1/false 가정)
};

// ===== 캡처-세이프 워크스페이스 =====
// 모든 포인터는 "이미 device 메모리로 할당되어 있어야" 하며 런처는 절대 cudaMalloc/cudaFree 하지 않는다.
struct RNNWorkspaceFwd {
  // [필수] 크기: TB=T*B, BH=B*H
  float* PreZ_all{nullptr}; // [TB, H] — save_z=true일 때 결과 보관; false면 scratch로 사용 가능
  float* TMP_H   {nullptr}; // [B, H]   — 타임스텝 간 임시
  float* TMP_Z   {nullptr}; // [B, H]   — save_z=false일 때 tanh 이전 값 임시
};

struct RNNWorkspaceBwd {
  // [필수]
  float* dHsum     {nullptr}; // [B, H]
  float* dh_next   {nullptr}; // [B, H]
  float* dZ_all    {nullptr}; // [TB, H] — save_z=true면 Z에서 바로 미분, false면 fwd에서 저장된 H로부터 dZ 재구성해 채워야 함
  float* Hprev_all {nullptr}; // [TB, H] — (선택) 필요 시 제공. 없으면 bwd에서 시퀀스 재스캔이 필요.
};

// ========== [새로 추가] 워크스페이스 크기 질의(메모리 계획/그래프 캡처용) ==========
struct RNNWorkspaceSizes {
  size_t bytes_PreZ_all{0};
  size_t bytes_TMP_H{0};
  size_t bytes_TMP_Z{0};
  size_t bytes_dHsum{0};
  size_t bytes_dh_next{0};
  size_t bytes_dZ_all{0};
  size_t bytes_Hprev_all{0};
};

inline RNNWorkspaceSizes RNNGetWorkspaceSizes(const RNNAttrs& a) {
  const long long TB = 1LL * a.T * a.B;
  const long long BH = 1LL * a.B * a.H;
  RNNWorkspaceSizes s{};
  s.bytes_PreZ_all = sizeof(float) * TB * a.H;               // fwd
  s.bytes_TMP_H    = sizeof(float) * BH;
  s.bytes_TMP_Z    = sizeof(float) * BH;

  s.bytes_dHsum     = sizeof(float) * BH;                     // bwd
  s.bytes_dh_next   = sizeof(float) * BH;
  s.bytes_dZ_all    = sizeof(float) * TB * a.H;
  s.bytes_Hprev_all = sizeof(float) * TB * a.H;
  return s;
}

// ========== [새로 추가] 런처의 고정 그리드/분기 계약 안내를 위해 “준비” 함수(선택) ==========
struct RNNLaunchPlan {
  // 런타임 분기 없는 그리드/블록 등 커널 메타 (필요 시 확장)
  dim3 grid_preZ, block_preZ;
  dim3 grid_tanh, block_tanh;
  // … Wx/Wh GEMM 런처가 별도면 거기 메타도 포함 가능
};
Status RNNMakeLaunchPlan(const RNNAttrs& attrs, RNNLaunchPlan* plan);

// ===== 연산 유틸 (그대로 유지) =====
Status fill_zero(Tensor& t, StreamHandle s);
Status add_bias_rowwise(Tensor& Y, const Tensor& b, int B, int H, StreamHandle s);
Status add_out(const Tensor& A, const Tensor& B, Tensor& C, StreamHandle s);
Status add_inplace(Tensor& A, const Tensor& B, StreamHandle s);
Status tanh_out(const Tensor& X, Tensor& Y, StreamHandle s);
Status tanh_bwd_from_out(const Tensor& Y, const Tensor& dY, Tensor& dZ, StreamHandle s);
Status rowwise_sum_accum(const Tensor& M, Tensor& out, int B, int H, StreamHandle s);
Status transpose_2d(const Tensor& A, Tensor& AT, int M, int N, StreamHandle s);

// ===== Forward =====
// 계약:
//  - 내부에서 어떤 cudaMalloc/cudaFree도 하지 않음
//  - 디폴트 스트림 사용 금지; 반드시 인자로 받은 s만 사용
//  - save_z=true면 Zbuf가 nullptr이면 오류 반환(ai::Status::InvalidArgument)
//  - ws_fwd가 nullptr이면 "임시 없이도 가능한 경로"만 사용 (추천: 캡처 시 반드시 제공)
Status RNNCudaLaunch(const Tensor& X, const Tensor& h0,
                     const Tensor& Wx, const Tensor& Wh,
                     const Tensor* b, Tensor& Hout, Tensor* Zbuf,
                     const RNNAttrs& attrs, StreamHandle s,
                     const RNNWorkspaceFwd* ws_fwd /*=nullptr*/);

// ===== Backward =====
// 계약:
//  - dX/dh0/dWx/dWh/dB는 nullptr일 수 있음(해당 그라디언트 스킵)
//  - save_z=true면 Zbuf가 nullptr이면 오류 (Z로부터 dZ 계산)
//  - save_z=false면 ws_bwd->dZ_all이 필수(혹은 내부에서 H로부터 재구성) — 캡처용으론 사전할당 권장
Status RNNCudaBackwardLaunch(const Tensor& X, const Tensor& Hout, const Tensor* Zbuf,
                             const Tensor& h0, const Tensor& Wx, const Tensor& Wh,
                             const Tensor& dHout,
                             Tensor* dX, Tensor* dh0, Tensor* dWx, Tensor* dWh, Tensor* dB,
                             const RNNAttrs& attrs, StreamHandle s,
                             const RNNWorkspaceBwd* ws_bwd /*=nullptr*/);

} // namespace ai
