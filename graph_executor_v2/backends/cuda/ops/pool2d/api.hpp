#pragma once

// 통합 빌드(코어) vs 독립 빌드(shim) 동시 지원
#ifdef BUILD_STANDALONE_OPS
  #include "backends/cuda/ops/_common/shim/ai_shim.hpp"
#else
  #include "ai/tensor.hpp"
  #include "ai/dispatch.hpp"
#endif

namespace ai {

// ---------------------------- Attributes ----------------------------
struct Pool2DAttrs {
  int kH{2}, kW{2};
  int sH{2}, sW{2};
  int pH{0}, pW{0};
  int dH{1}, dW{1};
  bool ceil_mode{false};          // 출력 크기 계산 시 ceil 적용
  bool count_include_pad{false};  // AvgPool 전용 (권장: false)
  // 역전파 정책
  bool recompute_indices_in_bwd{false}; // MaxPool BWD에서 FWD 인덱스가 없을 때 X/Y로 재계산
};

// (선택) 편의: 출력 크기 계산 유틸
inline void compute_pool2d_outHW(int H_in, int W_in, const Pool2DAttrs& a,
                                 int& H_out, int& W_out) {
  auto div_up = [](int x, int y){ return (x + y - 1) / y; };
  int eff_kH = a.dH * (a.kH - 1) + 1;
  int eff_kW = a.dW * (a.kW - 1) + 1;
  auto baseH = H_in + 2*a.pH - eff_kH;
  auto baseW = W_in + 2*a.pW - eff_kW;
  if (a.ceil_mode) {
    H_out = (baseH >= 0) ? div_up(baseH + 1, a.sH) : 0;
    W_out = (baseW >= 0) ? div_up(baseW + 1, a.sW) : 0;
  } else {
    H_out = (baseH >= 0) ? (baseH + 1) / a.sH : 0;
    W_out = (baseW >= 0) ? (baseW + 1) / a.sW : 0;
  }
}

// ----------------------- Capture-safe Workspaces ----------------------
struct MaxPool2DWorkspaceFwd {
  // 선택: FWD 동안 인덱스를 버퍼에 기록(저장 안 하더라도 BWD 대비)
  // [N, C, H_out, W_out] 크기의 int32 인덱스 버퍼
  int32_t* indices{nullptr};
};

struct MaxPool2DWorkspaceBwd {
  // BWD에서 사용할 인덱스 버퍼 (없으면 attrs.recompute_indices_in_bwd=true로 재계산)
  const int32_t* indices{nullptr};
  // 선택: 원자적 scatter 경로 등에서 사용할 scratch
  // 크기/형태는 구현에 따라 다르나, [N*C*H_out*W_out] 또는 [N*C*H_in*W_in] float 등
  float* scratch{nullptr};
};

struct AvgPool2DWorkspaceFwd {
  // 보통 필요 없음. (공유메모리 기반 합산이면 외부 버퍼 불요)
  // 대형 창/특수 최적화에서 필요 시 확장 포인트로 제공
  float* scratch{nullptr};
};

struct AvgPool2DWorkspaceBwd {
  // 분산 scatter를 원자적으로 수행 시 초기화/누적용 임시 버퍼
  // 구현에서 cudaMemsetAsync만으로 충분하면 nullptr 가능
  float* scratch{nullptr};
};

// ------------------------------ MaxPool ------------------------------
// Forward
//  - X: [N, C, H_in, W_in], Y: [N, C, H_out, W_out]
//  - Indices: (선택) 외부 저장용. nullptr면 버리지 않고 ws_fwd->indices에만 기록 가능.
//  - ws_fwd: (선택) 캡처 시 내부 임시 인덱스 저장에 사용.
Status MaxPool2DCudaLaunch(const Tensor& X,
                           Tensor& Y,
                           Tensor* Indices,                // nullable
                           const Pool2DAttrs& attrs,
                           StreamHandle stream,
                           const MaxPool2DWorkspaceFwd* ws_fwd = nullptr);

// Backward
//  - dY: [N, C, H_out, W_out], dX: [N, C, H_in, W_in]
//  - Indices: (선택) FWD에서 저장한 인덱스. 없으면 attrs.recompute_indices_in_bwd가 true여야 함.
//  - ws_bwd: (선택) 인덱스/스크래치 외부 주입.
Status MaxPool2DBackwardCudaLaunch(const Tensor& dY,
                                   Tensor& dX,
                                   const Tensor* Indices,            // nullable
                                   const Pool2DAttrs& attrs,
                                   StreamHandle stream,
                                   const MaxPool2DWorkspaceBwd* ws_bwd = nullptr);

// ------------------------------ AvgPool ------------------------------
// Forward
Status AvgPool2DCudaLaunch(const Tensor& X,
                           Tensor& Y,
                           const Pool2DAttrs& attrs,
                           StreamHandle stream,
                           const AvgPool2DWorkspaceFwd* ws_fwd = nullptr);

// Backward
Status AvgPool2DBackwardCudaLaunch(const Tensor& dY,
                                   Tensor& dX,
                                   const Pool2DAttrs& attrs,
                                   StreamHandle stream,
                                   const AvgPool2DWorkspaceBwd* ws_bwd = nullptr);

} // namespace ai
