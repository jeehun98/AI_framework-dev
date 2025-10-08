#pragma once
#include <vector>

// 통합 빌드(코어) vs 독립 빌드(shim) 동시 지원
#ifdef BUILD_STANDALONE_OPS
  #include "backends/cuda/ops/_common/shim/ai_shim.hpp"
#else
  #include "ai/tensor.hpp"
  #include "ai/dispatch.hpp"
#endif

namespace ai {

// --------------------- Spec ---------------------
// 현재는 Constant pad만 지원. (추후 reflect/replicate 확장 여지)
struct PadSpec {
  std::vector<int> before;   // 각 차원 앞쪽 pad 크기
  std::vector<int> after;    // 각 차원 뒤쪽 pad 크기
  float value{0.0f};         // 채울 상수값 (constant padding)
};

// ------------------ Workspace -------------------
// 캡처-세이프 훅(현재 비어있음). 추후 큰 텐서 채우기 최적화 시 scratch 등 확장 가능.
struct PadWorkspaceFwd {
  // 예: 대형 비제로 pad 채움에서 벡터화 fill을 위한 임시 버퍼 등
  // float* scratch{nullptr};
};
struct PadWorkspaceBwd {
  // 현재 필요 없음
  // float* scratch{nullptr};
};

// -------- 유틸: 출력 형태 계산(호출자 편의) --------
inline std::vector<int64_t> ComputePaddedShape(const std::vector<int64_t>& in_shape,
                                               const PadSpec& spec) {
  const size_t R = in_shape.size();
  std::vector<int64_t> out(R, 0);
  for (size_t i = 0; i < R; ++i) {
    const int b = (i < spec.before.size() ? spec.before[i] : 0);
    const int a = (i < spec.after.size()  ? spec.after[i]  : 0);
    out[i] = static_cast<int64_t>(in_shape[i]) + static_cast<int64_t>(b + a);
  }
  return out;
}

// ------------------ Forward/Backward ------------------
// Forward: Y = pad(X, spec)
//  - X, Y는 모두 호출자가 미리 할당/형태 세팅
//  - 내부 동적할당 없음 (캡처-세이프)
Status PadCudaLaunch(const Tensor& X, Tensor& Y,
                     const PadSpec& spec,
                     StreamHandle stream,
                     const PadWorkspaceFwd* ws_fwd /*=nullptr*/ = nullptr);

// Backward: dX = slice(dY, spec)  (pad로 채운 영역은 버림)
//  - dY, dX는 모두 호출자가 미리 준비
//  - 내부 동적할당 없음 (캡처-세이프)
Status PadBackwardCudaLaunch(const Tensor& dY, Tensor& dX,
                             const PadSpec& spec,
                             StreamHandle stream,
                             const PadWorkspaceBwd* ws_bwd /*=nullptr*/ = nullptr);

} // namespace ai
