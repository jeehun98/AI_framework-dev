// backends/cuda/ops/gemm/api.hpp
#pragma once
/**
 * @file api.hpp
 * @brief GEMM (CUDA) forward/backward API — Z(pre-activation) 저장 + capture-safe workspace 지원
 *
 * 규약 요약:
 *  - 모든 텐서는 row-major 2D.
 *  - Forward:
 *      Y = act( A @ B + bias )   // (현재 C는 경로상 미사용)
 *      attrs.save_z==true 이고 Z_saved!=nullptr 이면,
 *      Z_saved <- pre-activation Z (= A@B + bias) 을 한 패스에서 저장.
 *      Z_saved 가 Y와 같은 버퍼(alias)여도 허용되며 이때 ldZ는 내부적으로 ldd 사용.
 *  - Backward:
 *      gZ = gY ⊙ act'(Z)
 *      gA = gZ @ B^T,  gB = A^T @ gZ
 *      (옵션) gC = beta * gZ  (C,gC가 주어지고 beta!=0일 때)
 *      (옵션) gBias: Scalar/PerM/PerN 형태로 원자적 누적
 *
 * 워크스페이스:
 *  - lt_workspace: (선택) cublasLt용, 256B 정렬 권장
 *  - scratch: (선택) backward의 gZ 임시 버퍼. 크기 >= M*N*sizeof(float).
 */

#include "backends/cuda/ops/_common/shim/ai_shim.hpp"
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace ai {

// ---- Workspace ----
// Lt workspace는 256B 정렬 권장. 편의상 구조체 자체를 256 정렬로 둠.
struct alignas(256) GemmWorkspace {
  void*  lt_workspace       = nullptr;
  size_t lt_workspace_bytes = 0;

  // backward용 gZ scratch (row-major, ld = N 가정)
  void*  scratch            = nullptr;
  size_t scratch_bytes      = 0;
};

// ------------ 작은 헬퍼들(권장) ------------

// backward gZ scratch에 필요한 최소 바이트 수
inline constexpr size_t GemmRequiredBackwardScratchBytes(int64_t M, int64_t N) {
  return (M <= 0 || N <= 0) ? 0ull
                            : static_cast<size_t>(M) * static_cast<size_t>(N) * sizeof(float);
}

// Lt workspace 정렬 체크(256B 권장)
inline bool GemmIsLtWorkspaceAligned(const void* p, size_t alignment = 256) {
  if (!p) return true;
  return (reinterpret_cast<uintptr_t>(p) % alignment) == 0;
}

// Z_saved 규칙 요약(문서용): alias 허용/stride 규칙
// - Z_saved가 Y와 같은 버퍼면 ldZ는 내부적으로 ldd가 사용됨.


// ------------ Forward API ------------

// 정식 선언: 기본인자는 맨 끝 2개만 허용
ai::Status GemmCudaLaunch(
    const Tensor&    A,
    const Tensor&    B,
    const Tensor*    Bias,          // ← 기본값 주지 마세요!
    Tensor&          Y,
    const GemmAttrs& attrs,
    StreamHandle     stream,
    Tensor*          Z_saved = nullptr,
    const GemmWorkspace* ws = nullptr
);

// 편의 오버로드: Bias 생략 시 nullptr 사용
inline ai::Status GemmCudaLaunch(
    const Tensor&    A,
    const Tensor&    B,
    Tensor&          Y,
    const GemmAttrs& attrs,
    StreamHandle     stream,
    Tensor*          Z_saved = nullptr,
    const GemmWorkspace* ws = nullptr
) {
  return GemmCudaLaunch(A, B, /*Bias=*/nullptr, Y, attrs, stream, Z_saved, ws);
}


// ------------ Backward API ------------

ai::Status GemmCudaBackward(
    const Tensor&    A,
    const Tensor&    B,
    const Tensor*    C,              // 기본값 X
    const Tensor&    gY,
    const Tensor&    Z,
    Tensor*          gA,             // 기본값 X (포인터는 호출부에서 nullptr 전달)
    Tensor*          gB,
    Tensor*          gC,
    Tensor*          gBias,
    const GemmAttrs& attrs,
    StreamHandle     stream,
    const GemmWorkspace* ws = nullptr
);

// 하위호환 편의 래퍼(원하시면 유지)
inline ai::Status GemmCudaBackwardInto(
    const Tensor&    A,
    const Tensor&    B,
    const Tensor*    C,
    const Tensor&    gY,
    const Tensor&    Z,
    Tensor*          gA,
    Tensor*          gB,
    Tensor*          gC,
    Tensor*          gBias,
    const GemmAttrs& attrs,
    StreamHandle     stream,
    float*           dZ,
    void*            lt_ws       = nullptr,
    size_t           lt_ws_bytes = 0)
{
    GemmWorkspace ws{};
    ws.scratch            = static_cast<void*>(dZ);
    ws.scratch_bytes      = 0;                // 크기 체크는 내부에서 수행
    ws.lt_workspace       = lt_ws;
    ws.lt_workspace_bytes = lt_ws_bytes;

    return GemmCudaBackward(
        A, B, C, gY, Z, gA, gB, gC, gBias, attrs, stream, &ws
    );
}

} // namespace ai
