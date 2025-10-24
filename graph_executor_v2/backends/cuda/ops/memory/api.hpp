#pragma once

#ifdef BUILD_STANDALONE_OPS
  #include "backends/cuda/ops/_common/shim/ai_shim.hpp"
#else
  #include "ai/tensor.hpp"
  #include "ai/dispatch.hpp"
#endif

#include <cstdint>
#include <cstddef>

namespace ai {

// ---- 통계 구조 (Arena에서 읽어오기) ----
struct MemoryStats {
  uint64_t total_reserved{0};  // cudaMalloc 총합
  uint64_t peak_in_use{0};     // 동시에 사용된 최대 바이트
  uint64_t curr_in_use{0};     // 현재 사용 중
  int32_t  slabs{0};           // 슬랩 개수
};

// ---- Fill Ops (캡처-세이프) ----
Status FillF32Cuda(Tensor& dst, float value, StreamHandle stream);
Status FillI32Cuda(Tensor& dst, int32_t value, StreamHandle stream);

// ---- Capture-Safe Allocator 어댑터 ----
//  - Arena 본체는 executor에 구현되어 있어야 함
//  - 캡처 중 추가 cudaMalloc 금지 (Arena에서 보장)
Status MemoryReserveBytesCuda(uint64_t bytes);         // 캡처 전 사전예약
Status MemoryResetPoolCuda();                          // bump/LIFO 리셋(슬랩 유지)
Status MemoryStatsCuda(MemoryStats& out);              // 통계 조회

// 임시 워크스페이스 토큰 기반 API (연산자 내부에서 사용)
//  - token은 (ptr,size) 인코딩/혹은 핸들: Arena 구현에 따름
Status MemoryAllocTempCuda(uint64_t nbytes,
                           uint32_t align,
                           uint64_t& out_token,
                           StreamHandle stream);

Status MemoryFreeTempCuda(uint64_t token,
                          StreamHandle stream);

} // namespace ai
