#pragma once
/**
 * @file ge_v2_api.h
 * @brief graph_executor_v2 네이티브 모듈의 런타임 컨트랙트(ABI)
 *
 * 규약 요약
 * ----------
 * launch_kernel(kernel_name, buffers, descs, stream)
 *  - buffers: 입력들 → 출력들 → (옵션) Host 파라미터블록(ptr)
 *    * 각 입력/출력은 device pointer(uintptr_t)
 *    * Host 파라미터블록은 C/C++ 구조체의 주소(Host 메모리)
 *  - descs: (옵션) 디버그/로깅용 메타(JSON 직렬화 가능한 dict)
 *  - stream: CUDA 등의 네이티브 스트림(opaque)
 *
 * 반환 규칙
 *  - 0: 성공
 *  - 음수: 오류
 *     - -1: invalid args
 *     - -2: device/cublas error
 *     - -3: not implemented
 *
 * 커널 등록
 *  - ge_v2_kernel_table_raw(): { "kernel_name" -> 함수포인터 } 매핑
 *  - ge_v2_capability_table_raw(): { "OPTYPE__KERNEL" -> score } 힌트(선택)
 */

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef _WIN32
  #define GE2_API __declspec(dllexport)
#else
  #define GE2_API
#endif

// device pointer를 담는 안전한 정수형
using ge2_uintptr  = std::uintptr_t;

// 백엔드 스트림을 opaque 포인터로 전달 (CUDA면 cudaStream_t와 동일 크기)
using ge2_stream_t = void*;

// 공통 커널 시그니처
using ge2_kernel_fn = int(*)(const ge2_uintptr* buffers, int num_buffers,
                             ge2_stream_t stream);

// (옵션) 버퍼 설명자 예시 — 실제 ABI에 강제되지 않음
enum class ge2_dtype : uint8_t { f16=1, f32=2, i32=3 /* ... */ };

struct ge2_buf_desc {
  ge2_dtype dtype;
  uint8_t   rank;
  int64_t   shape[8];
  int64_t   stride[8];
};

extern "C" {

// 커널 테이블(필수)
GE2_API const std::unordered_map<std::string, ge2_kernel_fn>&
ge_v2_kernel_table_raw();

// 백엔드 capability table(선택)
GE2_API const std::unordered_map<std::string, int>&
ge_v2_capability_table_raw();

} // extern "C"

#define GE2_API_VERSION 1
