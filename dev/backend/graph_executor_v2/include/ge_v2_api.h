#pragma once
/**
 * @file ge_v2_api.h
 * @brief graph_executor_v2 네이티브 모듈의 "런타임 컨트랙트(ABI)"를 고정하는 헤더
 *
 * 중요 규약:
 * 1) launch_kernel(kernel_name, buffers, descs, stream): buffers는 입력→출력 순서의 device ptr(uintptr_t) 리스트
 * 2) query_capability(op_type, ...): "<OPTYPE>__<KERNEL_NAME>" → score(int) 테이블 제공
 * 3) 성공=0, 실패는 음수(예: -1 invalid args, -2 device error…)
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

using ge2_uintptr = std::uintptr_t;

/** opaque stream 포인터(CUDA면 cudaStream_t) */
using ge2_stream_t = void*;

/** 커널 런처 C ABI */
using ge2_kernel_fn = int(*)(const ge2_uintptr* buffers, int num_buffers,
                             ge2_stream_t stream);

/** (옵션) 디스크립터 스케치 */
enum class ge2_dtype : uint8_t { f16=1, f32=2, i32=3 /* ... */ };

struct ge2_buf_desc {
  ge2_dtype dtype;
  uint8_t   rank;
  int64_t   shape[8];
  int64_t   stride[8];
  // layout/device 등은 필요 시 확장
};

extern "C" {
/** 커널 이름 → 함수 포인터 */
GE2_API const std::unordered_map<std::string, ge2_kernel_fn>&
ge_v2_kernel_table_raw();

/** "<OPTYPE>__<KERNEL_NAME>" → score(int) */
GE2_API const std::unordered_map<std::string, int>&
ge_v2_capability_table_raw();
}

#define GE2_API_VERSION 1
