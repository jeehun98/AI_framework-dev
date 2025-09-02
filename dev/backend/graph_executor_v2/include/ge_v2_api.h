#pragma once
/**
 * @file ge_v2_api.h
 * @brief graph_executor_v2 네이티브 모듈의 런타임 컨트랙트(ABI)
 *
 * 규약 요약:
 *  - launch_kernel(kernel_name, buffers, descs, stream)
 *    * buffers: 입력들 → 출력들 → (옵션) Host 파라미터블록(ptr)
 *    * 입력/출력은 device pointer(uintptr_t)
 *  - query_capability(op_type, ...) : "<OPTYPE>__<KERNEL>" -> score(int)
 *  - 반환: 0=성공, 음수=오류(-1 invalid args, -2 device error, -3 not implemented ...)
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

using ge2_uintptr  = std::uintptr_t;  // pointer as integer
using ge2_stream_t = void*;           // opaque stream (CUDA면 cudaStream_t)

using ge2_kernel_fn = int(*)(const ge2_uintptr* buffers, int num_buffers,
                             ge2_stream_t stream);

// (옵션) 버퍼 설명자 예시
enum class ge2_dtype : uint8_t { f16=1, f32=2, i32=3 /* ... */ };

struct ge2_buf_desc {
  ge2_dtype dtype;
  uint8_t   rank;
  int64_t   shape[8];
  int64_t   stride[8];
};

extern "C" {
GE2_API const std::unordered_map<std::string, ge2_kernel_fn>&
ge_v2_kernel_table_raw();

GE2_API const std::unordered_map<std::string, int>&
ge_v2_capability_table_raw();
}

#define GE2_API_VERSION 1
