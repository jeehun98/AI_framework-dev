#pragma once
/**
 * @file ge_v2_api.h
 * @brief graph_executor_v2 네이티브 모듈의 "런타임 컨트랙트(ABI)" 정의
 *
 * 핵심 규약:
 * 1) launch_kernel(kernel_name, buffers, descs, stream)
 *    - buffers: 입력들 → 출력들 → (옵션) 커널별 파라미터-블록(Host memory ptr) 순서
 *      * 입력/출력은 device pointer(uintptr_t)
 *      * 맨 마지막에 올 수 있는 "파라미터-블록"은 Host 메모리 주소이며,
 *        커널 런처가 grid/block/shape 등을 계산할 때 읽을 수 있다(디바이스에서 참조 X).
 * 2) query_capability(op_type, ...)
 *    - "<OPTYPE>__<KERNEL_NAME>" -> score(int) 테이블
 * 3) 반환값: 0=성공, 음수=오류(-1 invalid args, -2 device error, -3 not implemented ...)
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

using ge2_uintptr  = std::uintptr_t;  // device/host pointer as integer
using ge2_stream_t = void*;           // opaque stream (CUDA면 cudaStream_t)

// 모든 커널 런처가 지켜야 하는 C ABI
using ge2_kernel_fn = int(*)(const ge2_uintptr* buffers, int num_buffers,
                             ge2_stream_t stream);

// (옵션) 파라미터 스키마 확장을 위한 예시
enum class ge2_dtype : uint8_t { f16=1, f32=2, i32=3 /* ... */ };

struct ge2_buf_desc {
  ge2_dtype dtype;
  uint8_t   rank;
  int64_t   shape[8];
  int64_t   stride[8];
  // layout/device 등은 필요 시 확장
};

extern "C" {
// 커널 이름 -> 함수 포인터 (예: "gemm_bias_act_f32" → &ge2_launch_…)
GE2_API const std::unordered_map<std::string, ge2_kernel_fn>&
ge_v2_kernel_table_raw();

// "<OPTYPE>__<KERNEL_NAME>" -> score(int) (예: "GEMM_BIAS_ACT__gemm_bias_act_f32" → 80)
GE2_API const std::unordered_map<std::string, int>&
ge_v2_capability_table_raw();
}

#define GE2_API_VERSION 1
