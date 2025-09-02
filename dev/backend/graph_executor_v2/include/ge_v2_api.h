#pragma once
/**
 * @file ge_v2_api.h
 * @brief graph_executor_v2 네이티브 모듈의 "런타임 컨트랙트(ABI)"를 고정하는 헤더
 *
 * 파이썬 컴파일러/런타임과 C++/CUDA 실행기 사이의 경계를 명확히 합니다.
 * - 공통 타입 정의(포인터, 스트림, 커널 진입점 시그니처)
 * - 네이티브가 노출해야 하는 레지스트리(Symbol table) 형태
 * - (선택) 버전 등 호환성 헬퍼
 *
 * 중요 규약:
 * 1) launch_kernel(kernel_name, buffers, descs, stream) 에서
 *    buffers 는 "입력들 먼저, 그 다음 출력들" 순서로 uintptr_t(Device pointer) 리스트입니다.
 * 2) query_capability(op_type, ...) 에서 네이티브는
 *    key = "<OPTYPE>__<KERNEL_NAME>" 형태의 스코어 테이블을 제공합니다.
 * 3) 성공=0, 실패는 음수. (예: -1 invalid args, -2 device error, -3 not implemented 등)
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

// === 공통 타입: 포인터/스트림/커널 진입점 ===

using ge2_uintptr = std::uintptr_t;

/**
 * @note CUDA/ROCm/CPU 등 백엔드에 의존하지 않기 위해 stream은 opaque 포인터로 유지합니다.
 *       - CUDA의 경우 cudaStream_t 를 reinterpret_cast<void*> 로 전달/복구합니다.
 *       - CPU 백엔드는 nullptr 또는 내부 스케줄러 핸들 등으로 해석할 수 있습니다.
 */
using ge2_stream_t = void*;

/**
 * @brief 모든 커널 런처가 따라야 하는 C ABI 시그니처
 * @param buffers  입력→출력 순서의 디바이스 포인터 배열 (uintptr_t)
 * @param num_buffers buffers 길이
 * @param stream   opaque 스트림 포인터 (CUDA면 cudaStream_t)
 * @return 0=성공, 음수=오류
 */
using ge2_kernel_fn = int(*)(const ge2_uintptr* buffers, int num_buffers,
                             ge2_stream_t stream);

// === (옵션) 버퍼 디스크립터 예시 스케치 ===
// 초기 단계에서는 파이썬이 descs를 검증하고 C++은 opaque 로 둘 수 있으나,
// 장기적으로는 아래와 같은 구조를 직/역직렬화하여 사용해도 됩니다.

enum class ge2_dtype : uint8_t { f16=1, f32=2, i32=3 /* ... */ };

struct ge2_buf_desc {
  ge2_dtype dtype;
  uint8_t   rank;       // shape/stride 사용 길이
  int64_t   shape[8];   // 최대 8D까지 지원 예시
  int64_t   stride[8];  // rowmajor 시 stride 계산 가능
  // layout/device 등은 필요 시 확장 가능
};

// === 네이티브가 제공해야 하는 레지스트리 심볼 ===

extern "C" {
/**
 * @brief 커널 이름 → 커널 진입점 함수 포인터
 * 예: "gemm_bias_act_tc_f16" → &ge2_launch_gemm_bias_act_tc_f16
 */
GE2_API const std::unordered_map<std::string, ge2_kernel_fn>&
ge_v2_kernel_table_raw();

/**
 * @brief "<OPTYPE>__<KERNEL_NAME>" → score(int)
 * 예: "GEMM_BIAS_ACT__gemm_bias_act_tc_f16" → 80
 * 선택기가 네이티브 쪽의 스코어와 파이썬 휴리스틱을 합산해 최종 선택합니다.
 */
GE2_API const std::unordered_map<std::string, int>&
ge_v2_capability_table_raw();
}

// === 버전/호환 ===
#define GE2_API_VERSION 1
