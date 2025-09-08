#pragma once
#include <cstddef>
#include <cstdint>
#include "normal_export.h"

namespace normal {

// CPU: out[0..n) 에 N(mean, std) 샘플 생성
NORMAL_API void generate_cpu(float* out, std::size_t n,
                             float mean = 0.f, float std = 1.f,
                             std::uint64_t seed = 0);

// (선택) 평균/표준편차 추정 유틸 (디버그/예제용)
NORMAL_API void estimate_stats(const float* data, std::size_t n,
                               double& mean, double& stdev);

#ifdef NORMAL_WITH_CUDA
// CUDA: out(호스트/디바이스 포인터 모두 허용; host면 내부에서 임시 버퍼 사용)
// stream==0 이면 default stream
NORMAL_API void generate_cuda(float* out, std::size_t n,
                              float mean = 0.f, float std = 1.f,
                              std::uint64_t seed = 0,
                              void* stream = nullptr /* cudaStream_t */);

// 장치에 직접 쓸 디바이스 포인터 버전 (out은 device 메모리)
NORMAL_API void generate_cuda_device(float* d_out, std::size_t n,
                                     float mean = 0.f, float std = 1.f,
                                     std::uint64_t seed = 0,
                                     void* stream = nullptr /* cudaStream_t */);
#endif

} // namespace normal
