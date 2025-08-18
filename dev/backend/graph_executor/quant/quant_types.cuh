#pragma once
#include <unordered_map>
#include <string>
#include <vector>
#include <cuda_runtime.h>

#ifndef CUDA_CHECK
#define CUDA_CHECK(x) do { cudaError_t _e=(x); if(_e!=cudaSuccess){ \
  fprintf(stderr,"[CUDA] %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); exit(1);} } while(0)
#endif

namespace quant {

struct QuantParams {
    float scale = 1.0f;
    int32_t zero_point = 0; // 대칭이면 0
};

struct MinMax {
    float min = 0.f, max = 0.f;
    bool initialized = false;
};

struct WeightInt8 {
    // row-major Wq: [OC, K], 또한 GEMM을 위해 column-major Bq도 준비
    int8_t* row_major = nullptr;  // [OC, K]
    int8_t* col_major = nullptr;  // [K, OC] (dp4a 접근 최적)
    float*  per_channel_scales = nullptr; // [OC]
    int OC=0, K=0;
};

struct QuantCache {
    // 활성화 텐서: 텐서ID -> (minmax, qp)
    std::unordered_map<std::string, MinMax> act_minmax;
    std::unordered_map<std::string, QuantParams> act_qparams;
    // 가중치: weightID -> 정수화 결과
    std::unordered_map<std::string, WeightInt8> weights_q;
};

struct RuntimeFlags {
    bool observe_enabled = false;
    bool quant_enabled = false;
};

RuntimeFlags& runtime();
QuantCache& cache();

} // namespace quant
