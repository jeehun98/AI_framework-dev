#pragma once
#include "quant/quant_types.cuh"

#if __CUDACC_VER_MAJOR__ < 8
#error "dp4a requires CUDA 8.0+ and SM6.1+"
#endif

namespace quant {

// dp4a: s8x4 dot → s32
// A: [M,K] row-major (int8), B: [K,N] col-major (int8), C: [M,N] int32
__global__ void k_gemm_s8s8_s32_dp4a(const int8_t* __restrict__ A, const int8_t* __restrict__ B,
                                     int32_t* __restrict__ C, int M, int N, int K){
#if __CUDA_ARCH__ >= 610
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (m>=M || n>=N) return;
    int32_t acc = 0;

    // K multiple of 4 요구
    const int8_t* a_ptr = A + m*K;
    const int8_t* b_col = B + n; // col-major: (k,n) -> k*N + n, 여기선 B는 [K,N] col-major라 stride N
    for (int k4 = 0; k4 < K; k4 += 4){
        // 안전한 4바이트 로드: char4 사용
        char4 a4 = *reinterpret_cast<const char4*>(a_ptr + k4);
        // B는 col-major라 (k+i,n)가 연속
        char4 b4 = *reinterpret_cast<const char4*>(b_col + k4 * N);

        int ai = *reinterpret_cast<int*>(&a4);
        int bi = *reinterpret_cast<int*>(&b4);
        acc = __dp4a(ai, bi, acc);
    }
    C[m*N + n] = acc;
#else
    // 호환 아키텍처가 아니면 컴파일러 단계에서 에러가 나도록 위 #if 사용
#endif
}

} // namespace quant
