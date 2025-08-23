#pragma once
#include <cublas_v2.h>
#include "../ge/cuda_check.cuh"

// 전역 cuBLAS 핸들 싱글턴
inline cublasHandle_t ge_cublas() {
    static cublasHandle_t h = nullptr;
    if (!h) {
        CUBLAS_CHECK(cublasCreate(&h));
        // Ampere+ TF32 쓰려면 필요시 주석 해제
        // CUBLAS_CHECK(cublasSetMathMode(h, CUBLAS_TF32_TENSOR_OP_MATH));
    }
    return h;
}
