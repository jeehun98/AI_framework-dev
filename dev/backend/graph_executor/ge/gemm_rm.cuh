#pragma once
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "../ge/cuda_check.cuh"

// Row-major 매핑 래퍼들

// 단일 배치 GEMM (TF32)
inline void gemm_rm_tf32(
    cublasHandle_t h,
    bool transA, bool transB,
    int M, int N, int K,
    const float* A, int lda,
    const float* B, int ldb,
    float* C, int ldc,
    float alpha=1.f, float beta=0.f)
{
    cublasOperation_t opA = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t opB = transB ? CUBLAS_OP_T : CUBLAS_OP_N;

    CUBLAS_CHECK(
        cublasGemmEx(
            h,
            /*opB,opA*/ opB, opA,
            /*m,n,k*/   N,   M,   K,
            &alpha,
            /*B*/ B, CUDA_R_32F, ldb,
            /*A*/ A, CUDA_R_32F, lda,
            &beta,
            /*C*/ C, CUDA_R_32F, ldc,
            CUBLAS_COMPUTE_32F_FAST_TF32,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP
        )
    );
}

// Strided-batched GEMM (TF32)
inline void gemm_rm_strided_batched_tf32(
    cublasHandle_t h,
    bool transA, bool transB,
    int M, int N, int K,
    const float* A, int lda, long long strideA,
    const float* B, int ldb, long long strideB,
    float* C, int ldc, long long strideC,
    int batch,
    float alpha=1.f, float beta=0.f)
{
    cublasOperation_t opA = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t opB = transB ? CUBLAS_OP_T : CUBLAS_OP_N;

    CUBLAS_CHECK(
        cublasGemmStridedBatchedEx(
            h,
            /*opB,opA*/ opB, opA,
            /*m,n,k*/   N,   M,   K,
            &alpha,
            /*B*/ B, CUDA_R_32F, ldb, strideB,
            /*A*/ A, CUDA_R_32F, lda, strideA,
            &beta,
            /*C*/ C, CUDA_R_32F, ldc, strideC,
            /*batch*/ batch,
            CUBLAS_COMPUTE_32F_FAST_TF32,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP
        )
    );
}
