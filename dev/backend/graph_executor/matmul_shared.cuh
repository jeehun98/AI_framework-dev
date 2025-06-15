#pragma once

#define TILE_WIDTH 16

// 입력: A (1 x in_dim), B (in_dim x out_dim) → C (1 x out_dim)
// 단일 스레드용 shared memory 기반 matmul (배치 단위 내부에서만 사용)
__device__ void matmul_shared(float* A, float* B, float* C, int in_dim, int out_dim) {
    // Shared memory는 thread block 단위에서 선언되어야 하므로, 여기서는 생략
    // 대신 구조만 타일 구조를 따름 (실제 병렬 타일링은 나중에)
    for (int j = 0; j < out_dim; ++j) {
        float sum = 0.0f;
        for (int i = 0; i < in_dim; ++i) {
            sum += A[i] * B[i * out_dim + j];  // row vector × col of W
        }
        C[j] = sum;
    }
}
