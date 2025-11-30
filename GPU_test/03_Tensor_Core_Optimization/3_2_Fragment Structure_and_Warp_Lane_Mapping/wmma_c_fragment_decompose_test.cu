#include <cstdio>
#include <vector>
#include <utility>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

constexpr int M = 16;
constexpr int N = 16;
constexpr int K = 16;
constexpr int MMA_ITERS = 4; // mma_sync 반복 횟수

// 단일 warp (32 threads)에서:
// 1) C fragment에 코드 값(lane*100 + frag_idx) 채우기
// 2) A/B fragment에 1.0 채우고 mma_sync 반복해서 C_mma에 누적
// 3) 두 개의 C 타일 (C_map, C_mma)에 store
__global__ void wmma_c_fragment_decompose_kernel(float* C_map, float* C_mma)
{
    const int lane_id = threadIdx.x & 31;

    // C fragment (mapping 확인용)
    wmma::fragment<wmma::accumulator, M, N, K, float> c_frag;
    // C fragment (mma_sync 누적용)
    wmma::fragment<wmma::accumulator, M, N, K, float> c_accum;

    // A/B fragment (mma_sync용)
    wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::col_major> b_frag;

    // 1) mapping용 C fragment: lane*100 + frag_idx
    #pragma unroll
    for (int i = 0; i < c_frag.num_elements; ++i) {
        c_frag.x[i] = static_cast<float>(lane_id * 100 + i);
    }

    // 2) mma_accum용 C fragment 초기화 (0)
    #pragma unroll
    for (int i = 0; i < c_accum.num_elements; ++i) {
        c_accum.x[i] = 0.0f;
    }

    // 3) A/B fragment를 1.0으로 채우기
    #pragma unroll
    for (int i = 0; i < a_frag.num_elements; ++i) {
        a_frag.x[i] = __float2half(1.0f);
    }
    #pragma unroll
    for (int i = 0; i < b_frag.num_elements; ++i) {
        b_frag.x[i] = __float2half(1.0f);
    }

    // 4) mma_sync 반복: c_accum = A * B + c_accum
    #pragma unroll
    for (int it = 0; it < MMA_ITERS; ++it) {
        wmma::mma_sync(c_accum, a_frag, b_frag, c_accum);
    }

    // 5) warp 전체가 합쳐서 C_map, C_mma에 16x16 row-major로 저장
    // (모든 lane이 같은 포인터, 같은 leading dimension으로 호출해야 함)
    wmma::store_matrix_sync(C_map, c_frag, M, wmma::mem_row_major);
    wmma::store_matrix_sync(C_mma, c_accum, M, wmma::mem_row_major);
}

int main()
{
    printf("WMMA C fragment decompose & MMA reuse test (m16n16k16)\n");

    const int num_elements = M * N;
    const size_t bytes = num_elements * sizeof(float);

    float* d_C_map = nullptr;
    float* d_C_mma = nullptr;

    cudaMalloc(&d_C_map, bytes);
    cudaMalloc(&d_C_mma, bytes);

    cudaMemset(d_C_map, 0, bytes);
    cudaMemset(d_C_mma, 0, bytes);

    dim3 block(32, 1, 1); // single warp
    dim3 grid(1, 1, 1);

    wmma_c_fragment_decompose_kernel<<<grid, block>>>(d_C_map, d_C_mma);
    cudaDeviceSynchronize();

    // host 버퍼로 복사
    std::vector<float> h_C_map(num_elements);
    std::vector<float> h_C_mma(num_elements);

    cudaMemcpy(h_C_map.data(), d_C_map, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_C_mma.data(), d_C_mma, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_C_map);
    cudaFree(d_C_mma);

    // ------------------------------------------------------------------------
    // 1) C fragment 8-element → (row, col) 매핑 복원
    //    value = lane*100 + frag_idx 라고 넣었으므로
    //    lane = value / 100, frag_idx = value % 100
    // ------------------------------------------------------------------------
    struct Coord { int r, c; };
    std::vector<Coord> coords[32]; // lane별 8개 좌표
    for (int l = 0; l < 32; ++l) {
        coords[l].resize(8, {-1, -1});
    }

    int ldc = M;
    for (int r = 0; r < M; ++r) {
        for (int c = 0; c < N; ++c) {
            int v = static_cast<int>(h_C_map[r * ldc + c] + 0.5f);
            int lane = v / 100;
            int frag_idx = v % 100;
            if (lane >= 0 && lane < 32 && frag_idx >= 0 && frag_idx < 8) {
                coords[lane][frag_idx] = {r, c};
            }
        }
    }

    printf("\n=== [C fragment 8-element mapping per lane] ===\n");
    for (int lane = 0; lane < 32; ++lane) {
        printf("lane %2d:", lane);
        for (int i = 0; i < 8; ++i) {
            auto p = coords[lane][i];
            printf(" f%-2d=(%2d,%2d)", i, p.r, p.c);
        }
        printf("\n");
    }

    // 참고: 전체 16x16 타일도 같이 출력하고 싶으면 아래 주석 해제
    printf("\n=== [C_map raw 16x16 tile (value = lane*100 + frag_idx)] ===\n");
    for (int r = 0; r < M; ++r) {
        for (int c = 0; c < N; ++c) {
            printf("%5.0f ", h_C_map[r * ldc + c]);
        }
        printf("\n");
    }

    // ------------------------------------------------------------------------
    // 2) MMA 반복에 따른 accumulator reuse 확인
    //    A=B=1 이고 m16n16k16 이므로, 한 번의 mma_sync 후
    //      C[i,j] = 16
    //    MMA_ITERS 번 반복하면
    //      C[i,j] = 16 * MMA_ITERS
    // ------------------------------------------------------------------------
    float minv = 1e30f, maxv = -1e30f;
    for (int i = 0; i < num_elements; ++i) {
        float v = h_C_mma[i];
        if (v < minv) minv = v;
        if (v > maxv) maxv = v;
    }

    printf("\n=== [C_mma after %d mma_sync iterations] ===\n", MMA_ITERS);
    printf("expected each C[i,j] ≈ %d (16 * %d)\n", 16 * MMA_ITERS, MMA_ITERS);
    printf("min = %.1f, max = %.1f\n", minv, maxv);

    // 상위 4x4만 샘플로 출력
    printf("\nC_mma[0:4, 0:4]:\n");
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            printf("%6.1f ", h_C_mma[r * ldc + c]);
        }
        printf("\n");
    }

    cudaDeviceReset();
    return 0;
}
/*
nvcc -std=c++17 -O3   -arch=sm_86   wmma_c_fragment_decompose_test.cu   -o wmma_c_fragment_decompose_test.exe
*/
