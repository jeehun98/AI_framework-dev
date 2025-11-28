#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#ifndef CHECK_CUDA
#define CHECK_CUDA(call)                                                        \
    do {                                                                        \
        cudaError_t err__ = (call);                                             \
        if (err__ != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA Error %s:%d: %s\n",                           \
                    __FILE__, __LINE__, cudaGetErrorString(err__));             \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (0)
#endif

// =======================
// GEMM 설정
// =======================
// C[M x N] = A[M x K] * B[K x N]
constexpr int M = 1024;
constexpr int N = 1024;
constexpr int K = 1024;

// 타일 크기
constexpr int TILE_M = 16;
constexpr int TILE_N = 16;
constexpr int TILE_K = 16;

// =======================
// 유틸 함수
// =======================
void init_matrix(float* mat, int rows, int cols, float scale = 1.0f) {
    for (int i = 0; i < rows * cols; ++i) {
        mat[i] = scale * ((i % 13) - 6);  // -6 ~ +6
    }
}

float max_abs_diff(const float* ref, const float* test, int n) {
    float max_diff = 0.0f;
    for (int i = 0; i < n; ++i) {
        float d = std::fabs(ref[i] - test[i]);
        if (d > max_diff) max_diff = d;
    }
    return max_diff;
}

// host-side GEMM (검증용)
void gemm_host_ref(const float* A, const float* B, float* C,
                   int M, int N, int K) {
    for (int row = 0; row < M; ++row) {
        for (int col = 0; col < N; ++col) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k) {
                acc += A[row * K + k] * B[k * N + col];
            }
            C[row * N + col] = acc;
        }
    }
}

// =====================================================
// 1) Thread-level Tiled GEMM
//    - blockDim = (16,16) = 256 threads
//    - block 전체가 C의 16x16 타일 담당
//    - 각 thread가 C의 1 element 담당
// =====================================================
__global__ void gemm_thread_tiled_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K)
{
    __shared__ float As[TILE_M][TILE_K]; // 16x16
    __shared__ float Bs[TILE_K][TILE_N]; // 16x16

    int row = blockIdx.y * TILE_M + threadIdx.y;
    int col = blockIdx.x * TILE_N + threadIdx.x;

    float acc = 0.0f;

    for (int k0 = 0; k0 < K; k0 += TILE_K) {
        // load A tile
        int a_row = row;
        int a_col = k0 + threadIdx.x;
        if (a_row < M && a_col < K)
            As[threadIdx.y][threadIdx.x] = A[a_row * K + a_col];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        // load B tile
        int b_row = k0 + threadIdx.y;
        int b_col = col;
        if (b_row < K && b_col < N)
            Bs[threadIdx.y][threadIdx.x] = B[b_row * N + b_col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        // compute
        #pragma unroll
        for (int kk = 0; kk < TILE_K; ++kk) {
            acc += As[threadIdx.y][kk] * Bs[kk][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}

// =====================================================
// 2) Warp-level Tiled GEMM
//    - blockDim = 32 (1 warp per block)
//    - warp가 C의 16x16 tile 하나를 통째로 담당
//    - 각 lane가 2x4 = 8개의 C 요소(acc[2][4]) 담당
// =====================================================
__global__ void gemm_warp_tiled_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K)
{
    // block당 warp 1개만 사용
    const int lane = threadIdx.x; // 0..31

    // 이 warp가 담당하는 C 타일의 시작 위치
    int tile_row = blockIdx.y * TILE_M;
    int tile_col = blockIdx.x * TILE_N;

    __shared__ float As[TILE_M][TILE_K]; // 16x16
    __shared__ float Bs[TILE_K][TILE_N]; // 16x16

    // lane를 8x4 그리드로 재해석
    //   - row_group: 0..7
    //   - col_group: 0..3
    // 각 lane이 2x4 = 8개의 결과를 담당
    int row_group = lane % 8;   // 0..7
    int col_group = lane / 8;   // 0..3

    int row0 = tile_row + row_group * 2 + 0;
    int row1 = tile_row + row_group * 2 + 1;
    int col0 = tile_col + col_group * 4; // col0..col0+3

    float acc[2][4]; // [row][col]
    #pragma unroll
    for (int r = 0; r < 2; ++r)
        for (int c = 0; c < 4; ++c)
            acc[r][c] = 0.0f;

    for (int k0 = 0; k0 < K; k0 += TILE_K) {
        // -------------------------------
        // 1) shared memory load (warp-wide)
        //    16x16 = 256 요소를 32개 lane이 나눠서 로드
        // -------------------------------
        // A tile: [16 x 16]
        for (int idx = lane; idx < TILE_M * TILE_K; idx += warpSize) {
            int r = idx / TILE_K;  // 0..15
            int c = idx % TILE_K;  // 0..15
            int g_row = tile_row + r;
            int g_col = k0 + c;
            float val = 0.0f;
            if (g_row < M && g_col < K) {
                val = A[g_row * K + g_col];
            }
            As[r][c] = val;
        }

        // B tile: [16 x 16]
        for (int idx = lane; idx < TILE_K * TILE_N; idx += warpSize) {
            int r = idx / TILE_N;  // 0..15
            int c = idx % TILE_N;  // 0..15
            int g_row = k0 + r;
            int g_col = tile_col + c;
            float val = 0.0f;
            if (g_row < K && g_col < N) {
                val = B[g_row * N + g_col];
            }
            Bs[r][c] = val;
        }

        __syncwarp();

        // -------------------------------
        // 2) 타일 내부 연산 (warp-only)
        // -------------------------------
        #pragma unroll
        for (int kk = 0; kk < TILE_K; ++kk) {
            float a0 = 0.0f;
            float a1 = 0.0f;

            if (row0 < M) a0 = As[row0 - tile_row][kk];
            if (row1 < M) a1 = As[row1 - tile_row][kk];

            float b[4];
            #pragma unroll
            for (int c = 0; c < 4; ++c) {
                int col = col0 + c;
                if (col < N)
                    b[c] = Bs[kk][col - tile_col];
                else
                    b[c] = 0.0f;
            }

            // 2x4 outer product
            #pragma unroll
            for (int c = 0; c < 4; ++c) {
                acc[0][c] += a0 * b[c];
                acc[1][c] += a1 * b[c];
            }
        }

        __syncwarp();
    }

    // -------------------------------
    // 3) 결과 쓰기
    // -------------------------------
    if (row0 < M) {
        #pragma unroll
        for (int c = 0; c < 4; ++c) {
            int col = col0 + c;
            if (col < N) {
                C[row0 * N + col] = acc[0][c];
            }
        }
    }
    if (row1 < M) {
        #pragma unroll
        for (int c = 0; c < 4; ++c) {
            int col = col0 + c;
            if (col < N) {
                C[row1 * N + col] = acc[1][c];
            }
        }
    }
}

// =======================
// main: thread-level vs warp-level 비교
// =======================
int main()
{
    printf("=== Warp-level Tiling Test: Thread-level vs Warp-level GEMM ===\n");
    printf("GEMM config: C[%d x %d] = A[%d x %d] * B[%d x %d]\n",
           M, N, M, K, K, N);
    printf("Tile: M=N=K=%d\n\n", TILE_M);

    size_t bytesA = sizeof(float) * M * K;
    size_t bytesB = sizeof(float) * K * N;
    size_t bytesC = sizeof(float) * M * N;

    float *hA      = (float*)malloc(bytesA);
    float *hB      = (float*)malloc(bytesB);
    float *hC_ref  = (float*)malloc(bytesC);
    float *hC_tmp  = (float*)malloc(bytesC);

    init_matrix(hA, M, K, 0.01f);
    init_matrix(hB, K, N, 0.02f);

    printf("Computing host reference...\n");
    gemm_host_ref(hA, hB, hC_ref, M, N, K);

    float *dA, *dB, *dC;
    CHECK_CUDA(cudaMalloc(&dA, bytesA));
    CHECK_CUDA(cudaMalloc(&dB, bytesB));
    CHECK_CUDA(cudaMalloc(&dC, bytesC));

    CHECK_CUDA(cudaMemcpy(dA, hA, bytesA, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, hB, bytesB, cudaMemcpyHostToDevice));

    // 공용 grid: 16x16 타일 기준
    dim3 grid((N + TILE_N - 1) / TILE_N,
              (M + TILE_M - 1) / TILE_M);

    // -----------------------
    // 1) Thread-level 커널
    // -----------------------
    {
        printf("\n[thread_tiled] warm-up + timing\n");
        dim3 block(16, 16); // 256 threads
        gemm_thread_tiled_kernel<<<grid, block>>>(dA, dB, dC, M, N, K);
        CHECK_CUDA(cudaDeviceSynchronize());

        cudaEvent_t start, stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));

        CHECK_CUDA(cudaEventRecord(start));
        gemm_thread_tiled_kernel<<<grid, block>>>(dA, dB, dC, M, N, K);
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

        CHECK_CUDA(cudaMemcpy(hC_tmp, dC, bytesC, cudaMemcpyDeviceToHost));
        float diff = max_abs_diff(hC_ref, hC_tmp, M * N);

        printf("[thread_tiled] kernel time: %.3f ms, max diff = %e\n",
               ms, diff);

        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));
    }

    // -----------------------
    // 2) Warp-level 커널
    // -----------------------
    {
        printf("\n[warp_tiled] warm-up + timing\n");
        dim3 block(32, 1); // 1 warp per block
        gemm_warp_tiled_kernel<<<grid, block>>>(dA, dB, dC, M, N, K);
        CHECK_CUDA(cudaDeviceSynchronize());

        cudaEvent_t start, stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));

        CHECK_CUDA(cudaEventRecord(start));
        gemm_warp_tiled_kernel<<<grid, block>>>(dA, dB, dC, M, N, K);
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

        CHECK_CUDA(cudaMemcpy(hC_tmp, dC, bytesC, cudaMemcpyDeviceToHost));
        float diff = max_abs_diff(hC_ref, hC_tmp, M * N);

        printf("[warp_tiled]   kernel time: %.3f ms, max diff = %e\n",
               ms, diff);

        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));
    }

    CHECK_CUDA(cudaFree(dA));
    CHECK_CUDA(cudaFree(dB));
    CHECK_CUDA(cudaFree(dC));

    free(hA);
    free(hB);
    free(hC_ref);
    free(hC_tmp);

    CHECK_CUDA(cudaDeviceReset());
    return 0;
}

/* 
nvcc -O3 -arch=sm_86 -lineinfo -o gemm_warp_level_tiling_test.exe gemm_warp_level_tiling_test.cu

ncu --kernel-name regex:gemm_thread_tiled_kernel.* --metrics smsp__inst_executed.sum,smsp__pipe_fma_cycles_active.avg.pct_of_peak_sustained_elapsed ./gemm_warp_level_tiling_test.exe

ncu --kernel-name regex:gemm_warp_tiled_kernel.* --metrics smsp__inst_executed.sum,smsp__pipe_fma_cycles_active.avg.pct_of_peak_sustained_elapsed ./gemm_warp_level_tiling_test.exe
*/