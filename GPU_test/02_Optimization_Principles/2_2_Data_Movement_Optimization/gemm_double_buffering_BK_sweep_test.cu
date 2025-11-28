#include <cstdio>
#include <cstdlib>
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

// block tile M,N 고정
constexpr int BM = 16;
constexpr int BN = 16;

// =======================
// 호스트 유틸
// =======================
void init_matrix(float* a, int rows, int cols, float scale = 1.0f) {
    for (int i = 0; i < rows * cols; ++i) {
        a[i] = scale * ((i % 13) - 6); // -6 ~ 6
    }
}

void gemm_host_ref(const float* A, const float* B, float* C,
                   int M, int N, int K)
{
    // A: [M x K], B: [K x N], C: [M x N] (row-major)
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

float max_abs_diff(const float* ref, const float* test, int n) {
    float maxd = 0.0f;
    for (int i = 0; i < n; ++i) {
        float d = fabsf(ref[i] - test[i]);
        if (d > maxd) maxd = d;
    }
    return maxd;
}

// =======================
// Double-buffer GEMM (BK 템플릿)
// =======================
// BK = 8,16,32 등에 대해 인스턴스 생성.
// shared double buffer:
//   As[2][BM][BK], Bs[2][BK][BN]
template<int BK>
__global__ void gemm_double_buffer_BK_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K)
{
    __shared__ float As[2][BM][BK];
    __shared__ float Bs[2][BK][BN];

    int tx = threadIdx.x; // 0..BN-1 (col in block)
    int ty = threadIdx.y; // 0..BM-1 (row in block)

    int row = blockIdx.y * BM + ty;
    int col = blockIdx.x * BN + tx;

    float acc = 0.0f;

    const int num_tiles = (K + BK - 1) / BK;

    int read_buf  = 0;
    int write_buf = 1;

    // Prologue: tile 0 load
    if (num_tiles > 0) {
        int k0 = 0;
        int kA = k0 + tx;
        int kB = k0 + ty;

        if (row < M && kA < K)
            As[read_buf][ty][tx] = A[row * K + kA];
        else
            As[read_buf][ty][tx] = 0.0f;

        if (kB < K && col < N)
            Bs[read_buf][ty][tx] = B[kB * N + col];
        else
            Bs[read_buf][ty][tx] = 0.0f;
    }

    __syncthreads(); // tile0 준비 완료

    // Main loop
    for (int tile = 0; tile < num_tiles; ++tile) {
        int k0 = tile * BK;

        // 1) 현재 tile(read_buf)로 compute
        #pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            acc += As[read_buf][ty][kk] * Bs[read_buf][kk][tx];
        }

        // 2) 다음 tile을 write_buf로 prefetch
        int next_tile = tile + 1;
        if (next_tile < num_tiles) {
            int k_next = next_tile * BK;
            int kA = k_next + tx;
            int kB = k_next + ty;

            if (row < M && kA < K)
                As[write_buf][ty][tx] = A[row * K + kA];
            else
                As[write_buf][ty][tx] = 0.0f;

            if (kB < K && col < N)
                Bs[write_buf][ty][tx] = B[kB * N + col];
            else
                Bs[write_buf][ty][tx] = 0.0f;
        }

        __syncthreads(); // write_buf를 다음 iteration에서 read_buf로 쓰기 전에 sync

        // 3) buffer swap
        int tmp    = read_buf;
        read_buf   = write_buf;
        write_buf  = tmp;
    }

    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}

// BK에 따른 shared memory 크기 (bytes) 계산
size_t smem_bytes_for_BK(int BK) {
    // As[2][BM][BK] + Bs[2][BK][BN]
    // float = 4 bytes
    return static_cast<size_t>(2 * BM * BK + 2 * BK * BN) * sizeof(float);
}

// =======================
// main: BK = 8,16,32 sweep
// =======================
int main()
{
    printf("=== Double Buffering Test 2: BK loop size sweep ===\n");
    printf("GEMM config: C[%d x %d] = A[%d x %d] * B[%d x %d]\n",
           M, N, M, K, K, N);
    printf("Block tile: BM=%d, BN=%d, BK in {8,16,32}\n\n", BM, BN);

    size_t bytesA = sizeof(float) * M * K;
    size_t bytesB = sizeof(float) * K * N;
    size_t bytesC = sizeof(float) * M * N;

    float *hA = (float*)malloc(bytesA);
    float *hB = (float*)malloc(bytesB);
    float *hC_ref    = (float*)malloc(bytesC);
    float *hC_device = (float*)malloc(bytesC);

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

    dim3 block(BN, BM); // (16,16)
    dim3 grid((N + BN - 1) / BN,
              (M + BM - 1) / BM);

    // BK configs
    const int BK_configs[3] = {8, 16, 32};

    printf("\nRunning BK configs...\n");

    // BK=8
    {
        const int BK = 8;
        size_t smem_bytes = smem_bytes_for_BK(BK);

        printf("Config BK8   BK=%2d  | smem=%.1f KB\n",
               BK, smem_bytes / 1024.0f);

        // warm-up
        gemm_double_buffer_BK_kernel<BK><<<grid, block>>>(
            dA, dB, dC, M, N, K);
        CHECK_CUDA(cudaDeviceSynchronize());

        cudaEvent_t start, stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));

        CHECK_CUDA(cudaEventRecord(start));
        gemm_double_buffer_BK_kernel<BK><<<grid, block>>>(
            dA, dB, dC, M, N, K);
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

        CHECK_CUDA(cudaMemcpy(hC_device, dC, bytesC, cudaMemcpyDeviceToHost));
        float diff = max_abs_diff(hC_ref, hC_device, M * N);

        printf("Config BK8   -> time=%.3f ms | max diff=%e\n\n", ms, diff);

        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));
    }

    // BK=16
    {
        const int BK = 16;
        size_t smem_bytes = smem_bytes_for_BK(BK);

        printf("Config BK16  BK=%2d  | smem=%.1f KB\n",
               BK, smem_bytes / 1024.0f);

        gemm_double_buffer_BK_kernel<BK><<<grid, block>>>(
            dA, dB, dC, M, N, K);
        CHECK_CUDA(cudaDeviceSynchronize());

        cudaEvent_t start, stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));

        CHECK_CUDA(cudaEventRecord(start));
        gemm_double_buffer_BK_kernel<BK><<<grid, block>>>(
            dA, dB, dC, M, N, K);
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

        CHECK_CUDA(cudaMemcpy(hC_device, dC, bytesC, cudaMemcpyDeviceToHost));
        float diff = max_abs_diff(hC_ref, hC_device, M * N);

        printf("Config BK16  -> time=%.3f ms | max diff=%e\n\n", ms, diff);

        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));
    }

    // BK=32
    {
        const int BK = 32;
        size_t smem_bytes = smem_bytes_for_BK(BK);

        printf("Config BK32  BK=%2d  | smem=%.1f KB\n",
               BK, smem_bytes / 1024.0f);

        gemm_double_buffer_BK_kernel<BK><<<grid, block>>>(
            dA, dB, dC, M, N, K);
        CHECK_CUDA(cudaDeviceSynchronize());

        cudaEvent_t start, stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));

        CHECK_CUDA(cudaEventRecord(start));
        gemm_double_buffer_BK_kernel<BK><<<grid, block>>>(
            dA, dB, dC, M, N, K);
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

        CHECK_CUDA(cudaMemcpy(hC_device, dC, bytesC, cudaMemcpyDeviceToHost));
        float diff = max_abs_diff(hC_ref, hC_device, M * N);

        printf("Config BK32  -> time=%.3f ms | max diff=%e\n\n", ms, diff);

        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));
    }

    CHECK_CUDA(cudaFree(dA));
    CHECK_CUDA(cudaFree(dB));
    CHECK_CUDA(cudaFree(dC));

    free(hA);
    free(hB);
    free(hC_ref);
    free(hC_device);

    CHECK_CUDA(cudaDeviceReset());
    return 0;
}


/*
nvcc -O3 -arch=sm_86 -lineinfo -o gemm_double_buffering_BK_sweep_test.exe gemm_double_buffering_BK_sweep_test.cu

ncu --kernel-name regex:gemm_double_buffer_BK_kernel.* --metrics smsp__pipe_fma_cycles_active.avg.pct_of_peak_sustained_elapsed,smsp__warp_issue_stalled_membar_per_warp_active.avg,smsp__warp_issue_stalled_membar_per_warp_active.pct,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum   ./gemm_double_buffering_BK_sweep_test.exe


*/