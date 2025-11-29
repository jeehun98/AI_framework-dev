// mma_pipeline_stage_test.cu
// Test 3: 2-stage vs 4-stage vs 6-stage MMA pipeline (WMMA 기반)

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <mma.h>

using namespace nvcuda;

// ---------------------------
// Config
// ---------------------------
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

constexpr int M = 1024;
constexpr int N = 1024;
constexpr int K = 1024;

// 한 block이 16x16 tile 하나 담당 (1 warp/block)
const dim3 BLOCK_DIM(32, 1, 1);

// ---------------------------
// CUDA 체크 매크로
// ---------------------------
#define CHECK_CUDA(call)                                                          \
    do {                                                                          \
        cudaError_t err__ = (call);                                               \
        if (err__ != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA Error %s:%d: %s\n", __FILE__, __LINE__,         \
                    cudaGetErrorString(err__));                                   \
            std::exit(EXIT_FAILURE);                                              \
        }                                                                         \
    } while (0)

// ---------------------------
// Host helper
// ---------------------------
void init_half_matrix(__half* h, int rows, int cols) {
    for (int i = 0; i < rows * cols; ++i) {
        float v = static_cast<float>(rand()) / RAND_MAX - 0.5f;
        h[i] = __float2half(v);
    }
}

void init_float_matrix(float* h, int rows, int cols) {
    for (int i = 0; i < rows * cols; ++i) {
        float v = static_cast<float>(rand()) / RAND_MAX - 0.5f;
        h[i] = v;
    }
}

// ---------------------------
// WMMA MMA pipeline kernel
// PIPE_STAGES: 2, 4, 6 등
//
// 각 block:
//   - (blockIdx.y, blockIdx.x)에 해당하는 16x16 C tile 담당
//   - 1 warp/block (32 threads)
//   - K dimension에서 WMMA_K(=16) 단위로 나누고,
//     PIPE_STAGES 만큼 묶어서 unroll된 MMA burst 실행
// ---------------------------
template <int PIPE_STAGES>
__global__ void gemm_mma_pipeline_stage_kernel(
    const __half* __restrict__ A,  // row-major, lda = K
    const __half* __restrict__ B,  // col-major, ldb = K
    float* __restrict__ C,         // row-major, ldc = N
    int M, int N, int K)
{
    // 단일 warp kernel 가정
    int warp_id = (threadIdx.x >> 5); // 여기서는 항상 0
    int lane_id = (threadIdx.x & 31);
    if (warp_id != 0) return;

    // 이 block이 담당하는 tile의 시작 좌표
    int tile_row = blockIdx.y * WMMA_M;
    int tile_col = blockIdx.x * WMMA_N;

    if (tile_row >= M || tile_col >= N) return;

    // accumulator fragment (FP32)
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    // A/B fragments를 PIPE_STAGES 개 준비
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major>
        a_frags[PIPE_STAGES];
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::col_major>
        b_frags[PIPE_STAGES];

    int num_k_tiles = K / WMMA_K;

    // prologue: 처음 PIPE_STAGES 타일 load
    int k_tile = 0;
    for (int s = 0; s < PIPE_STAGES && k_tile + s < num_k_tiles; ++s) {
        int k0 = (k_tile + s) * WMMA_K;

        const __half* tile_ptr_A = A + tile_row * K + k0;      // row-major
        const __half* tile_ptr_B = B + tile_col * K + k0;      // col-major (K x N)

        wmma::load_matrix_sync(a_frags[s], tile_ptr_A, K);
        wmma::load_matrix_sync(b_frags[s], tile_ptr_B, K);
    }

    // main loop: PIPE_STAGES 단위로 MMA burst
    while (k_tile < num_k_tiles) {
        // 1) 현재 윈도우에 대해 MMA burst (PIPE_STAGES번)
        for (int s = 0; s < PIPE_STAGES && k_tile + s < num_k_tiles; ++s) {
            wmma::mma_sync(c_frag, a_frags[s], b_frags[s], c_frag);
        }

        // 2) 다음 윈도우를 위해 fragment prefetch
        k_tile += PIPE_STAGES;
        if (k_tile >= num_k_tiles) break;

        for (int s = 0; s < PIPE_STAGES && k_tile + s < num_k_tiles; ++s) {
            int k0 = (k_tile + s) * WMMA_K;

            const __half* tile_ptr_A = A + tile_row * K + k0;
            const __half* tile_ptr_B = B + tile_col * K + k0;

            wmma::load_matrix_sync(a_frags[s], tile_ptr_A, K);
            wmma::load_matrix_sync(b_frags[s], tile_ptr_B, K);
        }
    }

    // C에 store
    float* tile_ptr_C = C + tile_row * N + tile_col;
    wmma::store_matrix_sync(tile_ptr_C, c_frag, N, wmma::mem_row_major);
}

// ---------------------------
// Host: reference FP32 GEMM (간단 구현, 검증용)
// C = A * B, A/B/C: row-major (B는 여기서 col-major로 쓰지 않음, 비교용)
// ---------------------------
void gemm_ref(const float* A, const float* B, float* C,
              int M, int N, int K)
{
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            double acc = 0.0;
            for (int k = 0; k < K; ++k) {
                acc += static_cast<double>(A[m * K + k]) *
                       static_cast<double>(B[k * N + n]);
            }
            C[m * N + n] = static_cast<float>(acc);
        }
    }
}

// ---------------------------
// diff check
// ---------------------------
float max_abs_diff(const float* C1, const float* C2, int elems) {
    float max_diff = 0.0f;
    for (int i = 0; i < elems; ++i) {
        float d = fabsf(C1[i] - C2[i]);
        if (d > max_diff) max_diff = d;
    }
    return max_diff;
}

// ---------------------------
// main
// ---------------------------
int main() {
    printf("=== MMA Pipeline Stage Test: PIPE_STAGES = 2,4,6 ===\n");
    printf("GEMM: C[%d x %d] = A[%d x %d] * B[%d x %d]\n", M, N, M, K, K, N);

    // Host buffers
    __half* hA_half = nullptr;
    __half* hB_half = nullptr;
    float*  hA_f32  = nullptr;
    float*  hB_f32  = nullptr;
    float*  hC_ref  = nullptr;
    float*  hC_out  = nullptr;

    hA_half = (__half*)malloc(sizeof(__half) * M * K);
    hB_half = (__half*)malloc(sizeof(__half) * K * N);
    hA_f32  = (float*)malloc(sizeof(float) * M * K);
    hB_f32  = (float*)malloc(sizeof(float) * K * N);
    hC_ref  = (float*)malloc(sizeof(float) * M * N);
    hC_out  = (float*)malloc(sizeof(float) * M * N);

    if (!hA_half || !hB_half || !hA_f32 || !hB_f32 || !hC_ref || !hC_out) {
        fprintf(stderr, "Host alloc failed\n");
        return EXIT_FAILURE;
    }

    // init
    srand(0);
    init_float_matrix(hA_f32, M, K);
    init_float_matrix(hB_f32, K, N);

    // FP16 버전도 같이 생성 (device용)
    for (int i = 0; i < M * K; ++i)
        hA_half[i] = __float2half(hA_f32[i]);
    for (int i = 0; i < K * N; ++i)
        hB_half[i] = __float2half(hB_f32[i]);

    // reference GEMM (FP32, row-major * row-major)
    printf("Computing reference GEMM (host FP32)...\n");
    gemm_ref(hA_f32, hB_f32, hC_ref, M, N, K);

    // Device buffers
    __half *dA = nullptr, *dB = nullptr;
    float  *dC = nullptr;

    CHECK_CUDA(cudaMalloc(&dA, sizeof(__half) * M * K));
    CHECK_CUDA(cudaMalloc(&dB, sizeof(__half) * K * N));
    CHECK_CUDA(cudaMalloc(&dC, sizeof(float)  * M * N));

    CHECK_CUDA(cudaMemcpy(dA, hA_half, sizeof(__half) * M * K,
                          cudaMemcpyHostToDevice));
    // B는 MMA kernel에서 col-major로 해석.
    // 여기선 row-major 데이터를 그대로 주고, load 시 col-major indexing 사용.
    CHECK_CUDA(cudaMemcpy(dB, hB_half, sizeof(__half) * K * N,
                          cudaMemcpyHostToDevice));

    dim3 gridDim(N / WMMA_N, M / WMMA_M, 1);

    // timing
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    auto run_config = [&](int stages) {
        // C 초기화
        CHECK_CUDA(cudaMemset(dC, 0, sizeof(float) * M * N));
        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaEventRecord(start));
        switch (stages) {
            case 2:
                gemm_mma_pipeline_stage_kernel<2><<<gridDim, BLOCK_DIM>>>(
                    dA, dB, dC, M, N, K);
                break;
            case 4:
                gemm_mma_pipeline_stage_kernel<4><<<gridDim, BLOCK_DIM>>>(
                    dA, dB, dC, M, N, K);
                break;
            case 6:
                gemm_mma_pipeline_stage_kernel<6><<<gridDim, BLOCK_DIM>>>(
                    dA, dB, dC, M, N, K);
                break;
            default:
                printf("Unsupported stages=%d\n", stages);
                return;
        }
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

        // copy back
        CHECK_CUDA(cudaMemcpy(hC_out, dC, sizeof(float) * M * N,
                              cudaMemcpyDeviceToHost));

        float max_diff = max_abs_diff(hC_out, hC_ref, M * N);

        // FLOPs = 2 * M * N * K
        double flops = 2.0 * (double)M * (double)N * (double)K;
        double gflops = flops / (ms * 1e6); // ms -> s

        printf("PIPE_STAGES=%d  |  time=%.3f ms  |  GFLOPs=%.2f  |  max diff=%.6e\n",
               stages, ms, gflops, max_diff);
    };

    printf("Running MMA pipeline configs...\n");
    run_config(2);
    run_config(4);
    run_config(6);

    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    free(hA_half);
    free(hB_half);
    free(hA_f32);
    free(hB_f32);
    free(hC_ref);
    free(hC_out);

    return 0;
}

/*
# 컴파일 (SM 86 기준)
nvcc -O3 -arch=sm_86 mma_pipeline_stage_test.cu -o mma_pipeline_stage_test.exe

# PIPE_STAGES=2 프로파일
ncu --kernel-name regex:gemm_mma_pipeline_stage_kernel.* --metrics smsp__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed,smsp__pipe_fma_cycles_active.avg.pct_of_peak_sustained_elapsed,smsp__warp_issue_stalled_memory_dependency_per_warp_active.avg     ./mma_pipeline_stage_test.exe


*/
