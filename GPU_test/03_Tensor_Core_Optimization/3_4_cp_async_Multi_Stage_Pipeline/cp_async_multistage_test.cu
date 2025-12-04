// cp_async_multistage_test.cu

#include <cstdio>
#include <cstdlib>
#include <cuda.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

#define CUDA_CHECK(call)                                                \
    do {                                                                \
        cudaError_t err = (call);                                       \
        if (err != cudaSuccess) {                                       \
            fprintf(stderr, "CUDA error %s:%d: %s\n",                   \
                    __FILE__, __LINE__, cudaGetErrorString(err));       \
            exit(EXIT_FAILURE);                                         \
        }                                                               \
    } while (0)

constexpr int TILE_M = 16;
constexpr int TILE_N = 16;
constexpr int TILE_K = 16;

// ---- cp.async helpers (16B) ----
__device__ __forceinline__
void cp_async_16b(void* smem_ptr, const void* gmem_ptr) {
#if __CUDA_ARCH__ >= 800
    // shared pointer -> 32bit shared address
    unsigned smem_addr = static_cast<unsigned>(__cvta_generic_to_shared(smem_ptr));

    asm volatile(
        "cp.async.ca.shared.global [%0], [%1], %2;\n"
        :
        : "r"(smem_addr), "l"(gmem_ptr), "n"(16)
    );
#endif
}

__device__ __forceinline__
void cp_async_commit_group() {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;\n" ::);
#endif
}

__device__ __forceinline__
void cp_async_wait_all() {
#if __CUDA_ARCH__ >= 800
    // wait_group 0: 모든 async 그룹 완료 대기
    asm volatile("cp.async.wait_group 0;\n" ::);
#endif
}

// -----------------------------
//  WMMA GEMM + cp.async multi-stage
//  STAGES = 2 / 3 / 4 비교용
// -----------------------------
template<int STAGES>
__global__ void wmma_cp_async_multistage_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
#if __CUDA_ARCH__ >= 800
    const int warp_lane = threadIdx.x;  // 0..31 (한 블록당 1 warp 가정)

    const int tile_m = blockIdx.y;      // warp tile row index
    const int tile_n = blockIdx.x;      // warp tile col index

    const int row_start = tile_m * TILE_M;
    const int col_start = tile_n * TILE_N;

    const int num_k_tiles = K / TILE_K;

    extern __shared__ half shared[];
    // [A stage buffer][B stage buffer]
    half* smemA = shared;
    half* smemB = shared + STAGES * TILE_M * TILE_K;

    // 각 stage마다 16x16 tile 하나
    const int stage_tile_elems = TILE_M * TILE_K; // 256

    // WMMA fragment
    wmma::fragment<wmma::matrix_a, TILE_M, TILE_N, TILE_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, TILE_M, TILE_N, TILE_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, TILE_M, TILE_N, TILE_K, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    // 한 stage에 대해 A/B tile을 cp.async로 옮기는 helper
    auto cp_async_load_tile = [&](int stage, int k_tile) {
        // A: [M x K] row-major
        const half* tileA_g = A + (row_start * K + k_tile * TILE_K);
        // B: [K x N] row-major
        const half* tileB_g = B + (k_tile * TILE_K * N + col_start);

        half* tileA_s = smemA + stage * stage_tile_elems;
        half* tileB_s = smemB + stage * stage_tile_elems;

        // 16x16 tile = 256 elements = 512 bytes
        // thread당 8 half (16B)씩 복사 -> 32 * 8 = 256 elements
        int idx = warp_lane * 8; // element index (0..255, step 8)
        if (idx < stage_tile_elems) {
            int r = idx / TILE_K;
            int c = idx % TILE_K;

            // A: row-major
            const half* srcA = tileA_g + r * K + c;
            half* dstA = tileA_s + r * TILE_K + c;
            cp_async_16b(dstA, srcA);

            // B: row-major
            const half* srcB = tileB_g + r * N + c;
            half* dstB = tileB_s + r * TILE_N + c;
            cp_async_16b(dstB, srcB);
        }
    };

    // ----------------------
    // 1) 먼저 STAGES 개 tile preload
    // ----------------------
    int preload_tiles = num_k_tiles < STAGES ? num_k_tiles : STAGES;
    for (int t = 0; t < preload_tiles; ++t) {
        cp_async_load_tile(t, t);
    }
    cp_async_commit_group();
    cp_async_wait_all();   // preload 완료 보장
    __syncthreads();

    // ----------------------
    // 2) K loop: multi-stage pipeline
    //    (개념상 ring-buffer로 stage를 돌려쓴다)
    // ----------------------
    for (int k_tile = 0; k_tile < num_k_tiles; ++k_tile) {
        int stage = k_tile % STAGES;

        // 현재 stage에 있는 tile 사용
        half* tileA_s = smemA + stage * stage_tile_elems;
        half* tileB_s = smemB + stage * stage_tile_elems;

        wmma::load_matrix_sync(a_frag, tileA_s, TILE_K);
        wmma::load_matrix_sync(b_frag, tileB_s, TILE_N);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

        // 다음에 이 stage에 올려둘 tile(k_tile + STAGES) prefetch
        int next_k_tile = k_tile + STAGES;
        if (next_k_tile < num_k_tiles) {
            cp_async_load_tile(stage, next_k_tile);
            cp_async_commit_group();
            // 보수적으로 wait + sync (패턴 확인용)
            cp_async_wait_all();
            __syncthreads();
        }
    }

    // ----------------------
    // 3) 결과 C에 저장
    // ----------------------
    float* tileC_g = C + row_start * N + col_start;
    wmma::store_matrix_sync(tileC_g, c_frag, N, wmma::mem_row_major);

#endif // __CUDA_ARCH__ >= 800
}

// -----------------------------
//  Host helper: matrix init
// -----------------------------
void init_matrix_half(half* h_ptr, int rows, int cols, float value) {
    for (int i = 0; i < rows * cols; ++i) {
        h_ptr[i] = __float2half(value);
    }
}

void check_C_range(const float* hC, int M, int N, float& minv, float& maxv) {
    minv = 1e30f;
    maxv = -1e30f;
    for (int i = 0; i < M * N; ++i) {
        float v = hC[i];
        if (v < minv) minv = v;
        if (v > maxv) maxv = v;
    }
}

// -----------------------------
//  main: 2 / 3 / 4 stage 비교
// -----------------------------
int main() {
    const int M = 128;
    const int N = 128;
    const int K = 128;

    printf("WMMA cp.async multi-stage pipeline test (M=%d, N=%d, K=%d)\n", M, N, K);

    size_t bytesA = sizeof(half) * M * K;
    size_t bytesB = sizeof(half) * K * N;
    size_t bytesC = sizeof(float) * M * N;

    half* hA = (half*)malloc(bytesA);
    half* hB = (half*)malloc(bytesB);
    float* hC = (float*)malloc(bytesC);

    init_matrix_half(hA, M, K, 1.0f);
    init_matrix_half(hB, K, N, 1.0f);

    half *dA, *dB;
    float* dC;
    CUDA_CHECK(cudaMalloc(&dA, bytesA));
    CUDA_CHECK(cudaMalloc(&dB, bytesB));
    CUDA_CHECK(cudaMalloc(&dC, bytesC));

    CUDA_CHECK(cudaMemcpy(dA, hA, bytesA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB, bytesB, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dC, 0, bytesC));

    dim3 blockDim(32, 1, 1); // warp 1개
    dim3 gridDim(N / TILE_N, M / TILE_M, 1); // warp tile당 16x16

    auto launch_and_time = [&](int stages) {
        int smem_elems = stages * TILE_M * TILE_K * 2; // A,B 두 세트
        size_t smem_bytes = smem_elems * sizeof(half);

        CUDA_CHECK(cudaMemset(dC, 0, bytesC));

        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        int iters = 200;

        CUDA_CHECK(cudaEventRecord(start));

        for (int i = 0; i < iters; ++i) {
            switch (stages) {
                case 2:
                    wmma_cp_async_multistage_kernel<2><<<gridDim, blockDim, smem_bytes>>>(
                        dA, dB, dC, M, N, K
                    );
                    break;
                case 3:
                    wmma_cp_async_multistage_kernel<3><<<gridDim, blockDim, smem_bytes>>>(
                        dA, dB, dC, M, N, K
                    );
                    break;
                case 4:
                    wmma_cp_async_multistage_kernel<4><<<gridDim, blockDim, smem_bytes>>>(
                        dA, dB, dC, M, N, K
                    );
                    break;
                default:
                    printf("Unsupported stages=%d\n", stages);
                    return;
            }
        }

        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= iters;  // 1회 평균

        double flops = 2.0 * (double)M * (double)N * (double)K;
        double tflops = (flops / (ms * 1.0e-3)) / 1.0e12;

        CUDA_CHECK(cudaMemcpy(hC, dC, bytesC, cudaMemcpyDeviceToHost));
        float minv, maxv;
        check_C_range(hC, M, N, minv, maxv);

        printf("[Stage=%d] time = %.3f ms, TFLOPs = %.3f, C range = [%.1f, %.1f]\n",
               stages, ms, tflops, minv, maxv);

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    };

    launch_and_time(2);
    launch_and_time(3);
    launch_and_time(4);

    free(hA);
    free(hB);
    free(hC);
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));

    return 0;
}

/*
nvcc -arch=sm_80 -O3 cp_async_multistage_test.cu -o cp_async_multistage_test.exe


ncu --kernel-name regex:.*wmma_cp_async_multistage_kernel.*     --metrics smsp__pipe_tensor_cycles_active,smsp__warp_issue_stalled_lg_throttle_per_warp_active.avg,dram__bytes_read.sum,lts__t_sectors_aperture_device_hit_rate.pct  --set full --launch-skip 0 --launch-count 1    ./cp_async_multistage_test.exe

*/