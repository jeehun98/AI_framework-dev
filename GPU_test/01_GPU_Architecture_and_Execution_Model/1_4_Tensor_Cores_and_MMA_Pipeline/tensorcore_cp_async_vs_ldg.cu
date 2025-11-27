// tensorcore_cp_async_vs_ldg.cu
//
// 1.4.4 Test — cp.async + double-buffering vs naive ld.global
//
// 동일한 "K 방향 타일 루프" 구조에서
//   - naive_ldg_kernel : ld.global + __syncthreads()
//   - cp_async_kernel  : cp.async + double-buffer + wait_group
// 의 차이를 Nsight Compute 로 비교하는 마이크로 벤치마크.
//
// RTX 30 (SM 8.6) 기준으로 작성 (cp.async 지원)

#include <cstdio>
#include <cuda_runtime.h>

#ifndef CHECK_CUDA
#define CHECK_CUDA(call)                                                      \
    do {                                                                      \
        cudaError_t err__ = (call);                                           \
        if (err__ != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA Error %s:%d: %s\n",                         \
                    __FILE__, __LINE__, cudaGetErrorString(err__));           \
            std::exit(EXIT_FAILURE);                                          \
        }                                                                     \
    } while (0)
#endif

// =====================================
// 커널 설정
// =====================================

static constexpr int BLOCK_SIZE   = 128;   // threads per block
static constexpr int GRID_SIZE    = 80;    // blocks
static constexpr int NUM_TILES    = 2048;  // thread 당 타일 개수 (K 반복 횟수)

// 전체 원소 수 = 모든 thread 가 처리하는 원소 수
// NUM_ELEMS = BLOCK_SIZE * GRID_SIZE * NUM_TILES
static constexpr int NUM_ELEMS = BLOCK_SIZE * GRID_SIZE * NUM_TILES;

// shared double-buffer 에 사용할 사이즈 (float 기준)
static constexpr int SMEM_FLOATS_PER_BLOCK = BLOCK_SIZE * 2; // double-buffer

// =====================================
// naive kernel: ld.global + __syncthreads()
// =====================================

__global__ void naive_ldg_kernel(float* __restrict__ out,
                                 const float* __restrict__ in,
                                 int num_elems)
{
    extern __shared__ float smem[]; // BLOCK_SIZE floats 사용
    float* tile = smem;

    int tid_in_block = threadIdx.x;
    int global_tid   = blockIdx.x * blockDim.x + tid_in_block;
    int stride       = blockDim.x * gridDim.x;

    // 각 thread 는 NUM_TILES 개의 원소를 stride 간격으로 담당
    float acc = 0.0f;

    // precondition: num_elems == BLOCK_SIZE * GRID_SIZE * NUM_TILES
    // 따라서 boundary 체크는 생략 (실제 코드에선 넣는 게 안전)
#pragma unroll 1
    for (int tile_idx = 0; tile_idx < NUM_TILES; ++tile_idx) {
        int idx = global_tid + tile_idx * stride;

        // 1) global -> shared (ld.global + __syncthreads())
        tile[tid_in_block] = in[idx];
        __syncthreads();

        // 2) shared 에서 읽어서 dummy compute (GEMM 의 inner FMA 비슷하게)
        float v = tile[tid_in_block];

#pragma unroll 8
        for (int k = 0; k < 32; ++k) {
            v = v * 1.000001f + 0.000001f;
        }

        acc += v;
        __syncthreads();
    }

    // 각 thread 당 하나의 결과만 기록
    out[global_tid] = acc;
}

// =====================================
// cp.async kernel: double-buffering
// =====================================

#if __CUDA_ARCH__ >= 800
// cp.async wrapper (32-bit shared address, 64-bit global address)
static __device__ __forceinline__
void cp_async_ca_shared_global(void* smem_ptr, const void* global_ptr, int bytes)
{
    // shared pointer 는 32-bit 주소로 변환해야 함
    unsigned smem_addr = static_cast<unsigned>(__cvta_generic_to_shared(smem_ptr));

    // global pointer 는 64-bit 그대로 사용
    asm volatile(
        "cp.async.ca.shared.global [%0], [%1], %2;\n" :: 
        "r"(smem_addr), "l"(global_ptr), "n"(sizeof(float))
    );
}
#endif
__global__ void cp_async_kernel(float* __restrict__ out,
                                const float* __restrict__ in,
                                int num_elems)
{
    extern __shared__ float smem[];
    float* buf0 = smem;
    float* buf1 = smem + blockDim.x; // double-buffer

    int tid_in_block = threadIdx.x;
    int global_tid   = blockIdx.x * blockDim.x + tid_in_block;
    int stride       = blockDim.x * gridDim.x;

    float acc = 0.0f;

#if __CUDA_ARCH__ < 800
    // fallback: 그냥 naive 버전
    for (int tile_idx = 0; tile_idx < NUM_TILES; ++tile_idx) {
        int idx = global_tid + tile_idx * stride;
        buf0[tid_in_block] = in[idx];
        __syncthreads();

        float v = buf0[tid_in_block];
#pragma unroll 8
        for (int k = 0; k < 32; ++k) {
            v = v * 1.000001f + 0.000001f;
        }
        acc += v;
        __syncthreads();
    }
    out[global_tid] = acc;
#else
    // ============================
    // 1) warm-up: tile 0, 1 prefetch
    // ============================
    {
        int idx0 = global_tid + 0 * stride;
        cp_async_ca_shared_global(&buf0[tid_in_block], in + idx0, sizeof(float));
        asm volatile("cp.async.commit_group;\n" ::);

        if (NUM_TILES > 1) {
            int idx1 = global_tid + 1 * stride;
            cp_async_ca_shared_global(&buf1[tid_in_block], in + idx1, sizeof(float));
            asm volatile("cp.async.commit_group;\n" ::);
        }
    }

    // ============================
    // 2) 타일 루프: 2-stage pipeline
    // ============================
    for (int tile = 0; tile < NUM_TILES; ++tile) {
        // (a) 현재 타일(t) 데이터는 준비된 상태여야 함
        //     - tile 0/1 은 warm-up 에서 미리 올림
        //     - tile 2부터는 이전 iteration에서 prefetch
        asm volatile("cp.async.wait_group 1;\n" ::);
        __syncthreads();

        // cur / next 버퍼 선택
        float* cur_buf = (tile & 1) ? buf1 : buf0;
        float* nxt_buf = (tile & 1) ? buf0 : buf1;

        // (b) cur_buf 에서 compute 수행 (GEMM inner loop 비슷한 더미 연산)
        float v = cur_buf[tid_in_block];

#pragma unroll 8
        for (int k = 0; k < 32; ++k) {
            v = v * 1.000001f + 0.000001f;
        }

        acc += v;
        __syncthreads();

        // (c) 다음 다음 타일(t+2)을 재사용 버퍼(nxt_buf)에 prefetch
        //     → 이 로드가 *다음 iteration의 compute* 와 겹치게 됨
        int preload_tile = tile + 2;
        if (preload_tile < NUM_TILES) {
            int preload_idx = global_tid + preload_tile * stride;
            cp_async_ca_shared_global(&nxt_buf[tid_in_block],
                                      in + preload_idx,
                                      sizeof(float));
            asm volatile("cp.async.commit_group;\n" ::);
        }
    }

    out[global_tid] = acc;
#endif
}

// =====================================
// 메인: 실행 + 타이밍
// =====================================

int main()
{
    printf("== 1.4.4 Test — cp.async + double-buffering vs naive ld.global ==\n");

    // 장치 정보
    cudaDeviceProp prop{};
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    printf("Device 0: %s (SM %d.%d)\n",
           prop.name, prop.major, prop.minor);

    if (prop.major < 8) {
        printf("WARNING: cp.async is only supported on SM 8.0+. "
               "This test is designed for Ampere or newer GPUs.\n");
    }

    // 호스트 메모리 할당
    float* h_in  = nullptr;
    float* h_out = nullptr;

    size_t bytes = static_cast<size_t>(NUM_ELEMS) * sizeof(float);

    h_in  = (float*)malloc(bytes);
    h_out = (float*)malloc(bytes);
    if (!h_in || !h_out) {
        fprintf(stderr, "Host malloc failed\n");
        return EXIT_FAILURE;
    }

    // 입력 초기화
    for (int i = 0; i < NUM_ELEMS; ++i) {
        h_in[i] = 1.0f; // 단순 값 (acc 결과는 NUM_TILES * some_constant 정도)
    }

    // 디바이스 메모리
    float *d_in = nullptr, *d_out = nullptr;
    CHECK_CUDA(cudaMalloc(&d_in, bytes));
    CHECK_CUDA(cudaMalloc(&d_out, bytes));

    CHECK_CUDA(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    dim3 block(BLOCK_SIZE);
    dim3 grid(GRID_SIZE);

    size_t smem_naive   = BLOCK_SIZE * sizeof(float);         // single buffer
    size_t smem_cpasync = BLOCK_SIZE * 2 * sizeof(float);     // double buffer

    // 이벤트 생성
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // -------------------------------------
    // 1) naive_ldg_kernel
    // -------------------------------------
    printf("\n[naive_ldg_kernel]\n");

    // warm-up
    naive_ldg_kernel<<<grid, block, smem_naive>>>(d_out, d_in, NUM_ELEMS);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaEventRecord(start));
    naive_ldg_kernel<<<grid, block, smem_naive>>>(d_out, d_in, NUM_ELEMS);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms_naive = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms_naive, start, stop));

    double gb_naive = (double)bytes / 1e9; // read bytes only (approx)
    double bw_naive = gb_naive / (ms_naive * 1e-3);

    printf("  Time   = %.3f ms\n", ms_naive);
    printf("  BW     = %.2f GB/s (read-only approx)\n", bw_naive);

    CHECK_CUDA(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));
    printf("  Sample out[0] = %f\n", h_out[0]);

    // -------------------------------------
    // 2) cp_async_kernel
    // -------------------------------------
    printf("\n[cp_async_kernel]\n");

    // warm-up
    cp_async_kernel<<<grid, block, smem_cpasync>>>(d_out, d_in, NUM_ELEMS);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaEventRecord(start));
    cp_async_kernel<<<grid, block, smem_cpasync>>>(d_out, d_in, NUM_ELEMS);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms_cp = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms_cp, start, stop));

    double gb_cp = (double)bytes / 1e9; // read bytes only (approx)
    double bw_cp = gb_cp / (ms_cp * 1e-3);

    printf("  Time   = %.3f ms\n", ms_cp);
    printf("  BW     = %.2f GB/s (read-only approx)\n", bw_cp);

    CHECK_CUDA(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));
    printf("  Sample out[0] = %f\n", h_out[0]);

    // 정리
    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_out));
    free(h_in);
    free(h_out);

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    printf("\nDone. Use Nsight Compute (ncu) to inspect memory throughput / SM busy / DRAM busy.\n");
    printf("Example:\n");
    printf("  ncu --set full --kernel-name regex:.*naive_ldg_kernel.*  ./tensorcore_cp_async_vs_ldg.exe\n");
    printf("  ncu --set full --kernel-name regex:.*cp_async_kernel.*   ./tensorcore_cp_async_vs_ldg.exe\n");

    return 0;
}


/*
# VS + nvcc 환경이면 대략 이런 느낌
nvcc -O3 -arch=sm_86 tensorcore_cp_async_vs_ldg.cu -o tensorcore_cp_async_vs_ldg.exe

./tensorcore_cp_async_vs_ldg.exe

ncu --set full --kernel-name regex:.*naive_ldg_kernel.*  ./tensorcore_cp_async_vs_ldg.exe
ncu --set full --kernel-name regex:.*cp_async_kernel.*   ./tensorcore_cp_async_vs_ldg.exe

*/