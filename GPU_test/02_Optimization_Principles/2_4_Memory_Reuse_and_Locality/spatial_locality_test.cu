// spatial_locality_test.cu
#include <cstdio>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err__ = (call);                                             \
        if (err__ != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA Error %s (err=%d) at %s:%d\n",                \
                    cudaGetErrorString(err__), err__, __FILE__, __LINE__);      \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (0)

// ----------------------------------------
// Kernel 1: per-thread 연속 접근 (stride = 1)
//   tid 별로 연속된 구간을 쭉 훑음
// ----------------------------------------
__global__ void spatial_locality_contiguous_kernel(
    const float* __restrict__ in,
    float* __restrict__ out,
    int iters)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int base = tid * iters;  // thread마다 iters 길이 구간

    float acc = 0.0f;
#pragma unroll 4
    for (int i = 0; i < iters; ++i) {
        int idx = base + i;  // stride=1
        acc += in[idx];
    }
    out[tid] = acc;
}

// ----------------------------------------
// Kernel 2: per-thread stride 접근 (stride = STRIDE_ELEM)
//   tid 별로 멀리 떨어진 위치들만 건너뛰며 접근
// ----------------------------------------
__global__ void spatial_locality_strided_kernel(
    const float* __restrict__ in,
    float* __restrict__ out,
    int iters,
    int stride_elems)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // thread마다 시작점도 stride_elems 간격으로 띄워서,
    // 전체적으로 더 넓은 영역을 흩어 읽도록 설계
    int base = tid * iters * stride_elems;

    float acc = 0.0f;
#pragma unroll 4
    for (int i = 0; i < iters; ++i) {
        int idx = base + i * stride_elems;  // stride=K
        acc += in[idx];
    }
    out[tid] = acc;
}

int main()
{
    // 실험 파라미터
    const int THREADS_PER_BLOCK = 256;
    const int BLOCKS            = 80;
    const int TOTAL_THREADS     = THREADS_PER_BLOCK * BLOCKS;

    const int ITERS        = 1024;  // per-thread 반복 횟수
    const int STRIDE_ELEM  = 32;    // K 역할 (stride = K)

    // 연속 접근 버전에서 필요한 원소 수
    const size_t CONTIG_TOTAL_ELEMS =
        static_cast<size_t>(TOTAL_THREADS) * ITERS;

    // stride 버전에서 필요한 원소 수
    const size_t STRIDED_TOTAL_ELEMS =
        static_cast<size_t>(TOTAL_THREADS) * ITERS * STRIDE_ELEM;

    // 통일된 버퍼 하나에 최대 크기 기준으로 할당 (그냥 넉넉히)
    const size_t TOTAL_ELEMS =
        STRIDED_TOTAL_ELEMS > CONTIG_TOTAL_ELEMS
            ? STRIDED_TOTAL_ELEMS
            : CONTIG_TOTAL_ELEMS;

    printf("=== Spatial Locality Test 1: contiguous vs stride access ===\n");
    printf("THREADS_PER_BLOCK = %d, BLOCKS = %d, TOTAL_THREADS = %d\n",
           THREADS_PER_BLOCK, BLOCKS, TOTAL_THREADS);
    printf("ITERS = %d, STRIDE_ELEM = %d\n", ITERS, STRIDE_ELEM);
    printf("CONTIG_TOTAL_ELEMS  = %zu\n", CONTIG_TOTAL_ELEMS);
    printf("STRIDED_TOTAL_ELEMS = %zu\n", STRIDED_TOTAL_ELEMS);
    printf("ALLOC TOTAL_ELEMS   = %zu\n\n", TOTAL_ELEMS);

    // 메모리 할당
    float* d_in  = nullptr;
    float* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in,  TOTAL_ELEMS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, TOTAL_THREADS * sizeof(float)));

    // 간단 초기화
    {
        float* h_tmp = new float[TOTAL_ELEMS];
        for (size_t i = 0; i < TOTAL_ELEMS; ++i) {
            h_tmp[i] = static_cast<float>(i & 0xFF) * 0.001f;
        }
        CUDA_CHECK(cudaMemcpy(d_in, h_tmp,
                              TOTAL_ELEMS * sizeof(float),
                              cudaMemcpyHostToDevice));
        delete[] h_tmp;
    }

    dim3 block(THREADS_PER_BLOCK);
    dim3 grid(BLOCKS);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // ---------------------------
    // 1) 연속 접근 커널
    // ---------------------------
    printf("[contiguous] warm-up + timing\n");
    spatial_locality_contiguous_kernel<<<grid, block>>>(d_in, d_out, ITERS);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    spatial_locality_contiguous_kernel<<<grid, block>>>(d_in, d_out, ITERS);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms_contig = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms_contig, start, stop));

    // 결과 샘플
    float sample_contig = 0.0f;
    CUDA_CHECK(cudaMemcpy(&sample_contig, d_out,
                          sizeof(float), cudaMemcpyDeviceToHost));

    printf("[contiguous] time = %.3f ms, sample out[0] = %f\n\n",
           ms_contig, sample_contig);

    // ---------------------------
    // 2) stride 접근 커널
    // ---------------------------
    printf("[strided] warm-up + timing\n");
    spatial_locality_strided_kernel<<<grid, block>>>(
        d_in, d_out, ITERS, STRIDE_ELEM);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    spatial_locality_strided_kernel<<<grid, block>>>(
        d_in, d_out, ITERS, STRIDE_ELEM);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms_strided = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms_strided, start, stop));

    float sample_strided = 0.0f;
    CUDA_CHECK(cudaMemcpy(&sample_strided, d_out,
                          sizeof(float), cudaMemcpyDeviceToHost));

    printf("[strided]    time = %.3f ms, sample out[0] = %f\n\n",
           ms_strided, sample_strided);

    printf("=== Summary ===\n");
    printf("contiguous: %.3f ms\n", ms_contig);
    printf("strided   : %.3f ms\n", ms_strided);
    if (ms_strided > 0.0f) {
        printf("speed ratio (strided / contiguous) = %.2fx\n",
               ms_strided / ms_contig);
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));

    return 0;
}
/*
nvcc -O3 -arch=sm_86 spatial_locality_test.cu -o spatial_locality_test.exe

ncu --kernel-name regex:spatial_locality_contiguous_kernel.*    --metrics dram__bytes_read.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum     ./spatial_locality_test.exe

ncu --kernel-name regex:spatial_locality_strided_kernel.*     --metrics dram__bytes_read.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum     ./spatial_locality_test.exe

*/