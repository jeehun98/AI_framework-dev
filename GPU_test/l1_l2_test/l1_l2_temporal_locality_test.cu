// l1_l2_temporal_locality_test.cu
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err__ = (call);                                        \
        if (err__ != cudaSuccess) {                                        \
            fprintf(stderr, "CUDA error: %s at %s:%d\n",                   \
                    cudaGetErrorString(err__), __FILE__, __LINE__);        \
            std::exit(1);                                                  \
        }                                                                  \
    } while (0)

// ------------------------------------------------------------
// 공통: 가벼운 FMA 형태의 "일" (연산량 맞추기용)
// ------------------------------------------------------------
__device__ __forceinline__
float do_work(float x) {
    // 너무 무거울 필요는 없고, 약간의 ALU 혼합용
    x = x * 1.000001f + 1.0f;
    x = x - 1.0f;
    return x;
}

// ------------------------------------------------------------
// (1) Streaming 패턴 커널
//
//  - 큰 배열 전체를 stride 크게 두고 훑으면서,
//    매번 거의 다른 cache line을 읽게 함
//  - temporal locality 거의 없음 → L1/L2 이득 적음
// ------------------------------------------------------------
__global__ void kernel_streaming(const float* __restrict__ in,
                                 float* __restrict__ out,
                                 int N,        // 총 원소 수
                                 int iters,    // 반복 횟수
                                 int stride)   // element 단위 stride
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= N) return;

    float acc = 0.0f;
    int idx = tid;

    for (int i = 0; i < iters; ++i) {
        acc += do_work(in[idx]);

        // 다음 접근 위치: stride만큼 멀리
        idx += stride;
        if (idx >= N) {
            idx -= N; // 간단 wrap-around
        }
    }

    out[tid] = acc;
}

// ------------------------------------------------------------
// (2) Hot reuse 패턴 커널
//
//  - 각 thread group이 작은 "hot region" 안에서만
//    데이터를 반복적으로 읽음
//  - 같은 cache line 재사용이 많이 일어나도록 설계
//    → L1/L2 hit 증가 기대
// ------------------------------------------------------------
__global__ void kernel_reuse(const float* __restrict__ in,
                             float* __restrict__ out,
                             int N,
                             int iters,
                             int hot_span)  // hot region 크기 (elements)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= N) return;

    // tid 기준으로 hot_span 단위 그룹을 묶어서,
    // 같은 그룹의 thread들이 비슷한 region 안에서만 놈
    int group_id      = tid / hot_span;
    int group_offset  = tid % hot_span;
    int base          = group_id * hot_span;

    // 전체 배열 범위를 넘지 않도록 보호
    if (base + hot_span > N) {
        // 끝쪽 일부 thread는 그냥 no-op에 가깝게 처리
        out[tid] = 0.0f;
        return;
    }

    float acc = 0.0f;
    int idx   = base + group_offset;

    for (int i = 0; i < iters; ++i) {
        // hot_span 안을 빙글빙글 도는 패턴
        int local = (group_offset + i) % hot_span;
        idx = base + local;

        acc += do_work(in[idx]);
    }

    out[tid] = acc;
}

// ------------------------------------------------------------
// 타이밍 유틸 (커널 템플릿 호출)
// ------------------------------------------------------------
template <typename Kernel>
float run_and_time(Kernel kernel,
                   const char* name,
                   float* d_out, const float* d_in,
                   int N, int iters, int pattern_param,
                   dim3 grid, dim3 block,
                   int repeat = 10)
{
    // warmup
    for (int i = 0; i < 3; ++i) {
        kernel<<<grid, block>>>(d_in, d_out, N, iters, pattern_param);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < repeat; ++i) {
        kernel<<<grid, block>>>(d_in, d_out, N, iters, pattern_param);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    ms /= repeat;

    printf("[%s] avg time: %.4f ms (iters=%d, N=%d, param=%d)\n",
           name, ms, iters, N, pattern_param);

    return ms;
}

// ------------------------------------------------------------
// main
// ------------------------------------------------------------
int main()
{
    // ===== 실험 파라미터 =====
    // 전체 배열 크기: 64 MB 정도 (float 4B * 16M = 64MB)
    const int N          = 1 << 24;    // 16,777,216
    const int block_size = 256;
    const int num_blocks = (N + block_size - 1) / block_size;

    const int iters      = 256;

    // streaming 접근에서 사용할 stride (elements 단위)
    // 4096 floats = 16KB → cache line들을 계속 바꿔가며 읽도록
    const int streaming_stride = 4096;

    // reuse 패턴에서 사용할 hot region 크기
    // 4096 floats = 16KB 정도 → L1/L2 안에 충분히 들어갈 수 있는 사이즈
    const int hot_span         = 4096;

    printf("N = %d (%.1f MB)\n", N, N * sizeof(float) / (1024.0f * 1024.0f));
    printf("block_size = %d, num_blocks = %d\n", block_size, num_blocks);
    printf("iters = %d\n", iters);
    printf("streaming_stride = %d (elements)\n", streaming_stride);
    printf("hot_span         = %d (elements)\n\n", hot_span);

    // ===== host 메모리 =====
    float* h_in  = (float*)malloc(N * sizeof(float));
    float* h_out = (float*)malloc(N * sizeof(float));
    if (!h_in || !h_out) {
        fprintf(stderr, "host malloc failed\n");
        return 1;
    }

    for (int i = 0; i < N; ++i) {
        h_in[i] = 0.001f * (float)i;
    }

    // ===== device 메모리 =====
    float *d_in = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in,  N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_in, h_in,
                          N * sizeof(float),
                          cudaMemcpyHostToDevice));

    dim3 block(block_size);
    dim3 grid(num_blocks);

    // ===== 타이밍 =====
    run_and_time(kernel_streaming, "streaming",
                 d_out, d_in,
                 N, iters, streaming_stride,
                 grid, block);

    run_and_time(kernel_reuse, "hot_reuse",
                 d_out, d_in,
                 N, iters, hot_span,
                 grid, block);

    // 결과 sanity check
    CUDA_CHECK(cudaMemcpy(h_out, d_out,
                          N * sizeof(float),
                          cudaMemcpyDeviceToHost));

    printf("sample outputs:\n");
    printf("  h_out[0]   = %f\n", h_out[0]);
    printf("  h_out[N/2] = %f\n", h_out[N/2]);
    printf("  h_out[N-1] = %f\n", h_out[N-1]);

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    free(h_in);
    free(h_out);

    CUDA_CHECK(cudaDeviceReset());
    return 0;
}

// 예시 빌드:
// nvcc -O3 -arch=sm_86 l1_l2_temporal_locality_test.cu -o l1l2_test
