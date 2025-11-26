#include <cstdio>
#include <cuda_runtime.h>

constexpr int N               = 1 << 24;  // 요소 개수 (16M)
constexpr int THREADS_PER_BLOCK = 256;
constexpr int STRIDE          = 32;       // warp size와 동일한 stride로 non-coalesced 유도

// ------------------------------------------------------------
// Coalesced: thread i -> in[i]
// ------------------------------------------------------------
__global__
void coalesced_read_kernel(const float* __restrict__ in,
                           float* __restrict__ out,
                           int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float acc = 0.0f;

    // grid stride loop
    for (int i = tid; i < n; i += blockDim.x * gridDim.x) {
        acc += in[i];
    }
    // 최적화 방지용: 결과를 out에 기록
    if (tid < n) {
        out[tid] = acc;
    }
}

// ------------------------------------------------------------
// Strided: thread i -> in[i * STRIDE]
//  - warp 내 인접 thread들이 서로 다른 cache line / transaction을 유발
// ------------------------------------------------------------
__global__
void strided_read_kernel(const float* __restrict__ in,
                         float* __restrict__ out,
                         int n, int stride)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float acc = 0.0f;

    for (int i = tid; i < n; i += blockDim.x * gridDim.x) {
        int idx = (i * stride) % n;  // 범위 안으로 fold
        acc += in[idx];
    }
    if (tid < n) {
        out[tid] = acc;
    }
}

// ------------------------------------------------------------
// 타이밍 헬퍼
// ------------------------------------------------------------
float run_and_time(void (*kernel)(const float*, float*, int),
                   const char* name,
                   const float* d_in,
                   float* d_out,
                   int n,
                   dim3 grid,
                   dim3 block,
                   int warmup = 3,
                   int repeat = 10)
{
    // 함수 포인터를 그대로 쓸 수 없으니 템플릿 대신 래핑하거나,
    // 여기서는 lambda로 감싸서 호출하는 편이 깔끔하지만,
    // 단순화를 위해 커널별로 따로 호출하는 편이 안전하다.
    // → 아래에서 커널별로 직접 호출할 거라 이 함수는
    //   strided용 오버로드와 함께 참고용으로만 둔다.
    return 0.0f;
}

int main()
{
    printf("== Global Memory Coalescing Test ==\n");

    // -----------------------------
    // Host / Device 메모리 할당
    // -----------------------------
    size_t bytes = N * sizeof(float);

    float* h_in  = (float*)malloc(bytes);
    float* h_out = (float*)malloc(bytes);

    // 간단한 초기화
    for (int i = 0; i < N; ++i) {
        h_in[i] = 1.0f;
    }

    float* d_in  = nullptr;
    float* d_out = nullptr;
    cudaMalloc(&d_in,  bytes);
    cudaMalloc(&d_out, bytes);

    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);

    // -----------------------------
    // 런치 파라미터
    // -----------------------------
    int blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    // 너무 많은 블록은 줄여서 적당한 범위 내로
    blocks = min(blocks, 256);

    dim3 grid(blocks);
    dim3 block(THREADS_PER_BLOCK);

    // -----------------------------
    // cudaEvent로 타이밍 측정
    // -----------------------------
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 공통: warm-up
    for (int i = 0; i < 3; ++i) {
        coalesced_read_kernel<<<grid, block>>>(d_in, d_out, N);
        strided_read_kernel<<<grid, block>>>(d_in, d_out, N, STRIDE);
    }
    cudaDeviceSynchronize();

    // =============================
    // 1) Coalesced
    // =============================
    cudaEventRecord(start);
    for (int i = 0; i < 10; ++i) {
        coalesced_read_kernel<<<grid, block>>>(d_in, d_out, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms_coalesced = 0.0f;
    cudaEventElapsedTime(&ms_coalesced, start, stop);
    ms_coalesced /= 10.0f; // 평균

    double gbytes = (double)bytes / 1e9;
    double bw_coalesced = gbytes / (ms_coalesced / 1e3); // GB/s

    printf("[Coalesced]\n");
    printf("  N            = %d\n", N);
    printf("  Time (ms)    = %.3f\n", ms_coalesced);
    printf("  Bandwidth    = %.2f GB/s\n\n", bw_coalesced);

    // =============================
    // 2) Strided
    // =============================
    cudaEventRecord(start);
    for (int i = 0; i < 10; ++i) {
        strided_read_kernel<<<grid, block>>>(d_in, d_out, N, STRIDE);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms_strided = 0.0f;
    cudaEventElapsedTime(&ms_strided, start, stop);
    ms_strided /= 10.0f;

    double bw_strided = gbytes / (ms_strided / 1e3);

    printf("[Strided] (stride = %d)\n", STRIDE);
    printf("  N            = %d\n", N);
    printf("  Time (ms)    = %.3f\n", ms_strided);
    printf("  Bandwidth    = %.2f GB/s\n\n", bw_strided);

    // -----------------------------
    // 간단 검증 (optional)
    // -----------------------------
    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);
    printf("Sample output h_out[0] = %f\n", h_out[0]);

    // -----------------------------
    // 자원 해제
    // -----------------------------
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_in);
    cudaFree(d_out);
    free(h_in);
    free(h_out);

    return 0;
}

// 빌드 예시:
// nvcc -O3 -arch=sm_86 global_coalescing_test.cu -o global_coalescing_test.exe
//
// Nsight Compute 예시:
// ncu --set full --kernel-name regex:.*coalesced.* ./global_coalescing_test.exe
// ncu --set full --kernel-name regex:.*strided.*   ./global_coalescing_test.exe
