#include <cstdio>
#include <cuda_runtime.h>

constexpr int N       = 1 << 24;   // 16M elements
constexpr int BLOCKS  = 80;
constexpr int THREADS = 256;
constexpr int ITERS   = 256;

// 완전 의존 체인: 다음 FMA가 항상 이전 결과에 의존
__global__
void dep_chain_kernel(float* __restrict__ out,
                      const float* __restrict__ in,
                      int n)
{
    int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    float acc = 0.0f;

    for (int i = tid; i < n; i += stride) {
        float v = in[i];

        #pragma unroll
        for (int k = 0; k < ITERS; ++k) {
            // 강한 데이터 의존성: 항상 acc에 의존
            acc = fmaf(acc, 1.0000001f, v);
        }
    }

    if (tid < n)
        out[tid] = acc;
}

// ILP 제공: 두 개의 독립 레지스터에 번갈아 FMA
__global__
void ilp_kernel(float* __restrict__ out,
                const float* __restrict__ in,
                int n)
{
    int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    float acc0 = 0.0f;
    float acc1 = 1.0f;

    for (int i = tid; i < n; i += stride) {
        float v = in[i];

        #pragma unroll
        for (int k = 0; k < ITERS; ++k) {
            // acc0, acc1 서로 독립 → 스케줄러가 사이사이 끼워넣기 가능
            acc0 = fmaf(acc0, 1.0000001f, v + 1.0f);
            acc1 = fmaf(acc1, 0.9999999f, v - 1.0f);
        }
    }

    if (tid < n)
        out[tid] = acc0 + acc1;
}

float run_kernel(const char* name,
                 void (*kernel)(float*, const float*, int),
                 float* d_out,
                 const float* d_in,
                 int n)
{
    dim3 block(THREADS);
    dim3 grid(BLOCKS);

    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    printf("[%s]\n", name);

    cudaEventRecord(start);
    kernel<<<grid, block>>>(d_out, d_in, n);
    cudaEventRecord(stop);

    cudaDeviceSynchronize();

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    // 읽기 기준 대략적인 BW (bytes: N * sizeof(float))
    double bytes = static_cast<double>(n) * sizeof(float);
    double gb    = bytes / (1024.0 * 1024.0 * 1024.0);
    double bw    = gb / (ms * 1e-3);  // GB/s

    printf("  Time = %.3f ms\n", ms);
    printf("  BW   = %.2f GB/s (read-only approx)\n\n", bw);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms;
}

int main()
{
    printf("== Issue & Instruction Scheduling Test ==\n");

    float* d_in  = nullptr;
    float* d_out = nullptr;

    cudaMalloc(&d_in,  N * sizeof(float));
    cudaMalloc(&d_out, N * sizeof(float));

    // 간단 초기화
    float* h_tmp = new float[N];
    for (int i = 0; i < N; ++i) {
        h_tmp[i] = static_cast<float>(i % 1024) * 0.1f;
    }
    cudaMemcpy(d_in, h_tmp, N * sizeof(float), cudaMemcpyHostToDevice);
    delete[] h_tmp;

    // 1) 완전 의존 체인
    run_kernel("dep_chain_kernel", dep_chain_kernel, d_out, d_in, N);

    // 2) ILP 제공 커널
    run_kernel("ilp_kernel", ilp_kernel, d_out, d_in, N);

    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}

// 빌드 예시 (Ampere):
// nvcc -O3 -arch=sm_86 issue_scheduling_test.cu -o issue_scheduling_test.exe
//
// Nsight Compute 예시:
// ncu --set full --kernel-name regex:.*dep_chain_kernel.*  ./issue_scheduling_test.exe
// ncu --set full --kernel-name regex:.*ilp_kernel.*        ./issue_scheduling_test.exe
