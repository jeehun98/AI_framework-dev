#include <cstdio>
#include <cuda_runtime.h>

constexpr int N       = 1 << 24;   // 16M elements
constexpr int ITERS   = 16;        // 연산량 고정

// 공통 커널 본체: 메모리 + FMA 섞어서 약간 memory-bound 성격 부여
template<int BLOCK_SIZE>
__global__
void occupancy_kernel(float* __restrict__ out,
                      const float* __restrict__ in,
                      int n)
{
    int tid      = blockIdx.x * blockDim.x + threadIdx.x;
    int stride   = gridDim.x * blockDim.x;

    float acc = 0.f;

    // 전체 배열을 grid-stride 로프 형태로 순회
    for (int i = tid; i < n; i += stride) {
        float v = in[i];

        #pragma unroll
        for (int k = 0; k < ITERS; ++k) {
            acc = acc * 1.0000001f + v;   // FMA + 의존성
        }
    }

    // 범위 밖인 thread는 그냥 리턴 (out 오염 방지)
    if (tid < n)
        out[tid] = acc;
}

template<int BLOCK_SIZE>
float run_case(const char* label, float* d_out, const float* d_in, int n)
{
    // 너무 작은 grid 는 피하고, 대충 SM 개수보다 큰 정도로 세팅
    int grid = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (grid > 80) grid = 80;   // A100 / RTX 30x0 기준 적당한 값

    printf("[%s] BLOCK_SIZE = %d, grid = %d\n", label, BLOCK_SIZE, grid);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaDeviceSynchronize();
    cudaEventRecord(start);

    occupancy_kernel<BLOCK_SIZE><<<grid, BLOCK_SIZE>>>(d_out, d_in, n);

    cudaEventRecord(stop);
    cudaDeviceSynchronize();

    float ms = 0.f;
    cudaEventElapsedTime(&ms, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // 대략적인 effective BW 계산 (read + write)
    double bytes = double(n) * sizeof(float) * 2.0;
    double gb    = bytes / 1e9;
    double bw    = gb / (ms / 1e3);

    printf("  Time   = %.3f ms\n", ms);
    printf("  BW     = %.2f GB/s\n\n", bw);

    return ms;
}

int main()
{
    printf("== Occupancy vs Performance Test ==\n\n");

    float* d_in  = nullptr;
    float* d_out = nullptr;

    cudaMalloc(&d_in,  N * sizeof(float));
    cudaMalloc(&d_out, N * sizeof(float));

    // dummy 초기화
    float init = 1.0f;
    cudaMemset(d_out, 0, N * sizeof(float));
    cudaMemcpy(d_in, &init, sizeof(float), cudaMemcpyHostToDevice);

    // 세 가지 block 크기: occupancy / 워프 수를 바꾸기 위한 실험
    run_case< 64>("LOW  occupancy-ish", d_out, d_in, N);
    run_case<256>("MID  occupancy-ish", d_out, d_in, N);
    run_case<1024>("HIGH occupancy-ish", d_out, d_in, N);

    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}
