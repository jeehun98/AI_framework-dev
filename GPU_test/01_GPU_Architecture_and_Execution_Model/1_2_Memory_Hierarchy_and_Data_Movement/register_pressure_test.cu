#include <cstdio>
#include <cuda_runtime.h>

constexpr int N         = 1 << 20;  // 1M elements
constexpr int BLOCKS    = 80;
constexpr int THREADS   = 256;

// ------------------------------
// 1) Low register pressure kernel
// ------------------------------
__global__
void low_reg_kernel(float* __restrict__ out,
                    const float* __restrict__ in,
                    int n)
{
    int tid  = blockIdx.x * blockDim.x + threadIdx.x;
    int step = blockDim.x * gridDim.x;

    // 레지스터 거의 안 쓰는 단일 accumulator 패턴
    for (int i = tid; i < n; i += step) {
        float x = in[i];
        float acc = 0.f;

        // 일부러 약간의 연산만
        #pragma unroll 4
        for (int k = 0; k < 16; ++k) {
            acc = acc * 1.0001f + x * 0.9999f;
        }

        out[i] = acc;
    }
}

// ------------------------------
// 2) High register pressure kernel
//    - 많은 live variable + 큰 unroll
// ------------------------------
__global__
void high_reg_kernel(float* __restrict__ out,
                     const float* __restrict__ in,
                     int n)
{
    int tid  = blockIdx.x * blockDim.x + threadIdx.x;
    int step = blockDim.x * gridDim.x;

    for (int i = tid; i < n; i += step) {
        float x = in[i];

        // 레지스터 사용을 강제로 끌어올리기 위한 다수의 accumulator
        float a0  = x,  a1  = x * 1.1f,  a2  = x * 1.2f,  a3  = x * 1.3f;
        float a4  = x,  a5  = x * 1.4f,  a6  = x * 1.5f,  a7  = x * 1.6f;
        float a8  = x,  a9  = x * 1.7f,  a10 = x * 1.8f,  a11 = x * 1.9f;
        float a12 = x,  a13 = x * 2.0f,  a14 = x * 2.1f,  a15 = x * 2.2f;
        float a16 = x,  a17 = x * 2.3f,  a18 = x * 2.4f,  a19 = x * 2.5f;
        float a20 = x,  a21 = x * 2.6f,  a22 = x * 2.7f,  a23 = x * 2.8f;
        float a24 = x,  a25 = x * 2.9f,  a26 = x * 3.0f,  a27 = x * 3.1f;
        float a28 = x,  a29 = x * 3.2f,  a30 = x * 3.3f,  a31 = x * 3.4f;

        // 큰 unroll 로 live range 더 늘리기
        #pragma unroll 64
        for (int k = 0; k < 64; ++k) {
            a0  = a0  * 1.0001f + x;
            a1  = a1  * 1.0002f + x;
            a2  = a2  * 1.0003f + x;
            a3  = a3  * 1.0004f + x;
            a4  = a4  * 1.0005f + x;
            a5  = a5  * 1.0006f + x;
            a6  = a6  * 1.0007f + x;
            a7  = a7  * 1.0008f + x;
            a8  = a8  * 1.0009f + x;
            a9  = a9  * 1.0010f + x;
            a10 = a10 * 1.0011f + x;
            a11 = a11 * 1.0012f + x;
            a12 = a12 * 1.0013f + x;
            a13 = a13 * 1.0014f + x;
            a14 = a14 * 1.0015f + x;
            a15 = a15 * 1.0016f + x;
            a16 = a16 * 1.0017f + x;
            a17 = a17 * 1.0018f + x;
            a18 = a18 * 1.0019f + x;
            a19 = a19 * 1.0020f + x;
            a20 = a20 * 1.0021f + x;
            a21 = a21 * 1.0022f + x;
            a22 = a22 * 1.0023f + x;
            a23 = a23 * 1.0024f + x;
            a24 = a24 * 1.0025f + x;
            a25 = a25 * 1.0026f + x;
            a26 = a26 * 1.0027f + x;
            a27 = a27 * 1.0028f + x;
            a28 = a28 * 1.0029f + x;
            a29 = a29 * 1.0030f + x;
            a30 = a30 * 1.0031f + x;
            a31 = a31 * 1.0032f + x;
        }

        // 최종 결과를 하나로 모아서 store (최적화 방지용)
        float sum =
            a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7 +
            a8 + a9 + a10 + a11 + a12 + a13 + a14 + a15 +
            a16 + a17 + a18 + a19 + a20 + a21 + a22 + a23 +
            a24 + a25 + a26 + a27 + a28 + a29 + a30 + a31;

        out[i] = sum;
    }
}

int main()
{
    printf("== Register Pressure Test ==\n");

    float* d_in;
    float* d_out;
    cudaMalloc(&d_in,  N * sizeof(float));
    cudaMalloc(&d_out, N * sizeof(float));

    // 간단 초기화
    float* h_in = new float[N];
    for (int i = 0; i < N; ++i) h_in[i] = 1.0f;
    cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 grid(BLOCKS);
    dim3 block(THREADS);

    // 1) low_reg_kernel
    printf("[1] low_reg_kernel\n");
    low_reg_kernel<<<grid, block>>>(d_out, d_in, N);
    cudaDeviceSynchronize();

    // 2) high_reg_kernel
    printf("[2] high_reg_kernel\n");
    high_reg_kernel<<<grid, block>>>(d_out, d_in, N);
    cudaDeviceSynchronize();

    cudaFree(d_in);
    cudaFree(d_out);
    delete[] h_in;

    return 0;
}

// 빌드 예시 (sm_86 기준)
// nvcc -O3 -arch=sm_86 register_pressure_test.cu -o register_pressure_test.exe

// Nsight Compute 예시:
// ncu --set full --kernel-name regex:.*low_reg.*   ./register_pressure_test.exe
// ncu --set full --kernel-name regex:.*high_reg.*  ./register_pressure_test.exe
//
// 또는 레지스터 상한 강제로 바꿔보기:
// nvcc -O3 -arch=sm_86 --maxrregcount=32 register_pressure_test.cu -o reg32.exe
// nvcc -O3 -arch=sm_86 --maxrregcount=128 register_pressure_test.cu -o reg128.exe
