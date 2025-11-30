#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        cudaError_t err__ = (call);                                            \
        if (err__ != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA Error at %s:%d: %s\n",                       \
                    __FILE__, __LINE__, cudaGetErrorString(err__));            \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

template <typename T>
void init_random(std::vector<T>& v, float scale = 1.0f) {
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    for (auto& x : v) {
        x = static_cast<T>(dist(gen) * scale);
    }
}

// -------------------- GPU kernels --------------------

constexpr int TILE_M = 16;
constexpr int TILE_N = 16;

// GEMM only: C = A * B
// A: (M x K), B: (K x N), C: (M x N)
// row-major
__global__ void gemm_only_kernel(const float* __restrict__ A,
                                 const float* __restrict__ B,
                                 float* __restrict__ C,
                                 int M, int N, int K)
{
    int row = blockIdx.y * TILE_M + threadIdx.y;
    int col = blockIdx.x * TILE_N + threadIdx.x;

    if (row >= M || col >= N) return;

    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        float a = A[row * K + k];
        float b = B[k * N + col];
        acc += a * b;
    }
    C[row * N + col] = acc;
}

// Bias add: C = C + bias (broadcast over rows)
// bias: (N)
__global__ void bias_add_kernel(float* __restrict__ C,
                                const float* __restrict__ bias,
                                int M, int N)
{
    int row = blockIdx.y * TILE_M + threadIdx.y;
    int col = blockIdx.x * TILE_N + threadIdx.x;

    if (row >= M || col >= N) return;

    int idx = row * N + col;
    C[idx] += bias[col];
}

// ReLU: C = max(C, 0)
__global__ void relu_kernel(float* __restrict__ C,
                            int M, int N)
{
    int row = blockIdx.y * TILE_M + threadIdx.y;
    int col = blockIdx.x * TILE_N + threadIdx.x;

    if (row >= M || col >= N) return;

    int idx = row * N + col;
    float x = C[idx];
    C[idx] = x > 0.0f ? x : 0.0f;
}

// Fused GEMM + bias + ReLU
// out = relu(A * B + bias)
__global__ void gemm_bias_relu_fused_kernel(const float* __restrict__ A,
                                            const float* __restrict__ B,
                                            const float* __restrict__ bias,
                                            float* __restrict__ C,
                                            int M, int N, int K)
{
    int row = blockIdx.y * TILE_M + threadIdx.y;
    int col = blockIdx.x * TILE_N + threadIdx.x;

    if (row >= M || col >= N) return;

    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        float a = A[row * K + k];
        float b = B[k * N + col];
        acc += a * b;
    }

    // bias + ReLU epilogue
    acc += bias[col];
    acc = acc > 0.0f ? acc : 0.0f;

    C[row * N + col] = acc;
}

// -------------------- CPU reference --------------------

void gemm_bias_relu_ref(const std::vector<float>& A,
                        const std::vector<float>& B,
                        const std::vector<float>& bias,
                        std::vector<float>& out,
                        int M, int N, int K)
{
    std::fill(out.begin(), out.end(), 0.0f);

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k) {
                acc += A[i * K + k] * B[k * N + j];
            }
            acc += bias[j];
            acc = acc > 0.0f ? acc : 0.0f; // ReLU
            out[i * N + j] = acc;
        }
    }
}

// -------------------- main --------------------

int main(int argc, char** argv)
{
    int M = 1024;
    int N = 1024;
    int K = 1024;
    int iters = 100;

    if (argc >= 5) {
        M = std::atoi(argv[1]);
        N = std::atoi(argv[2]);
        K = std::atoi(argv[3]);
        iters = std::atoi(argv[4]);
    }

    printf("GEMM size: M=%d, N=%d, K=%d, iters=%d\n", M, N, K, iters);

    size_t size_A = static_cast<size_t>(M) * K;
    size_t size_B = static_cast<size_t>(K) * N;
    size_t size_C = static_cast<size_t>(M) * N;

    std::vector<float> h_A(size_A);
    std::vector<float> h_B(size_B);
    std::vector<float> h_bias(N);
    std::vector<float> h_out_ref(size_C);
    std::vector<float> h_out_sep(size_C);
    std::vector<float> h_out_fused(size_C);

    init_random(h_A);
    init_random(h_B);
    init_random(h_bias, 0.1f);

    float *d_A = nullptr, *d_B = nullptr, *d_bias = nullptr;
    float *d_C_sep = nullptr, *d_C_fused = nullptr;

    CHECK_CUDA(cudaMalloc(&d_A, size_A * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_B, size_B * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_bias, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_C_sep, size_C * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_C_fused, size_C * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), size_A * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B.data(), size_B * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_bias, h_bias.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    dim3 block(TILE_N, TILE_M);
    dim3 grid((N + TILE_N - 1) / TILE_N,
              (M + TILE_M - 1) / TILE_M);

    // -------------------- Correctness check --------------------
    gemm_bias_relu_ref(h_A, h_B, h_bias, h_out_ref, M, N, K);

    // Separate path: GEMM -> bias -> ReLU (single run for correctness)
    CHECK_CUDA(cudaMemset(d_C_sep, 0, size_C * sizeof(float)));
    gemm_only_kernel<<<grid, block>>>(d_A, d_B, d_C_sep, M, N, K);
    bias_add_kernel<<<grid, block>>>(d_C_sep, d_bias, M, N);
    relu_kernel<<<grid, block>>>(d_C_sep, M, N);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_out_sep.data(), d_C_sep, size_C * sizeof(float), cudaMemcpyDeviceToHost));

    // Fused path: GEMM + bias + ReLU (single run for correctness)
    CHECK_CUDA(cudaMemset(d_C_fused, 0, size_C * sizeof(float)));
    gemm_bias_relu_fused_kernel<<<grid, block>>>(d_A, d_B, d_bias, d_C_fused, M, N, K);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_out_fused.data(), d_C_fused, size_C * sizeof(float), cudaMemcpyDeviceToHost));

    auto compute_max_abs_diff = [&](const std::vector<float>& x,
                                    const std::vector<float>& y) {
        double max_diff = 0.0;
        for (size_t i = 0; i < x.size(); ++i) {
            double d = std::fabs(static_cast<double>(x[i]) - static_cast<double>(y[i]));
            if (d > max_diff) max_diff = d;
        }
        return max_diff;
    };

    double diff_sep   = compute_max_abs_diff(h_out_ref, h_out_sep);
    double diff_fused = compute_max_abs_diff(h_out_ref, h_out_fused);

    printf("Max abs diff (ref vs separate) = %.6e\n", diff_sep);
    printf("Max abs diff (ref vs fused)    = %.6e\n", diff_fused);

    // -------------------- Timing --------------------
    float ms_sep = 0.0f, ms_fused = 0.0f;

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // Warmup
    for (int i = 0; i < 5; ++i) {
        CHECK_CUDA(cudaMemset(d_C_sep, 0, size_C * sizeof(float)));
        gemm_only_kernel<<<grid, block>>>(d_A, d_B, d_C_sep, M, N, K);
        bias_add_kernel<<<grid, block>>>(d_C_sep, d_bias, M, N);
        relu_kernel<<<grid, block>>>(d_C_sep, M, N);

        CHECK_CUDA(cudaMemset(d_C_fused, 0, size_C * sizeof(float)));
        gemm_bias_relu_fused_kernel<<<grid, block>>>(d_A, d_B, d_bias, d_C_fused, M, N, K);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    // Separate path timing (GEMM + bias + ReLU)
    CHECK_CUDA(cudaEventRecord(start));
    for (int it = 0; it < iters; ++it) {
        gemm_only_kernel<<<grid, block>>>(d_A, d_B, d_C_sep, M, N, K);
        bias_add_kernel<<<grid, block>>>(d_C_sep, d_bias, M, N);
        relu_kernel<<<grid, block>>>(d_C_sep, M, N);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&ms_sep, start, stop));
    ms_sep /= iters;

    // Fused path timing (single kernel)
    CHECK_CUDA(cudaEventRecord(start));
    for (int it = 0; it < iters; ++it) {
        gemm_bias_relu_fused_kernel<<<grid, block>>>(d_A, d_B, d_bias, d_C_fused, M, N, K);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&ms_fused, start, stop));
    ms_fused /= iters;

    printf("\n=== Timing (avg over %d iters) ===\n", iters);
    printf("Separate (GEMM + bias + ReLU): %.4f ms\n", ms_sep);
    printf("Fused   (GEMM+bias+ReLU)     : %.4f ms\n", ms_fused);
    printf("Speedup (separate / fused)   : %.2fx\n", ms_sep / ms_fused);

    // FLOPs (GEMM only) for reference: 2*M*N*K
    double flops = 2.0 * (double)M * (double)N * (double)K;
    double t_sec_sep   = ms_sep   * 1e-3;
    double t_sec_fused = ms_fused * 1e-3;
    double tflops_sep  = flops / t_sec_sep   / 1e12;
    double tflops_fused= flops / t_sec_fused / 1e12;

    printf("\nApprox GEMM TFLOP/s (ignoring bias/ReLU FLOPs)\n");
    printf("Separate: %.3f TFLOP/s\n", tflops_sep);
    printf("Fused   : %.3f TFLOP/s\n", tflops_fused);

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_bias));
    CHECK_CUDA(cudaFree(d_C_sep));
    CHECK_CUDA(cudaFree(d_C_fused));

    return 0;
}
/*
nvcc -O3 -arch=sm_86   elementwise_fusion_test.cu -o elementwise_fusion_test.exe

ncu --kernel-name regex:gemm_only_kernel.*     --set full     --launch-skip 5 --launch-count 1     .\elementwise_fusion_test.exe 1024 1024 1024 100

ncu --kernel-name regex:gemm_bias_relu_fused_kernel.*     --set full     --launch-skip 5 --launch-count 1     .\elementwise_fusion_test.exe 1024 1024 1024 100


*/