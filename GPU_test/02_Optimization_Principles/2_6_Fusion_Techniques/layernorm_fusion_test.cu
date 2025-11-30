#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>
#include <cmath>
#include <iostream>

#define CUDA_CHECK(cmd) do {                                   \
    cudaError_t e = (cmd);                                     \
    if (e != cudaSuccess) {                                   \
        fprintf(stderr, "CUDA error %s:%d: %s\n",             \
                __FILE__, __LINE__, cudaGetErrorString(e));    \
        std::exit(EXIT_FAILURE);                               \
    }                                                          \
} while (0)

constexpr int BLOCK_SIZE = 256;
constexpr float LN_EPS = 1e-5f;

// ----------------------------------------
// Two-pass LayerNorm
//   Kernel 1: mean per row
// ----------------------------------------
__global__ void layernorm_two_pass_mean_kernel(
    const float* __restrict__ x,
    float* __restrict__ mean,
    int rows, int cols)
{
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= rows) return;

    float sum = 0.0f;
    int base = row * cols;

    for (int j = tid; j < cols; j += blockDim.x) {
        sum += x[base + j];
    }

    __shared__ float sdata[BLOCK_SIZE];
    sdata[tid] = sum;
    __syncthreads();

    // block reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        mean[row] = sdata[0] / cols;
    }
}

// ----------------------------------------
// Two-pass LayerNorm
//   Kernel 2: var + norm + scale + shift
// ----------------------------------------
__global__ void layernorm_two_pass_var_norm_kernel(
    const float* __restrict__ x,
    const float* __restrict__ mean,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ y,
    int rows, int cols)
{
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= rows) return;

    int base = row * cols;
    float m = mean[row];

    // 1) variance reduction
    float sumsq = 0.0f;
    for (int j = tid; j < cols; j += blockDim.x) {
        float v = x[base + j] - m;
        sumsq += v * v;
    }

    __shared__ float sdata[BLOCK_SIZE];
    sdata[tid] = sumsq;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    float var;
    if (tid == 0) {
        var = sdata[0] / cols;
        sdata[0] = var;
    }
    __syncthreads();
    var = sdata[0];
    float inv_std = rsqrtf(var + LN_EPS);

    // 2) normalization + scale + shift
    for (int j = tid; j < cols; j += blockDim.x) {
        float v = x[base + j];
        float n = (v - m) * inv_std;
        float out = n * gamma[j] + beta[j];
        y[base + j] = out;
    }
}

// ----------------------------------------
// Fused LayerNorm: mean + var + norm + scale + shift
//   - block per row
//   - dynamic shared memory: row elements
// ----------------------------------------
__global__ void layernorm_fused_kernel(
    const float* __restrict__ x,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ y,
    int rows, int cols)
{
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= rows) return;

    extern __shared__ float shmem[]; // size = cols

    int base = row * cols;

    // 1) load row to shared + partial sum
    float sum = 0.0f;
    for (int j = tid; j < cols; j += blockDim.x) {
        float v = x[base + j];
        shmem[j] = v;
        sum += v;
    }

    __shared__ float stats[2]; // stats[0] = mean, stats[1] = var
    __shared__ float sdata[BLOCK_SIZE];

    sdata[tid] = sum;
    __syncthreads();

    // reduction for mean
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    float mean;
    if (tid == 0) {
        mean = sdata[0] / cols;
        stats[0] = mean;
    }
    __syncthreads();
    mean = stats[0];

    // 2) variance from shared
    float sumsq = 0.0f;
    for (int j = tid; j < cols; j += blockDim.x) {
        float v = shmem[j] - mean;
        sumsq += v * v;
    }

    sdata[tid] = sumsq;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    float var;
    if (tid == 0) {
        var = sdata[0] / cols;
        stats[1] = var;
    }
    __syncthreads();
    var = stats[1];
    float inv_std = rsqrtf(var + LN_EPS);

    // 3) norm + scale + shift, using shared row
    for (int j = tid; j < cols; j += blockDim.x) {
        float v = shmem[j];
        float n = (v - mean) * inv_std;
        float out = n * gamma[j] + beta[j];
        y[base + j] = out;
    }
}

// ----------------------------------------
// CPU reference LayerNorm
// ----------------------------------------
void layernorm_ref(
    const std::vector<float>& x,
    const std::vector<float>& gamma,
    const std::vector<float>& beta,
    std::vector<float>& y,
    int rows, int cols)
{
    for (int i = 0; i < rows; ++i) {
        const float* row_x = &x[i * cols];
        float* row_y = &y[i * cols];

        float mean = 0.0f;
        for (int j = 0; j < cols; ++j)
            mean += row_x[j];
        mean /= cols;

        float var = 0.0f;
        for (int j = 0; j < cols; ++j) {
            float v = row_x[j] - mean;
            var += v * v;
        }
        var /= cols;
        float inv_std = 1.0f / std::sqrt(var + LN_EPS);

        for (int j = 0; j < cols; ++j) {
            float n = (row_x[j] - mean) * inv_std;
            row_y[j] = n * gamma[j] + beta[j];
        }
    }
}

float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b)
{
    float maxd = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        float d = std::fabs(a[i] - b[i]);
        if (d > maxd) maxd = d;
    }
    return maxd;
}

int main(int argc, char** argv)
{
    int rows = 4096;
    int cols = 1024;
    int iters = 100;

    if (argc >= 3) {
        rows = std::atoi(argv[1]);
        cols = std::atoi(argv[2]);
    }
    if (argc >= 4) {
        iters = std::atoi(argv[3]);
    }

    std::cout << "LayerNorm size: rows=" << rows
              << ", cols=" << cols
              << ", iters=" << iters << "\n";

    size_t numel = static_cast<size_t>(rows) * cols;

    std::vector<float> h_x(numel);
    std::vector<float> h_gamma(cols);
    std::vector<float> h_beta(cols);
    std::vector<float> h_y_ref(numel);
    std::vector<float> h_y_two(numel);
    std::vector<float> h_y_fused(numel);

    // random init
    std::mt19937 rng(2025);
    std::uniform_real_distribution<float> dist_x(-1.0f, 1.0f);
    std::uniform_real_distribution<float> dist_g(0.5f, 1.5f);
    std::uniform_real_distribution<float> dist_b(-0.5f, 0.5f);

    for (size_t i = 0; i < numel; ++i)
        h_x[i] = dist_x(rng);
    for (int j = 0; j < cols; ++j) {
        h_gamma[j] = dist_g(rng);
        h_beta[j]  = dist_b(rng);
    }

    // CPU reference
    layernorm_ref(h_x, h_gamma, h_beta, h_y_ref, rows, cols);

    // device buffers
    float *d_x, *d_gamma, *d_beta;
    float *d_y_two, *d_y_fused;
    float *d_mean;
    CUDA_CHECK(cudaMalloc(&d_x,     numel * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gamma, cols   * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_beta,  cols   * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y_two,   numel * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y_fused, numel * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_mean,  rows  * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_x,     h_x.data(),     numel * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gamma, h_gamma.data(), cols   * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_beta,  h_beta.data(),  cols   * sizeof(float), cudaMemcpyHostToDevice));

    dim3 block(BLOCK_SIZE);
    dim3 grid(rows);

    // warmup
    for (int i = 0; i < 10; ++i) {
        layernorm_two_pass_mean_kernel<<<grid, block>>>(
            d_x, d_mean, rows, cols);
        layernorm_two_pass_var_norm_kernel<<<grid, block>>>(
            d_x, d_mean, d_gamma, d_beta, d_y_two, rows, cols);

        size_t shmem_bytes = cols * sizeof(float);
        layernorm_fused_kernel<<<grid, block, shmem_bytes>>>(
            d_x, d_gamma, d_beta, d_y_fused, rows, cols);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // correctness check (single run)
    layernorm_two_pass_mean_kernel<<<grid, block>>>(
        d_x, d_mean, rows, cols);
    layernorm_two_pass_var_norm_kernel<<<grid, block>>>(
        d_x, d_mean, d_gamma, d_beta, d_y_two, rows, cols);
    size_t shmem_bytes = cols * sizeof(float);
    layernorm_fused_kernel<<<grid, block, shmem_bytes>>>(
        d_x, d_gamma, d_beta, d_y_fused, rows, cols);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_y_two.data(),   d_y_two,   numel * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_y_fused.data(), d_y_fused, numel * sizeof(float), cudaMemcpyDeviceToHost));

    float diff_two   = max_abs_diff(h_y_ref, h_y_two);
    float diff_fused = max_abs_diff(h_y_ref, h_y_fused);

    std::cout << "Max abs diff (ref vs two-pass) = " << diff_two   << "\n";
    std::cout << "Max abs diff (ref vs fused)    = " << diff_fused << "\n\n";

    // timing: two-pass
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int it = 0; it < iters; ++it) {
        layernorm_two_pass_mean_kernel<<<grid, block>>>(
            d_x, d_mean, rows, cols);
        layernorm_two_pass_var_norm_kernel<<<grid, block>>>(
            d_x, d_mean, d_gamma, d_beta, d_y_two, rows, cols);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms_two = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms_two, start, stop));
    ms_two /= iters;

    // timing: fused
    CUDA_CHECK(cudaEventRecord(start));
    for (int it = 0; it < iters; ++it) {
        layernorm_fused_kernel<<<grid, block, shmem_bytes>>>(
            d_x, d_gamma, d_beta, d_y_fused, rows, cols);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms_fused = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms_fused, start, stop));
    ms_fused /= iters;

    std::cout << "=== Timing (avg over " << iters << " iters) ===\n";
    std::cout << "Two-pass LayerNorm : " << ms_two   << " ms\n";
    std::cout << "Fused  LayerNorm   : " << ms_fused << " ms\n";
    std::cout << "Speedup (two / fused): " << (ms_two / ms_fused) << "x\n";

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    cudaFree(d_x);
    cudaFree(d_gamma);
    cudaFree(d_beta);
    cudaFree(d_y_two);
    cudaFree(d_y_fused);
    cudaFree(d_mean);

    return 0;
}
/*
nvcc -O3 -arch=sm_86 layernorm_fusion_test.cu -o layernorm_fusion_test.exe                            

ncu --kernel-name regex:layernorm_two_pass_mean_kernel.*     --metrics dram__bytes_read.sum,dram__bytes_write.sum     --launch-skip 5 --launch-count 1     ./layernorm_fusion_test.exe 4096 1024 100

ncu --kernel-name regex:layernorm_two_pass_var_norm_kernel.*     --metrics dram__bytes_read.sum,dram__bytes_write.sum     --launch-skip 5 --launch-count 1     ./layernorm_fusion_test.exe 4096 1024 100

ncu --kernel-name regex:layernorm_fused_kernel.*     --metrics dram__bytes_read.sum,dram__bytes_write.sum     --launch-skip 5 --launch-count 1     ./layernorm_fusion_test.exe 4096 1024 100



*/