#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cstdio>
#include <vector>
#include <random>
#include <cmath>
#include <cfloat>

#ifndef CHECK_CUDA
#define CHECK_CUDA(call)                                                     \
    do {                                                                     \
        cudaError_t err = (call);                                            \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,    \
                    cudaGetErrorString(err));                                \
            std::exit(EXIT_FAILURE);                                         \
        }                                                                    \
    } while (0)
#endif

// -----------------------------
// CPU reference softmax
// -----------------------------
void softmax_ref(const std::vector<float>& x,
                 std::vector<float>& y,
                 int rows, int cols)
{
    for (int r = 0; r < rows; ++r) {
        const float* row_in  = &x[r * cols];
        float*       row_out = &y[r * cols];

        // 1) max
        float m = -FLT_MAX;
        for (int c = 0; c < cols; ++c) {
            m = std::max(m, row_in[c]);
        }

        // 2) exp + sum
        float sum = 0.0f;
        for (int c = 0; c < cols; ++c) {
            float e = std::exp(row_in[c] - m);
            row_out[c] = e;
            sum += e;
        }

        // 3) normalize
        float inv_sum = 1.0f / sum;
        for (int c = 0; c < cols; ++c) {
            row_out[c] *= inv_sum;
        }
    }
}

// -----------------------------
// Multi-kernel softmax (no warp shuffle)
// -----------------------------

// row-wise max: out_max[row] = max_j x[row, j]
__global__ void softmax_multi_max_kernel(const float* __restrict__ x,
                                         float* __restrict__ row_max,
                                         int rows, int cols)
{
    int row = blockIdx.x;
    if (row >= rows) return;

    extern __shared__ float smem[];
    int tid = threadIdx.x;

    float m = -FLT_MAX;
    // each thread processes multiple columns
    for (int c = tid; c < cols; c += blockDim.x) {
        float v = x[row * cols + c];
        m = fmaxf(m, v);
    }
    smem[tid] = m;
    __syncthreads();

    // block-wide reduction in shared memory
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            smem[tid] = fmaxf(smem[tid], smem[tid + offset]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        row_max[row] = smem[0];
    }
}

// tmp[row, j] = exp(x[row, j] - row_max[row])
__global__ void softmax_multi_exp_kernel(const float* __restrict__ x,
                                         const float* __restrict__ row_max,
                                         float* __restrict__ tmp,
                                         int rows, int cols)
{
    int row = blockIdx.x;
    if (row >= rows) return;

    int tid = threadIdx.x;
    float m = row_max[row];

    for (int c = tid; c < cols; c += blockDim.x) {
        int idx = row * cols + c;
        float v = x[idx];
        tmp[idx] = expf(v - m);
    }
}

// row_sum[row] = sum_j tmp[row, j]
__global__ void softmax_multi_sum_kernel(const float* __restrict__ tmp,
                                         float* __restrict__ row_sum,
                                         int rows, int cols)
{
    int row = blockIdx.x;
    if (row >= rows) return;

    extern __shared__ float smem[];
    int tid = threadIdx.x;

    float s = 0.0f;
    for (int c = tid; c < cols; c += blockDim.x) {
        s += tmp[row * cols + c];
    }
    smem[tid] = s;
    __syncthreads();

    // block-wide reduction in shared memory
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            smem[tid] += smem[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0) {
        row_sum[row] = smem[0];
    }
}

// y[row, j] = tmp[row, j] / row_sum[row]
__global__ void softmax_multi_norm_kernel(const float* __restrict__ tmp,
                                          const float* __restrict__ row_sum,
                                          float* __restrict__ y,
                                          int rows, int cols)
{
    int row = blockIdx.x;
    if (row >= rows) return;

    int tid = threadIdx.x;
    float s = row_sum[row];
    float inv_s = 1.0f / s;

    for (int c = tid; c < cols; c += blockDim.x) {
        int idx = row * cols + c;
        y[idx] = tmp[idx] * inv_s;
    }
}

// -----------------------------
// Fused softmax (warp-shuffle based reductions)
// -----------------------------

__inline__ __device__ float warp_reduce_max(float v) {
    // full warp mask
    unsigned mask = 0xffffffff;
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        float other = __shfl_down_sync(mask, v, offset);
        v = fmaxf(v, other);
    }
    return v;
}

__inline__ __device__ float warp_reduce_sum(float v) {
    unsigned mask = 0xffffffff;
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(mask, v, offset);
    }
    return v;
}

// Assumption: blockDim.x == warpSize (32), each block handles exactly one row.
__global__ void softmax_fused_kernel(const float* __restrict__ x,
                                     float* __restrict__ y,
                                     int rows, int cols)
{
    int row = blockIdx.x;
    if (row >= rows) return;

    int tid = threadIdx.x;
    int lane = tid;  // 0..31

    // 1) row-wise max
    float local_max = -FLT_MAX;
    for (int c = lane; c < cols; c += blockDim.x) {
        float v = x[row * cols + c];
        local_max = fmaxf(local_max, v);
    }
    float row_max = warp_reduce_max(local_max);
    // broadcast max to all threads in warp
    row_max = __shfl_sync(0xffffffff, row_max, 0);

    // 2) row-wise sum of exp(x - max)
    float local_sum = 0.0f;
    for (int c = lane; c < cols; c += blockDim.x) {
        float v = x[row * cols + c];
        float e = expf(v - row_max);
        local_sum += e;
    }
    float row_sum = warp_reduce_sum(local_sum);
    row_sum = __shfl_sync(0xffffffff, row_sum, 0);
    float inv_sum = 1.0f / row_sum;

    // 3) write softmax output
    for (int c = lane; c < cols; c += blockDim.x) {
        float v = x[row * cols + c];
        float e = expf(v - row_max);
        y[row * cols + c] = e * inv_sum;
    }
}

// -----------------------------
// Utility: max abs diff
// -----------------------------
float max_abs_diff(const std::vector<float>& a,
                   const std::vector<float>& b)
{
    float m = 0.0f;
    size_t n = a.size();
    for (size_t i = 0; i < n; ++i) {
        float d = std::fabs(a[i] - b[i]);
        if (d > m) m = d;
    }
    return m;
}

// -----------------------------
// Main
// -----------------------------
int main(int argc, char** argv)
{
    int rows  = 4096;
    int cols  = 1024;
    int iters = 100;

    if (argc >= 3) {
        rows = std::atoi(argv[1]);
        cols = std::atoi(argv[2]);
    }
    if (argc >= 4) {
        iters = std::atoi(argv[3]);
    }

    printf("Softmax size: rows=%d, cols=%d, iters=%d\n", rows, cols, iters);

    size_t num_elem = static_cast<size_t>(rows) * cols;

    // Host buffers
    std::vector<float> h_x(num_elem);
    std::vector<float> h_y_ref(num_elem);
    std::vector<float> h_y_multi(num_elem);
    std::vector<float> h_y_fused(num_elem);

    // Initialize input with random values
    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> dist(-5.0f, 5.0f);
    for (size_t i = 0; i < num_elem; ++i) {
        h_x[i] = dist(rng);
    }

    // CPU reference
    softmax_ref(h_x, h_y_ref, rows, cols);

    // Device buffers
    float *d_x = nullptr, *d_tmp = nullptr, *d_y_multi = nullptr, *d_y_fused = nullptr;
    float *d_row_max = nullptr, *d_row_sum = nullptr;

    CHECK_CUDA(cudaMalloc(&d_x,        num_elem * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_tmp,      num_elem * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_y_multi,  num_elem * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_y_fused,  num_elem * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_row_max,  rows * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_row_sum,  rows * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_x, h_x.data(),
                          num_elem * sizeof(float),
                          cudaMemcpyHostToDevice));

    // Launch config
    const int BLOCK_SIZE = 32; // 1 warp per row
    dim3 block(BLOCK_SIZE);
    dim3 grid(rows);

    size_t shmem_reduce = BLOCK_SIZE * sizeof(float);

    // -----------------------------
    // Correctness check (single run)
    // -----------------------------

    // Multi-kernel softmax
    softmax_multi_max_kernel<<<grid, block, shmem_reduce>>>(d_x, d_row_max, rows, cols);
    softmax_multi_exp_kernel<<<grid, block>>>(d_x, d_row_max, d_tmp, rows, cols);
    softmax_multi_sum_kernel<<<grid, block, shmem_reduce>>>(d_tmp, d_row_sum, rows, cols);
    softmax_multi_norm_kernel<<<grid, block>>>(d_tmp, d_row_sum, d_y_multi, rows, cols);
    CHECK_CUDA(cudaGetLastError());

    // Fused softmax
    softmax_fused_kernel<<<grid, block>>>(d_x, d_y_fused, rows, cols);
    CHECK_CUDA(cudaGetLastError());

    CHECK_CUDA(cudaMemcpy(h_y_multi.data(), d_y_multi,
                          num_elem * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_y_fused.data(), d_y_fused,
                          num_elem * sizeof(float),
                          cudaMemcpyDeviceToHost));

    float diff_multi = max_abs_diff(h_y_ref, h_y_multi);
    float diff_fused = max_abs_diff(h_y_ref, h_y_fused);
    printf("Max abs diff (ref vs multi-kernel) = %.6e\n", diff_multi);
    printf("Max abs diff (ref vs fused)        = %.6e\n", diff_fused);

    // -----------------------------
    // Timing: multi-kernel vs fused
    // -----------------------------

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // Warmup
    for (int i = 0; i < 10; ++i) {
        softmax_multi_max_kernel<<<grid, block, shmem_reduce>>>(d_x, d_row_max, rows, cols);
        softmax_multi_exp_kernel<<<grid, block>>>(d_x, d_row_max, d_tmp, rows, cols);
        softmax_multi_sum_kernel<<<grid, block, shmem_reduce>>>(d_tmp, d_row_sum, rows, cols);
        softmax_multi_norm_kernel<<<grid, block>>>(d_tmp, d_row_sum, d_y_multi, rows, cols);
        softmax_fused_kernel<<<grid, block>>>(d_x, d_y_fused, rows, cols);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    // Multi-kernel timing
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) {
        softmax_multi_max_kernel<<<grid, block, shmem_reduce>>>(d_x, d_row_max, rows, cols);
        softmax_multi_exp_kernel<<<grid, block>>>(d_x, d_row_max, d_tmp, rows, cols);
        softmax_multi_sum_kernel<<<grid, block, shmem_reduce>>>(d_tmp, d_row_sum, rows, cols);
        softmax_multi_norm_kernel<<<grid, block>>>(d_tmp, d_row_sum, d_y_multi, rows, cols);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms_multi = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms_multi, start, stop));
    ms_multi /= iters;

    // Fused timing
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) {
        softmax_fused_kernel<<<grid, block>>>(d_x, d_y_fused, rows, cols);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms_fused = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms_fused, start, stop));
    ms_fused /= iters;

    printf("\n=== Timing (avg over %d iters) ===\n", iters);
    printf("Multi-kernel softmax : %.4f ms\n", ms_multi);
    printf("Fused softmax        : %.4f ms\n", ms_fused);
    printf("Speedup (multi / fused): %.2fx\n", ms_multi / ms_fused);

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_x);
    cudaFree(d_tmp);
    cudaFree(d_y_multi);
    cudaFree(d_y_fused);
    cudaFree(d_row_max);
    cudaFree(d_row_sum);

    return 0;
}
/*
nvcc -O3 -arch=sm_86 softmax_fusion_test.cu -o softmax_fusion_test.exe

# row-wise max
ncu --kernel-name regex:softmax_multi_max_kernel.*    --metrics dram__bytes_read.sum,dram__bytes_write.sum,smsp__inst_executed_op_shuffle.sum     --launch-skip 5 --launch-count 1 --set full     .\softmax_fusion_test.exe 4096 1024 100

# exp(x - max)
ncu --kernel-name regex:softmax_multi_exp_kernel.*    --metrics dram__bytes_read.sum,dram__bytes_write.sum,smsp__inst_executed_op_shuffle.sum     --launch-skip 5 --launch-count 1 --set full     .\softmax_fusion_test.exe 4096 1024 100

# sum
ncu --kernel-name regex:softmax_multi_sum_kernel.*    --metrics dram__bytes_read.sum,dram__bytes_write.sum,smsp__inst_executed_op_shuffle.sum     --launch-skip 5 --launch-count 1 --set full     .\softmax_fusion_test.exe 4096 1024 100

# normalize
ncu --kernel-name regex:softmax_multi_norm_kernel.*    --metrics dram__bytes_read.sum,dram__bytes_write.sum,smsp__inst_executed_op_shuffle.sum     --launch-skip 5 --launch-count 1 --set full     .\softmax_fusion_test.exe 4096 1024 100


*/