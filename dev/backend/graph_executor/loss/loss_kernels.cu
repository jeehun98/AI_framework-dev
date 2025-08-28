// loss_kernels.cu (no cuBLAS dependency, with safe launch guards)

#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cfloat>
#include <vector>

#include "../ge/cuda_check.cuh"

// ======================================================================
// 공용 유틸: 리덕션 커널 / 런치 구성
// ======================================================================
__global__ void reduce_sum_kernel(const float* __restrict__ x,
                                  float* __restrict__ block_sums,
                                  int n)
{
    extern __shared__ float sdata[];
    int tid  = threadIdx.x;
    int gidx = blockIdx.x * blockDim.x + tid;

    float acc = 0.f;
    // grid-stride loop
    for (; gidx < n; gidx += blockDim.x * gridDim.x) {
        acc += x[gidx];
    }
    sdata[tid] = acc;
    __syncthreads();

    // block reduce
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }
    if (tid == 0) block_sums[blockIdx.x] = sdata[0];
}

static float reduce_sum_device(const float* d_vec, int n) {
    if (n <= 0) return 0.f;
    const int threads = 256;
    int blocks = (n + threads - 1) / threads;
    if (blocks < 1) blocks = 1;
    if (blocks > 65535) blocks = 65535;

    float* d_partials = nullptr;
    CUDA_CHECK(cudaMalloc(&d_partials, blocks * sizeof(float)));

    reduce_sum_kernel<<<blocks, threads, threads * sizeof(float)>>>(d_vec, d_partials, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> h_partials(blocks);
    CUDA_CHECK(cudaMemcpy(h_partials.data(), d_partials, blocks * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_partials));

    float sum = 0.f;
    for (int i = 0; i < blocks; ++i) sum += h_partials[i];
    return sum;
}

static inline void launch_conf(int n, dim3& grid, dim3& block, size_t& shmem) {
    const int threads = 256;
    int blocks = (n + threads - 1) / threads;
    if (blocks < 1) blocks = 1;
    if (blocks > 65535) blocks = 65535;
    block = dim3(threads, 1, 1);
    grid  = dim3(blocks, 1, 1);
    shmem = threads * sizeof(float);
}

// ======================================================================
/* MSE (forward / backward) */
// ======================================================================
__global__ void mse_loss_kernel(const float* __restrict__ y_true,
                                const float* __restrict__ y_pred,
                                float* __restrict__ loss_sum,
                                int size)
{
    extern __shared__ float cache[];
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float tmp = 0.0f;

    // grid-stride
    for (; tid < size; tid += blockDim.x * gridDim.x) {
        float d = y_true[tid] - y_pred[tid];
        tmp += d * d;
    }
    cache[threadIdx.x] = tmp;
    __syncthreads();

    // block reduce
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) cache[threadIdx.x] += cache[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicAdd(loss_sum, cache[0]);
}

__global__ void mse_loss_backward_kernel(const float* __restrict__ y_true,
                                         const float* __restrict__ y_pred,
                                         float* __restrict__ grad_out,
                                         int size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        grad_out[tid] = 2.0f * (y_pred[tid] - y_true[tid]) / (float)size; // mean
    }
}

float compute_mse_loss_cuda(const float* y_true, const float* y_pred, int size) {
    if (size <= 0) return 0.f;

    float *d_sum = nullptr;
    CUDA_CHECK(cudaMalloc(&d_sum, sizeof(float)));
    CUDA_CHECK(cudaMemset(d_sum, 0, sizeof(float)));

    dim3 grid, block; size_t shmem;
    launch_conf(size, grid, block, shmem);

    mse_loss_kernel<<<grid, block, shmem>>>(y_true, y_pred, d_sum, size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    float h_sum = 0.0f;
    CUDA_CHECK(cudaMemcpy(&h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_sum));

    return h_sum / (float)size;
}

void launch_mse_loss_backward(const float* y_true, const float* y_pred,
                              float* grad_out, int size, cudaStream_t stream)
{
    if (size <= 0) return;
    const int threads = 256;
    int blocks  = (size + threads - 1) / threads;
    if (blocks < 1) blocks = 1;
    mse_loss_backward_kernel<<<blocks, threads, 0, stream>>>(y_true, y_pred, grad_out, size);
    CUDA_CHECK(cudaGetLastError());
}

// ======================================================================
/* BCE (sigmoid probs) forward / backward) */
// ======================================================================
__global__ void bce_loss_kernel(const float* __restrict__ y_true,
                                const float* __restrict__ y_pred,
                                float* __restrict__ loss_sum,
                                int size)
{
    extern __shared__ float cache[];
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float tmp = 0.0f;
    const float eps = 1e-7f;

    for (; tid < size; tid += blockDim.x * gridDim.x) {
        float yt = y_true[tid];
        float yp = fminf(fmaxf(y_pred[tid], eps), 1.0f - eps);
        tmp += -(yt * logf(yp) + (1.0f - yt) * logf(1.0f - yp));
    }
    cache[threadIdx.x] = tmp;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) cache[threadIdx.x] += cache[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicAdd(loss_sum, cache[0]);
}

__global__ void bce_loss_backward_kernel(const float* __restrict__ y_true,
                                         const float* __restrict__ y_pred, // a
                                         float* __restrict__ grad_out,     // dL/da
                                         int size,
                                         int batch_size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= size) return;

    const float eps = 1e-7f;
    float a = fminf(fmaxf(y_pred[tid], eps), 1.f - eps);
    float y = y_true[tid];

    float denom = fmaxf(a * (1.f - a), eps); // σ'(z)=a(1-a)
    float dL_da = (a - y) / denom;
    float scale = (batch_size > 0) ? (1.f / (float)batch_size) : 1.f;
    grad_out[tid] = dL_da * scale;
}

float compute_bce_loss_cuda(const float* y_true, const float* y_pred, int size) {
    if (size <= 0) return 0.f;

    float *d_sum = nullptr;
    CUDA_CHECK(cudaMalloc(&d_sum, sizeof(float)));
    CUDA_CHECK(cudaMemset(d_sum, 0, sizeof(float)));

    dim3 grid, block; size_t shmem;
    launch_conf(size, grid, block, shmem);

    bce_loss_kernel<<<grid, block, shmem>>>(y_true, y_pred, d_sum, size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    float h_sum = 0.0f;
    CUDA_CHECK(cudaMemcpy(&h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_sum));

    return h_sum / (float)size; // mean
}

void launch_bce_loss_backward(const float* y_true, const float* y_pred,
                              float* grad_out, int size, int batch_size,
                              cudaStream_t stream)
{
    if (size <= 0) return;
    const int threads = 256;
    int blocks  = (size + threads - 1) / threads;
    if (blocks < 1) blocks = 1;
    bce_loss_backward_kernel<<<blocks, threads, 0, stream>>>(y_true, y_pred, grad_out, size, batch_size);
    CUDA_CHECK(cudaGetLastError());
}

// ======================================================================
/* CCE (softmax probs) forward / backward) */
// ======================================================================
__global__ void cce_per_sample_kernel(const float* __restrict__ y_true,
                                      const float* __restrict__ y_pred,
                                      float* __restrict__ per_sample, // [B]
                                      int B, int C)
{
    int b = blockIdx.x;
    if (b >= B) return;

    if (threadIdx.x == 0) {
        float acc = 0.f;
        for (int c = 0; c < C; ++c) {
            float y = y_true[b*C + c];
            if (y > 0.f) {
                float p = fminf(fmaxf(y_pred[b*C + c], 1e-7f), 1.f - 1e-7f);
                acc += -y * logf(p);
            }
        }
        per_sample[b] = acc;
    }
}

__global__ void cce_loss_backward_kernel(const float* __restrict__ y_true,
                                         const float* __restrict__ y_pred,
                                         float* __restrict__ grad_out, // dL/dY
                                         int B, int C)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x; // over B*C
    int N = B * C;
    if (i >= N) return;

    float y = y_true[i];
    if (y > 0.f) {
        float p = fmaxf(y_pred[i], 1e-7f);
        grad_out[i] = -(y / p) / (float)B;  // batch mean
    } else {
        grad_out[i] = 0.f;
    }
}

float compute_cce_loss_cuda(const float* y_true, const float* y_pred,
                            int batch_size, int num_classes)
{
    const int B = batch_size;
    const int C = num_classes;
    if (B <= 0 || C <= 0) return 0.f;

    float* d_per_sample = nullptr; // [B]
    CUDA_CHECK(cudaMalloc(&d_per_sample, B * sizeof(float)));

    int blocks = B;
    if (blocks < 1) blocks = 1;
    cce_per_sample_kernel<<<blocks, 1>>>(y_true, y_pred, d_per_sample, B, C);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    float sum = reduce_sum_device(d_per_sample, B);
    CUDA_CHECK(cudaFree(d_per_sample));
    return (B > 0) ? (sum / (float)B) : 0.f; // batch mean
}

void launch_cce_loss_backward(const float* y_true, const float* y_pred,
                              float* grad_out, int batch_size, int num_classes,
                              cudaStream_t stream)
{
    const int B = batch_size, C = num_classes;
    const int N = B * C;
    if (N <= 0) return;

    const int threads = 256;
    int blocks  = (N + threads - 1) / threads;
    if (blocks < 1) blocks = 1;
    cce_loss_backward_kernel<<<blocks, threads, 0, stream>>>(y_true, y_pred, grad_out, B, C);
    CUDA_CHECK(cudaGetLastError());
}

// ======================================================================
// softmax ⊗ cross-entropy fused backward: grad_z = (p - y)/B
// ======================================================================
__global__ void softmax_xent_fused_backward_kernel(const float* __restrict__ p,
                                                   const float* __restrict__ y,
                                                   float* __restrict__ grad_z,
                                                   int B, int C)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x; // over B*C
    int N = B * C;
    if (i >= N) return;
    grad_z[i] = (p[i] - y[i]) / (float)B;
}

void launch_softmax_xent_fused_backward(const float* y_prob,
                                        const float* y_true,
                                        float* grad_z,
                                        int batch_size, int num_classes,
                                        cudaStream_t stream)
{
    const int B = batch_size, C = num_classes;
    const int N = B * C;
    if (N <= 0) return;

    const int threads = 256;
    int blocks  = (N + threads - 1) / threads;
    if (blocks < 1) blocks = 1;
    softmax_xent_fused_backward_kernel<<<blocks, threads, 0, stream>>>(
        y_prob, y_true, grad_z, B, C
    );
    CUDA_CHECK(cudaGetLastError());
}
