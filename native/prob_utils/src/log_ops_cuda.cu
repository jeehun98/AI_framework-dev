// log_ops_cuda.cu
#include "prob/log_ops.hpp"

#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <math_constants.h>
#include <limits>

__host__ __device__ inline float neg_inf() {
#if defined(__CUDA_ARCH__)
    return -CUDART_INF_F;
#else
    return -std::numeric_limits<float>::infinity();
#endif
}

namespace prob {

__global__ void lse_reduce_kernel(const float* __restrict__ x, int n, float* __restrict__ out) {
    extern __shared__ float s[];
    const int tid = threadIdx.x;

    float m = -CUDART_INF_F;
    for (int i = tid; i < n; i += blockDim.x) m = fmaxf(m, x[i]);
    s[tid] = m;
    __syncthreads();

    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (tid < offset) s[tid] = fmaxf(s[tid], s[tid + offset]);
        __syncthreads();
    }

    const float maxv = s[0];

    if (maxv == -CUDART_INF_F) { if (tid == 0) *out = -CUDART_INF_F; return; }

    float sum = 0.f;
    for (int i = tid; i < n; i += blockDim.x) sum += __expf(x[i] - maxv);
    s[tid] = sum;
    __syncthreads();

    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (tid < offset) s[tid] += s[tid + offset];
        __syncthreads();
    }

    if (tid == 0) {
        #ifdef PROB_USE_FAST_MATH
            *out = maxv + __logf(s[0]);
        #else
            *out = maxv + logf(s[0]);
        #endif
    }
}

void logsumexp_cuda(const float* x, std::size_t n, float* out, void* stream_v) {
    if (!x || !out || n == 0) { if (out) *out = neg_inf(); return; }  // ← 여기!

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_v);
    const int block = 256;
    const size_t shmem = block * sizeof(float);

    lse_reduce_kernel<<<1, block, shmem, stream>>>(x, static_cast<int>(n), out);

    #ifndef NDEBUG
    cudaError_t st = cudaGetLastError();
    if (st != cudaSuccess) {
        // fprintf(stderr, "lse_reduce_kernel launch failed: %s\n", cudaGetErrorString(st));
    }
    #endif
}

} // namespace prob

#else // __CUDACC__ not defined (CPU fallback)

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>

namespace prob {

void logsumexp_cuda(const float* x, std::size_t n, float* out, void* /*stream_v*/) {
    if (!x || !out || n == 0) { if (out) *out = -std::numeric_limits<float>::infinity(); return; }

    float m = -std::numeric_limits<float>::infinity();
    for (std::size_t i = 0; i < n; ++i) m = std::max(m, x[i]);

    if (!std::isfinite(m)) { *out = -std::numeric_limits<float>::infinity(); return; }

    double sum = 0.0;
    for (std::size_t i = 0; i < n; ++i) sum += std::exp((double)x[i] - (double)m);

    *out = m + (float)std::log(sum);
}

} // namespace prob

#endif // __CUDACC__
