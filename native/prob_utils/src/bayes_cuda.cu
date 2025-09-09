#ifdef __CUDACC__

#include <cuda_runtime.h>
#include <math_constants.h>   // CUDART_INF_F
#include <limits>
#include "prob/bayes.hpp"
#include "prob/log_ops.hpp"

// host/device 공용 음의 무한대
__host__ __device__ inline float neg_inf() {
#if defined(__CUDA_ARCH__)
    return -CUDART_INF_F;                           // device 경로
#else
    return -std::numeric_limits<float>::infinity(); // host 경로
#endif
}

namespace prob {

// grid-stride: tmp[i] = log_prior[i] + log_lik[i]
__global__ void add_arrays_kernel(const float* __restrict__ a,
                                  const float* __restrict__ b,
                                  float* __restrict__ out, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n;
         i += blockDim.x * gridDim.x) {
        out[i] = a[i] + b[i];
    }
}

// log p(x) = logsumexp_i( log p(z=i) + log p(x|z=i) )
float log_evidence_cuda(const float* log_prior, const float* log_lik,
                        std::size_t K, void* stream_v) {
    // ✅ host 함수: 공용 헬퍼로 -∞ 반환(장소에 상관없이 안전)
    if (!log_prior || !log_lik || K == 0) return neg_inf();

    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream_v); // nullptr이면 default stream

    float* d_tmp = nullptr;
    float* d_out = nullptr;
    if (cudaMalloc(&d_tmp, K * sizeof(float)) != cudaSuccess) {
        return neg_inf();
    }
    if (cudaMalloc(&d_out, sizeof(float)) != cudaSuccess) {
        cudaFree(d_tmp);
        return neg_inf();
    }

    const int block = 256;
    const int grid  = static_cast<int>((K + block - 1) / block);
    add_arrays_kernel<<<grid, block, 0, s>>>(log_prior, log_lik, d_tmp, static_cast<int>(K));

    // logsumexp_cuda: device에서 d_out[0]에 결과 기록
    logsumexp_cuda(d_tmp, K, d_out, s);

    float logZ = 0.f;
    cudaMemcpyAsync(&logZ, d_out, sizeof(float), cudaMemcpyDeviceToHost, s);
    cudaStreamSynchronize(s);

    cudaFree(d_tmp);
    cudaFree(d_out);
    return logZ;
}

__global__ void posterior_kernel(const float* __restrict__ log_prior,
                                 const float* __restrict__ log_lik,
                                 int K, float logZ,
                                 float* __restrict__ post_prob,
                                 float* __restrict__ post_log) {
    const bool invalid_Z = !isfinite(logZ);

    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < K;
         i += blockDim.x * gridDim.x) {

        if (invalid_Z) {
            if (post_log) post_log[i] = -CUDART_INF_F;
            post_prob[i] = 0.f;
        } else {
            float lp = log_prior[i] + log_lik[i] - logZ;
            if (post_log) post_log[i] = lp;
            post_prob[i] = __expf(lp);
        }
    }
}

void posterior_from_prior_likelihood_cuda(const float* log_prior,
        const float* log_lik, std::size_t K,
        float* post_prob, float* post_log, void* stream_v) {
    if (!log_prior || !log_lik || !post_prob || K == 0) return;

    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream_v);

    // 증거(logZ) 계산
    float logZ = log_evidence_cuda(log_prior, log_lik, K, s);

    const int block = 256;
    const int grid  = static_cast<int>(std::min<std::size_t>(
                        (K + block - 1) / block, 65535));

    posterior_kernel<<<grid, block, 0, s>>>(log_prior, log_lik, static_cast<int>(K),
                                            logZ, post_prob, post_log);
}

} // namespace prob

#else  // __CUDACC__ 없는 CPU 폴백

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include "prob/bayes.hpp"
#include "prob/log_ops.hpp"

namespace prob {

// CPU 폴백: 단순 구현
float log_evidence_cuda(const float* log_prior, const float* log_lik,
                        std::size_t K, void* /*stream_v*/) {
    if (!log_prior || !log_lik || K == 0)
        return -std::numeric_limits<float>::infinity();

    float m = -std::numeric_limits<float>::infinity();
    for (std::size_t i = 0; i < K; ++i)
        m = std::max(m, log_prior[i] + log_lik[i]);

    if (!std::isfinite(m))
        return -std::numeric_limits<float>::infinity();

    long double sum = 0.0L;
    for (std::size_t i = 0; i < K; ++i)
        sum += std::exp((long double)log_prior[i] + (long double)log_lik[i] - (long double)m);

    return m + (float)std::log((double)sum);
}

void posterior_from_prior_likelihood_cuda(const float* log_prior,
        const float* log_lik, std::size_t K,
        float* post_prob, float* post_log, void* /*stream_v*/) {
    if (!log_prior || !log_lik || !post_prob || K == 0) return;

    float logZ = log_evidence_cuda(log_prior, log_lik, K, nullptr);

    if (!std::isfinite(logZ)) {
        for (std::size_t i = 0; i < K; ++i) {
            if (post_log) post_log[i] = -std::numeric_limits<float>::infinity();
            post_prob[i] = 0.f;
        }
        return;
    }

    for (std::size_t i = 0; i < K; ++i) {
        float lp = log_prior[i] + log_lik[i] - logZ;
        if (post_log) post_log[i] = lp;
        post_prob[i] = std::exp(lp);
    }
}

} // namespace prob
#endif
