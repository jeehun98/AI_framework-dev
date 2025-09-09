#pragma once
#include <cstddef>
#include "prob_export.h"

namespace prob {

// Posterior over K classes (discrete): 
// input: log_prior[K], log_lik[K] for an observation B
// output: posterior_prob[K] (sum=1), optionally posterior_log[K]
PROB_API void posterior_from_prior_likelihood_cpu(
    const float* log_prior, const float* log_lik,
    std::size_t K,
    float* posterior_prob,           // out, size K
    float* posterior_log = nullptr   // optional
);

// Evidence (marginal likelihood): log P(B) = logsum_i log_prior[i]+log_lik[i]
PROB_API float log_evidence_cpu(const float* log_prior, const float* log_lik, std::size_t K);

#ifdef PROB_WITH_CUDA
PROB_API void posterior_from_prior_likelihood_cuda(
    const float* log_prior, const float* log_lik,
    std::size_t K,
    float* posterior_prob, float* posterior_log=nullptr,
    void* stream=nullptr);
PROB_API float log_evidence_cuda(const float* log_prior, const float* log_lik, std::size_t K, void* stream=nullptr);
#endif

} // namespace prob
