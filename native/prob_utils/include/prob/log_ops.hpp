#pragma once
#include <cstddef>
#include <cstdint>
#include "prob_export.h"

namespace prob {

// Safe elementary ops (log-domain)
PROB_API float logaddexp(float a, float b);            // log(e^a + e^b)
PROB_API float logsumexp(const float* x, std::size_t n);

// Numerically-stable helpers
PROB_API float log1mexp(float x);  // log(1 - e^{-x}) with stability for small x>0
PROB_API float log1pexp(float x);  // log(1 + e^x) stable

// Simple conditional/marginal relations (log-domain)
inline float log_conditional(float log_joint, float log_margB) { return log_joint - log_margB; }
inline float log_joint_from_cond(float log_cond, float log_margB) { return log_cond + log_margB; }

#ifdef PROB_WITH_CUDA
// Device versions operate on arrays (len K), results to out
PROB_API void logsumexp_cuda(const float* x, std::size_t n, float* out, void* stream=nullptr);
#endif

} // namespace prob
