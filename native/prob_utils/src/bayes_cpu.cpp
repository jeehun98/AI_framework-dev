#include <cmath>
#include <vector>
#include "prob/log_ops.hpp"
#include "prob/bayes.hpp"

namespace prob {

float log_evidence_cpu(const float* log_prior, const float* log_lik, std::size_t K) {
    std::vector<float> tmp(K);
    for (std::size_t i=0;i<K;++i) tmp[i] = log_prior[i] + log_lik[i];
    return logsumexp(tmp.data(), K);
}

void posterior_from_prior_likelihood_cpu(
    const float* log_prior, const float* log_lik,
    std::size_t K, float* post_prob, float* post_log) {

    if (!log_prior || !log_lik || !post_prob || K==0) return;

    // log posterior up to constant: log_prior + log_lik - log_evidence
    float logZ = log_evidence_cpu(log_prior, log_lik, K);

    // Write log and prob (softmax in log-domain)
    // First pass: write log
    std::vector<float> lp(K);
    for (std::size_t i=0;i<K;++i) {
        lp[i] = log_prior[i] + log_lik[i] - logZ;
    }
    if (post_log) {
        for (std::size_t i=0;i<K;++i) post_log[i] = lp[i];
    }
    // Second pass: exp
    for (std::size_t i=0;i<K;++i) post_prob[i] = std::exp(lp[i]);
}

} // namespace prob
