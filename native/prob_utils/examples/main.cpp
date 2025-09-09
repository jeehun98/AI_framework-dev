#include <iostream>
#include <vector>
#include <cmath>
#include "prob/log_ops.hpp"
#include "prob/bayes.hpp"

static inline float flog(float x){ return std::log(x); }

int main() {
    // Two-class toy: A vs ~A
    // Prior: P(A)=0.6, P(~A)=0.4
    // Likelihood for evidence B: P(B|A)=0.7, P(B|~A)=0.2
    const int K=2;
    float prior[2]   = {0.6f, 0.4f};
    float lik_B[2]   = {0.7f, 0.2f};
    float lik_notB[2]= {1.f-0.7f, 1.f-0.2f}; // 0.3, 0.8

    float log_prior[2] = {flog(prior[0]), flog(prior[1])};
    float log_likB[2]  = {flog(lik_B[0]), flog(lik_B[1])};

    // Posterior P(A|B)
    float post_prob[2], post_log[2];
    prob::posterior_from_prior_likelihood_cpu(log_prior, log_likB, K, post_prob, post_log);

    float logZ = prob::log_evidence_cpu(log_prior, log_likB, K);
    float pB   = std::exp(logZ);

    std::cout << "Evidence P(B)        = " << pB << "\n";
    std::cout << "Posterior P(A|B)     = " << post_prob[0] << "\n";
    std::cout << "Check: P(A|B)P(B)    = " << post_prob[0]*pB << "\n";
    std::cout << "Check: P(B|A)P(A)    = " << lik_B[0]*prior[0] << "\n";

    // Should match by Bayes: P(A|B)P(B) == P(B|A)P(A)

    return 0;
}
