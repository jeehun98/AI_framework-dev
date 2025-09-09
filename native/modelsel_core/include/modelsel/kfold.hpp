#pragma once
#include "dataset.hpp"
#include "split.hpp"
#include <string>

namespace modelsel {

/** CV 결과 */
struct CVResult {
    double mean_logloss{0.0}, std_logloss{0.0};
    double mean_acc{0.0},     std_acc{0.0};
    int    k{0};
};

/** Bernoulli K-fold CV */
CVResult kfold_cv_bernoulli(const BernoulliData& D, int kfold, uint64_t seed,
                            bool stratified, std::string backend,
                            bool use_map=false, double alpha=1.0, double beta=1.0);

/** Binomial K-fold CV */
CVResult kfold_cv_binomial(const BinomialData& D, int kfold, uint64_t seed,
                           std::string backend,
                           bool use_map=false, double alpha=1.0, double beta=1.0);

} // namespace modelsel
