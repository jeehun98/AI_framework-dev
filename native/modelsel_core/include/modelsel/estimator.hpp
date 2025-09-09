#pragma once
#include "dataset.hpp"
#include <vector>

namespace modelsel {

/** 파라미터 구조체 */
struct BernoulliParams { double p; };
struct BinomialParams  { double p; };

/** 추정기 (MLE / MAP[Beta]) */
BernoulliParams bernoulli_mle(const BernoulliData& D, const std::vector<int>& idx);
BernoulliParams bernoulli_map_beta(const BernoulliData& D, const std::vector<int>& idx,
                                   double alpha, double beta);

BinomialParams  binomial_mle(const BinomialData& D, const std::vector<int>& idx);
BinomialParams  binomial_map_beta(const BinomialData& D, const std::vector<int>& idx,
                                  double alpha, double beta);

} // namespace modelsel
