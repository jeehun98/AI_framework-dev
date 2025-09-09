#pragma once
#include "dataset.hpp"
#include <string>

namespace modelsel {

/** 평가 통계 */
struct EvalStats {
    double loglik{0.0};
    double logloss{0.0};
    double accuracy{0.0};
    std::size_t count{0};
};

class BernoulliEvaluator {
public:
    explicit BernoulliEvaluator(std::string backend="cpu");
    EvalStats evaluate(const BernoulliData& D, const std::vector<int>& idx, double p);
private:
    std::string backend_;
};

class BinomialEvaluator {
public:
    explicit BinomialEvaluator(std::string backend="cpu");
    EvalStats evaluate(const BinomialData& D, const std::vector<int>& idx, double p);
private:
    std::string backend_;
};

} // namespace modelsel
