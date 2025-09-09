#include "modelsel/evaluator.hpp"
#include <cmath>

namespace modelsel {

static inline double clamp_p(double p) {
    const double eps = 1e-12;
    if (p < eps) return eps;
    if (p > 1.0 - eps) return 1.0 - eps;
    return p;
}

BernoulliEvaluator::BernoulliEvaluator(std::string backend) : backend_(std::move(backend)) {}

EvalStats BernoulliEvaluator::evaluate(const BernoulliData& D, const std::vector<int>& idx, double p) {
    p = clamp_p(p);
    double loglik = 0.0, acc = 0.0;
    for (int i : idx) {
        uint8_t y = D.y[i];
        loglik += y ? std::log(p) : std::log1p(-p);
        int pred = (p >= 0.5);
        if (pred == (int)y) acc += 1.0;
    }
    EvalStats s;
    s.loglik  = loglik;
    s.logloss = idx.empty() ? 0.0 : -loglik / (double)idx.size();
    s.accuracy= idx.empty() ? 0.0 :  acc    / (double)idx.size();
    s.count   = idx.size();
    return s;
}

BinomialEvaluator::BinomialEvaluator(std::string backend) : backend_(std::move(backend)) {}

EvalStats BinomialEvaluator::evaluate(const BinomialData& D, const std::vector<int>& idx, double p) {
    p = clamp_p(p);
    double loglik = 0.0, acc = 0.0;
    for (int i : idx) {
        int k = D.k[i], n = D.n[i];
        double lp = std::lgamma(n+1.0) - std::lgamma(k+1.0) - std::lgamma(n-k+1.0)
                  + k*std::log(p) + (n-k)*std::log1p(-p);
        loglik += lp;
        int pred = (p >= 0.5) ? n : 0; // 단순 정답 일치 기준(데모)
        if (pred == k) acc += 1.0;
    }
    EvalStats s;
    s.loglik  = loglik;
    s.logloss = idx.empty() ? 0.0 : -loglik / (double)idx.size();
    s.accuracy= idx.empty() ? 0.0 :  acc    / (double)idx.size();
    s.count   = idx.size();
    return s;
}

} // namespace modelsel
