#include "modelsel/estimator.hpp"

namespace modelsel {

BernoulliParams bernoulli_mle(const BernoulliData& D, const std::vector<int>& idx) {
    int sum = 0; for (int i : idx) sum += D.y[i];
    double p = (idx.empty() ? 0.0 : (double)sum / (double)idx.size());
    return {p};
}

BernoulliParams bernoulli_map_beta(const BernoulliData& D, const std::vector<int>& idx,
                                   double alpha, double beta) {
    int sum = 0; for (int i : idx) sum += D.y[i];
    double p = (idx.empty() ? 0.0 : (sum + alpha - 1.0) / (idx.size() + alpha + beta - 2.0));
    return {p};
}

BinomialParams binomial_mle(const BinomialData& D, const std::vector<int>& idx) {
    long long sumk=0, sumn=0;
    for (int i : idx) { sumk += D.k[i]; sumn += D.n[i]; }
    double p = (sumn == 0 ? 0.0 : (double)sumk / (double)sumn);
    return {p};
}

BinomialParams binomial_map_beta(const BinomialData& D, const std::vector<int>& idx,
                                 double alpha, double beta) {
    long long sumk=0, sumn=0;
    for (int i : idx) { sumk += D.k[i]; sumn += D.n[i]; }
    double p = (sumn + alpha + beta - 2.0 == 0.0)
             ? 0.0
             : (sumk + alpha - 1.0) / (sumn + alpha + beta - 2.0);
    return {p};
}

} // namespace modelsel
