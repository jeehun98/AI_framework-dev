// examples/main.cpp
#include <iostream>
#include <random>
#include <vector>
#include <cstdint>

#include "modelsel/dataset.hpp"
#include "modelsel/kfold.hpp"
#include "modelsel/report.hpp"

namespace ms = modelsel;

// 합성 Bernoulli 데이터
static ms::BernoulliData make_synthetic_bernoulli(size_t N, double p_true, uint64_t seed) {
    ms::BernoulliData D;
    D.y.resize(N);

    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> U(0.0, 1.0);
    for (size_t i = 0; i < N; ++i) {
        D.y[i] = static_cast<uint8_t>(U(rng) < p_true);
    }
    return D;
}

int main() {
    const size_t N = 5000;
    const double p_true = 0.30;
    const uint64_t seed = 42;

    auto D = make_synthetic_bernoulli(N, p_true, seed);

    const int kfold = 5;
    const bool stratified = false;
    const std::string backend = "cpu";

    const bool use_map = true;
    const double alpha = 2.0, beta = 2.0;

    auto cv = ms::kfold_cv_bernoulli(D, kfold, seed, stratified, backend,
                                     use_map, alpha, beta);

    std::cout << "==== Bernoulli 5-Fold CV Example ====\n";
    std::cout << "N=" << N << ", p_true=" << p_true
              << ", estimator=" << (use_map ? "MAP[Beta(" + std::to_string(alpha) + "," + std::to_string(beta) + ")]"
                                            : "MLE")
              << ", backend=" << backend << "\n";
    ms::print_cv_report(cv);
    return 0;
}
