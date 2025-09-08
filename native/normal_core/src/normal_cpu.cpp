#define _USE_MATH_DEFINES
#include <cmath>
#include <random>
#include <algorithm>
#include "normal/normal.hpp"

namespace normal {

void generate_cpu(float* out, std::size_t n, float mean, float std, std::uint64_t seed) {
    if (!out || n == 0) return;
    std::mt19937_64 rng(seed ? seed : 0x9E3779B97F4A7C15ull);
    std::normal_distribution<float> dist(mean, std > 0.f ? std : 0.f);

    // 간단한 벡터화-친화 루프
    for (std::size_t i = 0; i < n; ++i) {
        out[i] = dist(rng);
    }
}

void estimate_stats(const float* data, std::size_t n, double& mean, double& stdev) {
    if (!data || n == 0) { mean = stdev = 0.0; return; }
    long double m = 0.0L, m2 = 0.0L;
    for (std::size_t i = 0; i < n; ++i) {
        long double x = data[i];
        long double delta = x - m;
        m += delta / static_cast<long double>(i + 1);
        m2 += delta * (x - m);
    }
    mean = static_cast<double>(m);
    stdev = n > 1 ? std::sqrt(static_cast<double>(m2 / (n - 1))) : 0.0;
}

} // namespace normal
