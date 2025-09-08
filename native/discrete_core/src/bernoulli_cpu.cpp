#include <random>
#include <cmath>
#include "discrete/bernoulli.hpp"

namespace disc {
void bernoulli_cpu(float* out, std::size_t n, float p, std::uint64_t seed){
    if(!out || n==0) return;
    p = fminf(fmaxf(p, 1e-12f), 1.f-1e-12f);
    std::mt19937_64 rng(seed?seed:0x9E3779B97F4A7C15ull);
    std::uniform_real_distribution<float> U(0.f,1.f);
    for(std::size_t i=0;i<n;++i) out[i] = (U(rng) < p) ? 1.f : 0.f;
}
void bernoulli_logpmf_cpu(const float* x, std::size_t n, float p, float* out){
    if(!x || !out || n==0) return;
    p = fminf(fmaxf(p, 1e-12f), 1.f-1e-12f);
    const float lp = std::log(p), lq = std::log1p(-p);
    for(std::size_t i=0;i<n;++i) out[i] = (x[i]>0.5f) ? lp : lq;
}
} // namespace disc
