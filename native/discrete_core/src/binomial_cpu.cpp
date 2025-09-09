#include <random>
#include <cmath>
#include "discrete/binomial.hpp"

namespace disc {
void binomial_cpu(float* out, std::size_t m, int n, float p, std::uint64_t seed){
    if(!out || m==0 || n<=0) return;
    p = fminf(fmaxf(p, 1e-12f), 1.f-1e-12f);
    std::mt19937_64 rng(seed?seed:0x9E3779B97F4A7C15ull);
    std::uniform_real_distribution<float> U(0.f,1.f);
    for(std::size_t i=0;i<m;++i){
        int k=0; for(int t=0;t<n;++t) k += (U(rng) < p);
        out[i] = static_cast<float>(k);
    }
}
void binomial_logpmf_cpu(const float* kv, std::size_t nk, int n, float p, float* out){
    if(!kv || !out || nk==0 || n<0) return;
    p = fminf(fmaxf(p, 1e-12f), 1.f-1e-12f);
    for(std::size_t i=0;i<nk;++i){
        float kf = kv[i];
        if(kf<0.f || kf>n){ out[i] = -INFINITY; continue; }
        double k = std::floor(kf + 0.5);
        double logC = std::lgamma(n+1.0) - std::lgamma(k+1.0) - std::lgamma(n-k+1.0);
        out[i] = static_cast<float>( logC + k*std::log(p) + (n-k)*std::log1p(-p) );
    }
}
} // namespace disc
