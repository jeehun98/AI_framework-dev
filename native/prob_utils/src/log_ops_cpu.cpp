#include <cmath>
#include <cfloat>
#include "prob/log_ops.hpp"

namespace prob {

float logaddexp(float a, float b) {
    // log(e^a + e^b) = m + log(e^{a-m} + e^{b-m}) with m=max(a,b)
    float m = (a > b) ? a : b;
    if (std::isinf(m) && m < 0) return m; // both -inf
    return m + std::log(std::exp(a - m) + std::exp(b - m));
}

float logsumexp(const float* x, std::size_t n) {
    if (!x || n == 0) return -INFINITY;
    float m = x[0];
    for (std::size_t i=1;i<n;++i) if (x[i] > m) m = x[i];
    if (std::isinf(m) && m < 0) return m; // all -inf
    long double s = 0.0L;
    for (std::size_t i=0;i<n;++i) s += std::exp((long double)x[i] - (long double)m);
    return m + (float)std::log((double)s);
}

// log(1 - e^{-x}) for x>0, numerically stable
float log1mexp(float x) {
    // Use two branches for stability (cf. Mpmath/Boost)
    // If x < log(2): log1p(-exp(-x)), else: log(-expm1(-x))
    const float LOG2 = 0.6931471805599453f;
    if (x <= 0.f) return -INFINITY; // undefined, caller ensures x>0
    if (x < LOG2) return std::log1p(-std::exp(-x));
    return std::log(-std::expm1(-x));
}

float log1pexp(float x) {
    // Stable log(1+e^x)
    if (x <= -20.f) return std::exp(x);         // e^x small
    if (x <= 20.f)  return std::log1p(std::exp(x));
    return x; // when x is large, log(1+e^x) ~ x
}

} // namespace prob
