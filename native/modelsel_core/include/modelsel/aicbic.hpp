#pragma once
#include <cmath>
#include <cstddef>

namespace modelsel {

/** AIC / BIC */
inline double aic(double total_loglik, int k) {
    return -2.0 * total_loglik + 2.0 * k;
}
inline double bic(double total_loglik, int k, std::size_t n) {
    return -2.0 * total_loglik + std::log((double)n) * k;
}

} // namespace modelsel
