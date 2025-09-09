#include "decision/loss.hpp"
#include <stdexcept>
#include <limits>

namespace decision {

static void check_shapes(const std::vector<double>& posterior, const LossMatrix& L) {
    const int K = static_cast<int>(posterior.size());
    if (K == 0) throw std::invalid_argument("posterior is empty");
    if (static_cast<int>(L.size()) != K) throw std::invalid_argument("LossMatrix rows != K");
    for (const auto& row : L) {
        if (static_cast<int>(row.size()) != K) throw std::invalid_argument("LossMatrix cols != K");
    }
}

double expected_loss_for(const std::vector<double>& posterior, const LossMatrix& L, int j) {
    const int K = static_cast<int>(posterior.size());
    if (j < 0 || j >= K) throw std::out_of_range("j out of range");
    double r = 0.0;
    for (int k = 0; k < K; ++k) {
        r += L[k][j] * posterior[k];
    }
    return r;
}

int min_expected_loss(const std::vector<double>& posterior, const LossMatrix& L) {
    check_shapes(posterior, L);
    const int K = static_cast<int>(posterior.size());
    double best_val = std::numeric_limits<double>::infinity();
    int best_j = -1;
    for (int j = 0; j < K; ++j) {
        double r = expected_loss_for(posterior, L, j);
        if (r < best_val) {
            best_val = r;
            best_j = j;
        }
    }
    return best_j;
}

} // namespace decision
