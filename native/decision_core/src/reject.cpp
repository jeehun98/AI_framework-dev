#include "decision/reject.hpp"
#include <limits>
#include <stdexcept>

namespace decision {

/**
 * 거부의 기대손실 R(R|x) = sum_k L_{kR} p(C_k|x)를
 * L_{kR} = reject_loss (상수)로 가정하면,
 * R(R|x) = reject_loss * sum_k p(C_k|x) = reject_loss (posterior 정규화 무관)
 */
static double reject_expected_loss(const std::vector<double>& posterior, double reject_loss) {
    (void)posterior; // posterior 합이 1이 아니어도 상수이므로 무의미
    return reject_loss;
}

int decision_with_reject(const std::vector<double>& posterior,
                         const LossMatrix& L,
                         double reject_loss,
                         double threshold) {
    if (posterior.empty()) throw std::invalid_argument("posterior is empty");
    // 1) 임계값 규칙: max posterior < threshold -> 거부
    if (threshold >= 0.0) {
        double maxp = 0.0;
        for (double p : posterior) if (p > maxp) maxp = p;
        if (maxp < threshold) return -1; // Reject
    }

    // 2) 손실 기반: K개 행동(클래스) + 거부 행동 중 기대손실 최소 선택
    const int K = static_cast<int>(posterior.size());
    if (static_cast<int>(L.size()) != K) throw std::invalid_argument("LossMatrix rows != K");
    for (const auto& row : L) {
        if (static_cast<int>(row.size()) != K) throw std::invalid_argument("LossMatrix cols != K");
    }

    // 클래스 행동들의 최소 기대손실
    double best_val = std::numeric_limits<double>::infinity();
    int best_j = -2; // -2: 아직 없음, -1: 거부

    for (int j = 0; j < K; ++j) {
        double r = 0.0;
        for (int k = 0; k < K; ++k) r += L[k][j] * posterior[k];
        if (r < best_val) {
            best_val = r;
            best_j = j;
        }
    }

    // 거부 기대손실과 비교
    const double r_rej = reject_expected_loss(posterior, reject_loss);
    if (r_rej < best_val) return -1; // Reject 선택이 더 유리
    return best_j;
}

} // namespace decision
