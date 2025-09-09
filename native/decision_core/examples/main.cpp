#include <iostream>
#include <vector>
#include <sstream>   // 문자열 변환용
#include "decision/decision.hpp"
#include "decision/loss.hpp"
#include "decision/reject.hpp"
#include "decision/regression.hpp"

// 안전한 숫자 → 문자열 변환 함수
template <typename T>
std::string to_str(const T& v) {
    std::ostringstream oss;
    oss << v;
    return oss.str();
}

int main() {
    // -------------------------------
    // 1) MAP / Top-k
    // -------------------------------
    std::vector<double> posterior = {0.2, 0.7, 0.1};
    auto d = decision::map_decision(posterior);
    std::cout << "[MAP] class=" << d.cls << " score=" << d.score << "\n";

    auto top2 = decision::topk_decision(posterior, 2);
    std::cout << "[Top-2] ";
    for (auto& t : top2) std::cout << "(" << t.cls << "," << t.score << ") ";
    std::cout << "\n";

    // -------------------------------
    // 2) Min Expected Loss
    // -------------------------------
    decision::LossMatrix L = {
        {0.0, 1.0, 5.0},
        {1.0, 0.0, 1.0},
        {5.0, 1.0, 0.0}
    };
    int best = decision::min_expected_loss(posterior, L);
    std::cout << "[Min-Expected-Loss] class=" << best << "\n";

    // -------------------------------
    // 3) Reject Option
    // -------------------------------
    int cls_or_rej = decision::decision_with_reject(
        posterior, L,
        /*reject_loss=*/0.3,
        /*threshold=*/0.8
    );
    std::cout << "[Reject-Option] result="
              << (cls_or_rej < 0 ? std::string("Reject") : to_str(cls_or_rej))
              << "\n";

    int cls_or_rej2 = decision::decision_with_reject(
        posterior, L,
        /*reject_loss=*/0.3,
        /*threshold=*/0.5
    );
    std::cout << "[Reject-Option-2] result="
              << (cls_or_rej2 < 0 ? std::string("Reject") : to_str(cls_or_rej2))
              << "\n";

    // -------------------------------
    // 4) Regression: mean/median
    // -------------------------------
    std::vector<double> tvals = {-1.0, 0.0, 2.0};
    std::vector<double> probs = { 0.2, 0.5, 0.3}; // 합 1 아니어도 normalize 됨
    double m = decision::regression_mean(tvals, probs);
    double med = decision::regression_median(tvals, probs);
    std::cout << "[Regression] mean=" << m << " median=" << med << "\n";

    return 0;
}
