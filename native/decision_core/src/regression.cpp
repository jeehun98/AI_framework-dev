#include "decision/regression.hpp"
#include <numeric>
#include <cmath>

namespace decision {

static void normalize(std::vector<double>& probs) {
    double s = std::accumulate(probs.begin(), probs.end(), 0.0);
    if (s <= 0.0) throw std::invalid_argument("probabilities sum to non-positive");
    for (double& p : probs) p /= s;
}

double regression_mean(const std::vector<double>& values,
                       const std::vector<double>& probs) {
    if (values.size() != probs.size() || values.empty())
        throw std::invalid_argument("values/probs size mismatch or empty");
    std::vector<double> q = probs;
    normalize(q);
    double m = 0.0;
    for (std::size_t i = 0; i < values.size(); ++i) m += values[i] * q[i];
    return m;
}

double regression_median(std::vector<double> values,
                         std::vector<double> probs) {
    if (values.size() != probs.size() || values.empty())
        throw std::invalid_argument("values/probs size mismatch or empty");
    normalize(probs);

    // values 기준 오름차순 정렬 후, 누적 확률 0.5 이상이 되는 첫 지점
    std::vector<std::size_t> idx(values.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&](std::size_t a, std::size_t b){
        return values[a] < values[b];
    });

    double cdf = 0.0;
    for (std::size_t r = 0; r < idx.size(); ++r) {
        cdf += probs[idx[r]];
        if (cdf >= 0.5) return values[idx[r]];
    }
    // 수치적으로 마지막 원소 반환 (여기까지 오는 일은 거의 없음)
    return values[idx.back()];
}

} // namespace decision
