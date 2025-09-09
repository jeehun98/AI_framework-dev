#include "decision/decision.hpp"
#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace decision {

Decision map_decision(const std::vector<double>& posterior) {
    Decision out;
    if (posterior.empty()) return out;

    int best = 0;
    double best_score = posterior[0];
    for (int i = 1; i < static_cast<int>(posterior.size()); ++i) {
        if (posterior[i] > best_score) {
            best = i;
            best_score = posterior[i];
        }
    }
    out.cls = best;
    out.score = best_score;
    return out;
}

std::vector<Decision> topk_decision(const std::vector<double>& posterior, std::size_t k) {
    std::vector<Decision> res;
    if (posterior.empty() || k == 0) return res;

    // 인덱스 기반 정렬 (값 내림차순)
    std::vector<int> idx(posterior.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(idx.begin(), idx.begin() + std::min(k, idx.size()), idx.end(),
                      [&](int a, int b){ return posterior[a] > posterior[b]; });

    std::size_t outk = std::min(k, idx.size());
    res.reserve(outk);
    for (std::size_t i = 0; i < outk; ++i) {
        res.push_back({ idx[i], posterior[idx[i]] });
    }
    return res;
}

} // namespace decision
