#pragma once
#include <vector>
#include <cstddef>

namespace decision {

/**
 * @brief 단일 결정 결과 (MAP/Top-k에서 공통 사용)
 * cls   : 선택된 클래스 인덱스 (0-based)
 * score : 해당 클래스의 posterior 점수 (p(C|x))
 */
struct Decision {
    int cls{-1};
    double score{0.0};
};

/**
 * @brief MAP 결정 규칙: posterior에서 argmax를 선택
 * @param posterior  크기 K 벡터, 각 원소는 p(C_k | x), 합이 1이 아니어도 내부에서 비교만 하므로 OK
 * @return Decision{ argmax k, max value }
 */
Decision map_decision(const std::vector<double>& posterior);

/**
 * @brief Top-k 결정: posterior 상위 k개를 점수 높은 순으로 반환
 * @param posterior  posterior 벡터
 * @param k          반환할 개수 (k > 0)
 * @return 내림차순 정렬된 Decision 벡터 (size = min(k, K))
 */
std::vector<Decision> topk_decision(const std::vector<double>& posterior, std::size_t k);

} // namespace decision
