#pragma once
#include <vector>
#include "loss.hpp"

namespace decision {

/**
 * @brief 거부 옵션 포함 결정 규칙
 *
 * 두 가지 방식을 함께 제공:
 *  1) 임계값 기반: max posterior < threshold 이면 거부(= -1)
 *  2) 손실 기반: 거부 손실을 하나의 "의사 클래스"로 추가하여 기대 손실이 가장 작은 행동을 선택
 *
 * @param posterior   p(C_k | x)
 * @param L           손실행렬 K x K (대각이 0일 필요는 없지만 보통 0)
 * @param reject_loss 거부 시의 손실 L_{k, R} (모든 실제 k에 동일한 상수라고 가정)
 * @param threshold   임계값 (0~1). 음수 주면 임계값 규칙 비활성화
 * @return 예측 클래스 인덱스 (0..K-1), 거부 시 -1
 */
int decision_with_reject(const std::vector<double>& posterior,
                         const LossMatrix& L,
                         double reject_loss,
                         double threshold = -1.0);

} // namespace decision
