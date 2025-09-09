#pragma once
#include <vector>

namespace decision {

/**
 * @brief 손실 행렬: L_{k j} = 실제 클래스 k, 예측 클래스 j일 때의 손실
 * - 대각 L_{k k} = 0 (보통의 가정) 이지만, 일반화를 위해 0이 아닐 수도 있음
 * - 크기는 K x K
 */
using LossMatrix = std::vector<std::vector<double>>;

/**
 * @brief posterior와 손실행렬이 주어졌을 때, 기대 손실을 최소화하는 예측 클래스를 반환
 * R(j|x) = sum_k L_{k j} * p(C_k | x)
 * @param posterior p(C_k | x) 길이 K
 * @param L         손실행렬 K x K
 * @return argmin_j R(j|x) (0-based 인덱스)
 */
int min_expected_loss(const std::vector<double>& posterior, const LossMatrix& L);

/**
 * @brief 특정 클래스 j에 대한 기대 손실 R(j|x)을 계산 (디버그/분석용)
 */
double expected_loss_for(const std::vector<double>& posterior, const LossMatrix& L, int j);

} // namespace decision
