#pragma once
#include <vector>
#include <stdexcept>
#include <algorithm>

namespace decision {

/**
 * @brief 회귀: 제곱 손실 하 최적해 = 조건부 평균 (discrete p(t|x) 가정)
 * @param values  t_i 값들
 * @param probs   해당 값들의 확률 p(t_i | x), 합이 1 아니어도 내부에서 정규화
 * @return sum_i (t_i * normalized probs_i)
 */
double regression_mean(const std::vector<double>& values,
                       const std::vector<double>& probs);

/**
 * @brief 회귀: 절대 손실 하 최적해 = 조건부 중앙값 (discrete p(t|x) 가정)
 * @param values  t_i 값들
 * @param probs   확률들, 내부 정규화 후 누적이 0.5 넘는 첫 지점의 t_i 반환
 * @return median value
 */
double regression_median(std::vector<double> values,
                         std::vector<double> probs);

} // namespace decision
