// metrics.cpp
#include <vector>
#include <stdexcept>
#include <algorithm>

// Accuracy: 전체 예측 중에서 올바르게 예측한 비율
double accuracy(const std::vector<int>& y_true, const std::vector<int>& y_pred) {
    if (y_true.size() != y_pred.size()) {
        throw std::invalid_argument("Vectors y_true and y_pred must have the same length");
    }

    int correct = 0;
    for (size_t i = 0; i < y_true.size(); ++i) {
        if (y_true[i] == y_pred[i]) {
            correct++;
        }
    }
    return static_cast<double>(correct) / y_true.size();
}

// Precision: 올바르게 예측한 긍정 결과 / 예측된 모든 긍정 결과
double precision(const std::vector<int>& y_true, const std::vector<int>& y_pred) {
    if (y_true.size() != y_pred.size()) {
        throw std::invalid_argument("Vectors y_true and y_pred must have the same length");
    }

    int true_positive = 0;
    int false_positive = 0;
    for (size_t i = 0; i < y_true.size(); ++i) {
        if (y_pred[i] == 1) {
            if (y_true[i] == 1) {
                true_positive++;
            } else {
                false_positive++;
            }
        }
    }
    return static_cast<double>(true_positive) / (true_positive + false_positive);
}

// Recall: 올바르게 예측한 긍정 결과 / 실제 모든 긍정 결과
double recall(const std::vector<int>& y_true, const std::vector<int>& y_pred) {
    if (y_true.size() != y_pred.size()) {
        throw std::invalid_argument("Vectors y_true and y_pred must have the same length");
    }

    int true_positive = 0;
    int false_negative = 0;
    for (size_t i = 0; i < y_true.size(); ++i) {
        if (y_true[i] == 1) {
            if (y_pred[i] == 1) {
                true_positive++;
            } else {
                false_negative++;
            }
        }
    }
    return static_cast<double>(true_positive) / (true_positive + false_negative);
}

// F1 Score: Precision과 Recall의 조화 평균
double f1_score(const std::vector<int>& y_true, const std::vector<int>& y_pred) {
    double p = precision(y_true, y_pred);
    double r = recall(y_true, y_pred);
    
    if (p + r == 0) {
        return 0.0;
    }
    return 2 * (p * r) / (p + r);
}
