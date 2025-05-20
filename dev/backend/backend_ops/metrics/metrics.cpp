#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <stdexcept>
#include <algorithm>

namespace py = pybind11;

// Accuracy: 전체 예측 중에서 올바르게 예측한 비율
double accuracy(py::array_t<int> y_true, py::array_t<int> y_pred) {
    py::buffer_info buf_true = y_true.request(), buf_pred = y_pred.request();

    if (buf_true.size != buf_pred.size) {
        throw std::invalid_argument("Input arrays must have the same length");
    }

    int* ptr_true = static_cast<int*>(buf_true.ptr);
    int* ptr_pred = static_cast<int*>(buf_pred.ptr);
    int correct = 0;

    for (size_t i = 0; i < buf_true.size; ++i) {
        if (ptr_true[i] == ptr_pred[i]) {
            correct++;
        }
    }
    return static_cast<double>(correct) / buf_true.size;
}

// Precision: 올바르게 예측한 긍정 결과 / 예측된 모든 긍정 결과
double precision(py::array_t<int> y_true, py::array_t<int> y_pred) {
    py::buffer_info buf_true = y_true.request(), buf_pred = y_pred.request();

    if (buf_true.size != buf_pred.size) {
        throw std::invalid_argument("Input arrays must have the same length");
    }

    int* ptr_true = static_cast<int*>(buf_true.ptr);
    int* ptr_pred = static_cast<int*>(buf_pred.ptr);
    int true_positive = 0;
    int false_positive = 0;

    for (size_t i = 0; i < buf_true.size; ++i) {
        if (ptr_pred[i] == 1) {
            if (ptr_true[i] == 1) {
                true_positive++;
            } else {
                false_positive++;
            }
        }
    }
    return static_cast<double>(true_positive) / (true_positive + false_positive);
}

// Recall: 올바르게 예측한 긍정 결과 / 실제 모든 긍정 결과
double recall(py::array_t<int> y_true, py::array_t<int> y_pred) {
    py::buffer_info buf_true = y_true.request(), buf_pred = y_pred.request();

    if (buf_true.size != buf_pred.size) {
        throw std::invalid_argument("Input arrays must have the same length");
    }

    int* ptr_true = static_cast<int*>(buf_true.ptr);
    int* ptr_pred = static_cast<int*>(buf_pred.ptr);
    int true_positive = 0;
    int false_negative = 0;

    for (size_t i = 0; i < buf_true.size; ++i) {
        if (ptr_true[i] == 1) {
            if (ptr_pred[i] == 1) {
                true_positive++;
            } else {
                false_negative++;
            }
        }
    }
    return static_cast<double>(true_positive) / (true_positive + false_negative);
}

// F1 Score: Precision과 Recall의 조화 평균
double f1_score(py::array_t<int> y_true, py::array_t<int> y_pred) {
    double p = precision(y_true, y_pred);
    double r = recall(y_true, y_pred);
    
    if (p + r == 0) {
        return 0.0;
    }
    return 2 * (p * r) / (p + r);
}

double mean_squared_error(py::array_t<double> y_true, py::array_t<double> y_pred) {
    py::buffer_info buf_true = y_true.request(), buf_pred = y_pred.request();

    if (buf_true.ndim != buf_pred.ndim) {
        throw std::invalid_argument("Input arrays must have the same dimensions");
    }

    if (buf_true.size != buf_pred.size) {
        throw std::invalid_argument("Input arrays must have the same size");
    }

    double* ptr_true = static_cast<double*>(buf_true.ptr);
    double* ptr_pred = static_cast<double*>(buf_pred.ptr);
    double mse = 0.0;

    for (ssize_t i = 0; i < buf_true.size; ++i) {
        double diff = ptr_true[i] - ptr_pred[i];
        mse += diff * diff;
    }

    return mse / buf_true.size;
}