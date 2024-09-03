#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <stdexcept>

namespace py = pybind11;

double mean_squared_error(py::array_t<double> y_true, py::array_t<double> y_pred) {
    // 버퍼 정보 가져오기
    py::buffer_info buf_true = y_true.request(), buf_pred = y_pred.request();

    // 입력 배열이 1D인지 확인
    if (buf_true.ndim != 1 || buf_pred.ndim != 1) {
        throw std::invalid_argument("Input arrays must be 1-dimensional");
    }

    // 벡터 크기 일치 여부 확인
    if (buf_true.size != buf_pred.size) {
        throw std::invalid_argument("Input arrays must have the same length");
    }

    double* ptr_true = (double*)buf_true.ptr;
    double* ptr_pred = (double*)buf_pred.ptr;
    double mse = 0.0;

    // MSE 계산
    for (size_t i = 0; i < buf_true.size; ++i) {
        double diff = ptr_true[i] - ptr_pred[i];
        mse += diff * diff;
    }
    return mse / buf_true.size;
}

double cross_entropy_loss(py::array_t<double> y_true, py::array_t<double> y_pred) {
    // 버퍼 정보 가져오기
    py::buffer_info buf_true = y_true.request(), buf_pred = y_pred.request();

    // 입력 배열이 1D인지 확인
    if (buf_true.ndim != 1 || buf_pred.ndim != 1) {
        throw std::invalid_argument("Input arrays must be 1-dimensional");
    }

    // 벡터 크기 일치 여부 확인
    if (buf_true.size != buf_pred.size) {
        throw std::invalid_argument("Input arrays must have the same length");
    }

    double* ptr_true = (double*)buf_true.ptr;
    double* ptr_pred = (double*)buf_pred.ptr;
    double loss = 0.0;

    // Cross-Entropy Loss 계산
    for (size_t i = 0; i < buf_true.size; ++i) {
        loss -= ptr_true[i] * std::log(ptr_pred[i]) + (1.0 - ptr_true[i]) * std::log(1.0 - ptr_pred[i]);
    }
    return loss / buf_true.size;
}
