#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>

namespace py = pybind11;

// matrix_add
py::array_t<double> matrix_add(py::array_t<double> A, py::array_t<double> B) {
    // Numpy 배열의 버퍼 정보 가져오기
    py::buffer_info bufA = A.request(), bufB = B.request();

    // 입력 배열이 2D인지 확인
    if (bufA.ndim != 2 || bufB.ndim != 2) {
        throw std::runtime_error("Input should be 2-D NumPy arrays");
    }

    // 행과 열의 크기 확인
    if (bufA.shape[0] != bufB.shape[0] || bufA.shape[1] != bufB.shape[1]) {
        throw std::runtime_error("Input matrices must have the same shape");
    }

    // 행과 열의 크기
    size_t rows = bufA.shape[0];
    size_t cols = bufA.shape[1];

    // 결과 배열 생성
    py::array_t<double> result = py::array_t<double>({rows, cols});
    py::buffer_info bufResult = result.request();

    double* ptrA = static_cast<double*>(bufA.ptr);
    double* ptrB = static_cast<double*>(bufB.ptr);
    double* ptrResult = static_cast<double*>(bufResult.ptr);

    // 행렬 덧셈
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            ptrResult[i * cols + j] = ptrA[i * cols + j] + ptrB[i * cols + j];
        }
    }

    return result;
}

py::array_t<double> matrix_multiply(py::array_t<double> A, py::array_t<double> B) {
    // Numpy 배열의 버퍼 정보 가져오기
    py::buffer_info bufA = A.request(), bufB = B.request();

    // 입력 배열이 2D인지 확인
    if (bufA.ndim != 2 || bufB.ndim != 2) {
        throw std::runtime_error("Input should be 2-D NumPy arrays");
    }

    // 행렬 곱셈이 가능한지 확인
    if (bufA.shape[1] != bufB.shape[0]) {
        throw std::runtime_error("Inner matrix dimensions must agree");
    }

    // 행과 열의 크기
    size_t rows = bufA.shape[0];
    size_t cols = bufB.shape[1];
    size_t inner_dim = bufA.shape[1];

    // 결과 배열 생성
    py::array_t<double> result = py::array_t<double>({rows, cols});
    py::buffer_info bufResult = result.request();

    double* ptrA = static_cast<double*>(bufA.ptr);
    double* ptrB = static_cast<double*>(bufB.ptr);
    double* ptrResult = static_cast<double*>(bufResult.ptr);

    // 행렬 곱셈
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            ptrResult[i * cols + j] = 0;
            for (size_t k = 0; k < inner_dim; ++k) {
                ptrResult[i * cols + j] += ptrA[i * inner_dim + k] * ptrB[k * cols + j];
            }
        }
    }

    return result;
}
