#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <vector>

namespace py = pybind11;

py::array_t<double> relu(py::array_t<double> inputs) {
    py::buffer_info buf = inputs.request();
    double* ptr = static_cast<double*>(buf.ptr);

    py::array_t<double> result(buf.size);
    py::buffer_info buf_result = result.request();
    double* ptr_result = static_cast<double*>(buf_result.ptr);

    for (size_t i = 0; i < buf.size; ++i) {
        ptr_result[i] = ptr[i] > 0 ? ptr[i] : 0;
    }

    return result;
}

py::array_t<double> sigmoid(py::array_t<double> inputs) {
    py::buffer_info buf = inputs.request();
    double* ptr = static_cast<double*>(buf.ptr);

    py::array_t<double> result(buf.size);
    py::buffer_info buf_result = result.request();
    double* ptr_result = static_cast<double*>(buf_result.ptr);

    for (size_t i = 0; i < buf.size; ++i) {
        ptr_result[i] = 1.0 / (1.0 + std::exp(-ptr[i]));
    }

    return result;
}

py::array_t<double> tanh_activation(py::array_t<double> inputs) {
    py::buffer_info buf = inputs.request();
    double* ptr = static_cast<double*>(buf.ptr);

    py::array_t<double> result(buf.size);
    py::buffer_info buf_result = result.request();
    double* ptr_result = static_cast<double*>(buf_result.ptr);

    for (size_t i = 0; i < buf.size; ++i) {
        ptr_result[i] = std::tanh(ptr[i]);
    }

    return result;
}

py::array_t<double> leaky_relu(py::array_t<double> inputs, double alpha = 0.01) {
    py::buffer_info buf = inputs.request();
    double* ptr = static_cast<double*>(buf.ptr);

    py::array_t<double> result(buf.size);
    py::buffer_info buf_result = result.request();
    double* ptr_result = static_cast<double*>(buf_result.ptr);

    for (size_t i = 0; i < buf.size; ++i) {
        ptr_result[i] = ptr[i] > 0 ? ptr[i] : alpha * ptr[i];
    }

    return result;
}

py::array_t<double> softmax(py::array_t<double> inputs) {
    py::buffer_info buf = inputs.request();
    if (buf.ndim != 1)
        throw std::runtime_error("Input should be a 1-D array");

    size_t size = buf.size;
    double* ptr = static_cast<double*>(buf.ptr);

    py::array_t<double> result(size);
    py::buffer_info buf_result = result.request();
    double* ptr_result = static_cast<double*>(buf_result.ptr);

    double max_val = *std::max_element(ptr, ptr + size);
    double sum = 0.0;
    for (size_t i = 0; i < size; ++i) {
        ptr_result[i] = std::exp(ptr[i] - max_val);
        sum += ptr_result[i];
    }
    for (size_t i = 0; i < size; ++i) {
        ptr_result[i] /= sum;
    }

    return result;
}