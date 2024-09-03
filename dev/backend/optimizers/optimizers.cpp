#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <cmath>
#include <map>

namespace py = pybind11;

class SGD {
public:
    SGD(double learning_rate) : learning_rate(learning_rate) {}

    py::array_t<double> update(py::array_t<double> weights, py::array_t<double> gradients) {
        // 버퍼 정보 가져오기
        py::buffer_info buf_weights = weights.request();
        py::buffer_info buf_gradients = gradients.request();

        // 입력 크기 확인
        if (buf_weights.size != buf_gradients.size) {
            throw std::invalid_argument("Weights and gradients must have the same size");
        }

        // 결과 배열 생성
        py::array_t<double> updated_weights(buf_weights.size);
        py::buffer_info buf_updated_weights = updated_weights.request();

        double* ptr_weights = static_cast<double*>(buf_weights.ptr);
        double* ptr_gradients = static_cast<double*>(buf_gradients.ptr);
        double* ptr_updated_weights = static_cast<double*>(buf_updated_weights.ptr);

        // 업데이트 수행
        for (size_t i = 0; i < buf_weights.size; ++i) {
            ptr_updated_weights[i] = ptr_weights[i] - learning_rate * ptr_gradients[i];
        }

        return updated_weights;
    }

private:
    double learning_rate;
};

class Adam {
public:
    Adam(double learning_rate, double beta1 = 0.9, double beta2 = 0.999, double epsilon = 1e-8)
        : learning_rate(learning_rate), beta1(beta1), beta2(beta2), epsilon(epsilon), t(0) {}

    py::array_t<double> update(py::array_t<double>& weights, py::array_t<double>& gradients) {
        // 버퍼 정보 가져오기
        py::buffer_info buf_weights = weights.request();
        py::buffer_info buf_gradients = gradients.request();

        // 입력 크기 확인
        if (buf_weights.size != buf_gradients.size) {
            throw std::invalid_argument("Weights and gradients must have the same size");
        }

        double* ptr_weights = static_cast<double*>(buf_weights.ptr);
        double* ptr_gradients = static_cast<double*>(buf_gradients.ptr);

        t += 1;
        for (size_t i = 0; i < buf_weights.size; ++i) {
            m[i] = beta1 * m[i] + (1.0 - beta1) * ptr_gradients[i];
            v[i] = beta2 * v[i] + (1.0 - beta2) * ptr_gradients[i] * ptr_gradients[i];

            double m_hat = m[i] / (1.0 - std::pow(beta1, t));
            double v_hat = v[i] / (1.0 - std::pow(beta2, t));

            ptr_weights[i] -= learning_rate * m_hat / (std::sqrt(v_hat) + epsilon);
        }

        return weights;
    }

private:
    double learning_rate;
    double beta1;
    double beta2;
    double epsilon;
    int t;
    std::map<size_t, double> m; // 1st moment vector
    std::map<size_t, double> v; // 2nd moment vector
};
