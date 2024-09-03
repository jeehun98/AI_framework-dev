// optimizers.cpp
#include <vector>
#include <cmath>
#include <map>
#include <string>

class SGD {
public:
    SGD(double learning_rate) : learning_rate(learning_rate) {}

    std::vector<double> update(const std::vector<double>& weights, const std::vector<double>& gradients) {
        std::vector<double> updated_weights(weights.size());
        for (size_t i = 0; i < weights.size(); ++i) {
            updated_weights[i] = weights[i] - learning_rate * gradients[i];
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

    std::vector<double> update(std::vector<double>& weights, const std::vector<double>& gradients) {
        t += 1;
        for (size_t i = 0; i < weights.size(); ++i) {
            m[i] = beta1 * m[i] + (1.0 - beta1) * gradients[i];
            v[i] = beta2 * v[i] + (1.0 - beta2) * gradients[i] * gradients[i];

            double m_hat = m[i] / (1.0 - std::pow(beta1, t));
            double v_hat = v[i] / (1.0 - std::pow(beta2, t));

            weights[i] -= learning_rate * m_hat / (std::sqrt(v_hat) + epsilon);
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
