#include <iostream>
#include <vector>
#include <cmath>

double relu(double x){
    return x > 0 ? x : 0;
}

double sigmoid(double x){
    return 1.0 / (1.0 + std::exp(-x));
}

double tanh_activation(double x){
    return std::tanh(x);
}

double leaky_relu(double x, double alpha = 0.01){
    return x > 0 ? x : alpha * x;
}

std::vector<double> softmax(const std::vector<double>& inputs){
    std::vector<double> exp_values;
    double sum = 0.0;

    for (double x : inputs){
        exp_values.push_back(std::exp(x));
        sum += std::exp(x);
    }

    for (double& x : exp_values){
        x /= sum;
    }

    return exp_values;
}