#include <vector>
#include <cmath>
#include <stdexcept>

double mean_squared_error(const std::vector<double>& y_true, const std::vector<double>& y_pred){
    if(y_true.size() != y_pred.size()){
        throw std::invalid_argument("Vectors y_true and y_pred must have the same length");
    }

    double mse = 0.0;
    for (size_t i = 0; i < y_true.size(); ++i){
        double diff = y_true[i] - y_pred[i];
        mse += diff * diff;
    }
    return mse / y_true.size();
}

double cross_entropy_loss(const std::vector<double>& y_true, const std::vector<double>& y_pred){
    if(y_true.size() != y_pred.size()){
        throw std::invalid_argument("Vectors y_true and y_pred must have the same length");
    }

    double loss = 0.0;
    for (size_t i = 0; i < y_true.size(); ++i){
        loss -= y_true[i] * std::log(y_pred[i]) + (1.0 - y_true[i]) * std::log(1.0 - y_pred[i]);
    }
    return loss / y_true.size();   
}