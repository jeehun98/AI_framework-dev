#pragma once
#include "kfold.hpp"
#include <iostream>

namespace modelsel {

inline void print_cv_report(const CVResult& r) {
    std::cout << "[CV] k=" << r.k << "\n";
    std::cout << " LogLoss: " << r.mean_logloss << " += " << r.std_logloss << "\n";
    std::cout << " Accuracy: " << r.mean_acc   << " += " << r.std_acc    << "\n";
}

} // namespace modelsel
