#pragma once
#include <vector>
#include <cmath>

namespace modelsel {

inline double mean(const std::vector<double>& v) {
    if (v.empty()) return 0.0;
    double s = 0.0; for (double x : v) s += x;
    return s / v.size();
}
inline double stdev(const std::vector<double>& v) {
    if (v.size() < 2) return 0.0;
    double m = mean(v), s = 0.0;
    for (double x : v) s += (x - m) * (x - m);
    return std::sqrt(s / (v.size()-1));
}

} // namespace modelsel
