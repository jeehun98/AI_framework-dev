#pragma once
#include <vector>
#include <cstdint>

namespace modelsel {

/** 간단 데이터 컨테이너 */
struct BernoulliData {
    std::vector<uint8_t> y;   // 0/1
};
struct BinomialData {
    std::vector<int> k;       // successes
    std::vector<int> n;       // trials
};

} // namespace modelsel
