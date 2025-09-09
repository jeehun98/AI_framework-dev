#pragma once
#include <vector>
#include <cstdint>
#include <cstddef>

namespace modelsel {

/** 인덱스 분할 */
struct IndexSplit {
    std::vector<int> train_idx;
    std::vector<int> val_idx;
};

IndexSplit holdout_split(int N, double val_ratio, uint64_t seed,
                         bool stratified=false, const uint8_t* y=nullptr);

std::vector<IndexSplit> kfold_splits(int N, int k, uint64_t seed,
                                     bool stratified=false, const uint8_t* y=nullptr);

} // namespace modelsel
