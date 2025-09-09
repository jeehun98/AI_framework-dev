#include "modelsel/split.hpp"
#include <random>
#include <algorithm>

namespace modelsel {

IndexSplit holdout_split(int N, double val_ratio, uint64_t seed,
                         bool /*stratified*/, const uint8_t* /*y*/) {
    std::vector<int> idx(N);
    for (int i=0;i<N;++i) idx[i]=i;
    std::mt19937_64 rng(seed);
    std::shuffle(idx.begin(), idx.end(), rng);

    int val_size = static_cast<int>(N * val_ratio);
    IndexSplit sp;
    sp.val_idx.insert(sp.val_idx.end(), idx.begin(), idx.begin()+val_size);
    sp.train_idx.insert(sp.train_idx.end(), idx.begin()+val_size, idx.end());
    return sp;
}

std::vector<IndexSplit> kfold_splits(int N, int k, uint64_t seed,
                                     bool /*stratified*/, const uint8_t* /*y*/) {
    std::vector<int> idx(N);
    for (int i=0;i<N;++i) idx[i]=i;
    std::mt19937_64 rng(seed);
    std::shuffle(idx.begin(), idx.end(), rng);

    std::vector<IndexSplit> folds;
    int fold_size = (k>0) ? (N / k) : N;
    for (int i=0;i<k;++i) {
        IndexSplit sp;
        int start = i * fold_size;
        int end   = (i==k-1) ? N : (i+1) * fold_size;
        sp.val_idx.insert(sp.val_idx.end(), idx.begin()+start, idx.begin()+end);
        sp.train_idx.insert(sp.train_idx.end(), idx.begin(), idx.begin()+start);
        sp.train_idx.insert(sp.train_idx.end(), idx.begin()+end, idx.end());
        folds.push_back(std::move(sp));
    }
    return folds;
}

} // namespace modelsel
