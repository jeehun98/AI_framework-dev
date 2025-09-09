#include "modelsel_c.h"
#include "modelsel/kfold.hpp"
#include "modelsel/dataset.hpp"

namespace ms = modelsel;

ms_cv_result ms_kfold_bernoulli(const uint8_t* y, int N, int kfold,
                                unsigned long long seed,
                                int stratified, const char* backend,
                                int use_map, double alpha, double beta) {
    ms::BernoulliData D;
    D.y.assign(y, y + N);

    auto r = ms::kfold_cv_bernoulli(D, kfold, seed, stratified != 0,
                                    std::string(backend ? backend : "cpu"),
                                    use_map != 0, alpha, beta);

    ms_cv_result out;
    out.k = r.k;
    out.logloss.mean  = r.mean_logloss;
    out.logloss.stdev = r.std_logloss;
    out.acc.mean      = r.mean_acc;
    out.acc.stdev     = r.std_acc;
    return out;
}
