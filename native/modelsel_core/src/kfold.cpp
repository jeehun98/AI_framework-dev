#include "modelsel/kfold.hpp"
#include "modelsel/estimator.hpp"
#include "modelsel/evaluator.hpp"
#include "modelsel/metrics.hpp"
#include "modelsel/split.hpp"

namespace modelsel {

CVResult kfold_cv_bernoulli(const BernoulliData& D, int kfold, uint64_t seed,
                            bool stratified, std::string backend,
                            bool use_map, double alpha, double beta) {
    auto splits = kfold_splits((int)D.y.size(), kfold, seed, stratified, D.y.data());

    std::vector<double> loglosses, accs;
    BernoulliEvaluator eval(backend);

    for (const auto& sp : splits) {
        auto params = use_map
            ? bernoulli_map_beta(D, sp.train_idx, alpha, beta)
            : bernoulli_mle     (D, sp.train_idx);

        auto stats = eval.evaluate(D, sp.val_idx, params.p);
        loglosses.push_back(stats.logloss);
        accs.push_back(stats.accuracy);
    }

    CVResult r;
    r.k = kfold;
    r.mean_logloss = mean(loglosses);
    r.std_logloss  = stdev(loglosses);
    r.mean_acc     = mean(accs);
    r.std_acc      = stdev(accs);
    return r;
}

CVResult kfold_cv_binomial(const BinomialData& D, int kfold, uint64_t seed,
                           std::string backend,
                           bool use_map, double alpha, double beta) {
    auto splits = kfold_splits((int)D.k.size(), kfold, seed, /*stratified=*/false, nullptr);

    std::vector<double> loglosses, accs;
    BinomialEvaluator eval(backend);

    for (const auto& sp : splits) {
        auto params = use_map
            ? binomial_map_beta(D, sp.train_idx, alpha, beta)
            : binomial_mle     (D, sp.train_idx);

        auto stats = eval.evaluate(D, sp.val_idx, params.p);
        loglosses.push_back(stats.logloss);
        accs.push_back(stats.accuracy);
    }

    CVResult r;
    r.k = kfold;
    r.mean_logloss = mean(loglosses);
    r.std_logloss  = stdev(loglosses);
    r.mean_acc     = mean(accs);
    r.std_acc      = stdev(accs);
    return r;
}

} // namespace modelsel
