#pragma once
#include <cstddef>
#include <cstdint>

#ifndef MODELSEL_API
#  if defined(_WIN32) && defined(modelsel_core_EXPORTS)
#    define MODELSEL_API __declspec(dllexport)
#  elif defined(_WIN32)
#    define MODELSEL_API __declspec(dllimport)
#  else
#    define MODELSEL_API
#  endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    double mean, stdev;
} ms_scalar2;

typedef struct {
    ms_scalar2 logloss;
    ms_scalar2 acc;
    int k;
} ms_cv_result;

MODELSEL_API
ms_cv_result ms_kfold_bernoulli(const uint8_t* y, int N, int kfold,
                                unsigned long long seed,
                                int stratified, const char* backend,
                                int use_map, double alpha, double beta);

#ifdef __cplusplus
}
#endif
