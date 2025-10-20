#pragma once
#include <type_traits>
#include <cuda_fp16.h>

namespace epi {

template <typename T> struct IsSupported : std::false_type {};
template <> struct IsSupported<float> : std::true_type {};
template <> struct IsSupported<half>  : std::true_type {};

} // namespace epi
