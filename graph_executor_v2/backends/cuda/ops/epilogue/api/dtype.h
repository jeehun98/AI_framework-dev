#pragma once
#include <cstdint>
#include <cuda_fp16.h>

namespace epi {

enum class DType : int32_t { F16 = 0, F32 = 1 };

template <typename T> struct DTypeOf;
template <> struct DTypeOf<half> { static constexpr DType value = DType::F16; };
template <> struct DTypeOf<float> { static constexpr DType value = DType::F32; };

template <DType> struct CTypeOf;
template <> struct CTypeOf<DType::F16> { using type = half; };
template <> struct CTypeOf<DType::F32> { using type = float; };

} // namespace epi
