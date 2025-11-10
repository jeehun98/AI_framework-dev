// _common/shim/enums.hpp
#pragma once
namespace ai::cuda::shim {
enum class ActKind : int { None=0, ReLU=1, LeakyReLU=2, GELU=3, Sigmoid=4, Tanh=5 };
enum class BiasKind: int { None=0, Scalar=1, PerM=2, PerN=3 };
} // ns
