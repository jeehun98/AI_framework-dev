#pragma once
#include "ai/tensor.hpp"
#include "ai/dispatch.hpp"

namespace ai {

enum class UnaryOp : int {
  Identity = 0,
  ReLU,
  LeakyReLU,   // alpha 사용
  Sigmoid,
  Tanh,
  GELU,        // tanh 근사
  Exp,
  Log,
  Sqrt,
  Rsqrt,
  Clip,        // clip_min, clip_max 사용
};

enum class BinaryOp : int {
  Add = 0,
  Sub,
  Mul,
  Div,
  Max,
  Min,
  Pow,
};

struct EWiseUnaryAttrs {
  float alpha{0.01f};     // LeakyReLU slope 등
  float clip_min{-1e30f};
  float clip_max{ 1e30f};
  float eps{1e-12f};      // log/div 안전용
};

struct EWiseBinaryAttrs {
  float alpha{1.0f};      // 선택적 스칼라 전처리: Y = op(alpha*A, beta*B)
  float beta{1.0f};
  float eps{1e-12f};
};

Status EWiseUnaryCudaLaunch (const Tensor& X, Tensor& Y,
                             UnaryOp op, const EWiseUnaryAttrs& attrs,
                             StreamHandle stream);

Status EWiseBinaryCudaLaunch(const Tensor& A, const Tensor& B, Tensor& Y,
                             BinaryOp op, const EWiseBinaryAttrs& attrs,
                             StreamHandle stream);

} // namespace ai
