#pragma once
// BatchNorm-specific validation helpers.
// - Common (device/layout/contiguity/mixed-precision) checks are provided by ai_validate.hpp.
// - This header adds BN domain rules (save_* requirements, affine combos, running_* semantics).

#include "backends/cuda/ops/_common/shim/ai_shim.hpp"   // Status, Tensor, attrs, validate commons
#include "backends/cuda/ops/batchnorm/api.hpp"          // BatchNormAttrs

namespace ai {

// Parsed dims in canonical NCHW order (even if input is NHWC).
struct BNParsed {
  int N, C, H, W;
};

// -----------------------------------------------------------------------------
// Forward common validator (training/inference 공통 부분)
// -----------------------------------------------------------------------------
inline Status BNValidateForward(const Tensor& X,
                                const Tensor& Y,
                                const Tensor* gamma,
                                const Tensor* beta,
                                Tensor* running_mean,
                                Tensor* running_var,
                                const BatchNormAttrs& a,
                                /*out*/ BNParsed& dims)
{
  // Basic shape/layout/contiguity/mixed-precision
  AI_RETURN_IF_ERROR(expect_rowmajor_4d(X));
  AI_RETURN_IF_ERROR(expect_rowmajor_4d(Y));
  AI_RETURN_IF_ERROR(expect_same_shape_layout(X, Y));
  AI_RETURN_IF_ERROR(expect_io_mixed_f16_f32(X));
  AI_RETURN_IF_ERROR(expect_io_mixed_f16_f32(Y));

  // Prohibit aliasing between input and output.
  AI_RETURN_IF_ERROR(no_alias(X, Y));

  // Parse C from layout.
  NCHW4 d{};
  AI_RETURN_IF_ERROR(get_dims_4d(X, a.channels_last, d));
  dims = {d.N, d.C, d.H, d.W};

  // Running stats: must be provided; F32 vectors of length C.
  if (!running_mean || !running_var) return Status::MissingInput;
  AI_RETURN_IF_ERROR(expect_rowmajor_1d(*running_mean));
  AI_RETURN_IF_ERROR(expect_rowmajor_1d(*running_var));
  AI_RETURN_IF_ERROR(expect_param_f32(*running_mean));
  AI_RETURN_IF_ERROR(expect_param_f32(*running_var));
  AI_RETURN_IF_ERROR(expect_vec_len(*running_mean, d.C));
  AI_RETURN_IF_ERROR(expect_vec_len(*running_var,  d.C));

  // Affine parameters
  if (a.with_affine) {
    if (!gamma || !beta) return Status::MissingInput;
    AI_RETURN_IF_ERROR(expect_rowmajor_1d(*gamma));
    AI_RETURN_IF_ERROR(expect_rowmajor_1d(*beta));
    AI_RETURN_IF_ERROR(expect_param_f32(*gamma));
    AI_RETURN_IF_ERROR(expect_param_f32(*beta));
    AI_RETURN_IF_ERROR(expect_vec_len(*gamma, d.C));
    AI_RETURN_IF_ERROR(expect_vec_len(*beta,  d.C));
  } else {
    // Disallow accidental passing of gamma/beta when affine is disabled.
    if (gamma || beta) return Status::InvalidArgument;
  }

  return Status::Ok;
}

// -----------------------------------------------------------------------------
// Forward (training) validator
//  - save_mean/save_invstd are required (F32 vectors of length C).
// -----------------------------------------------------------------------------
inline Status BNValidateForwardTraining(const Tensor& X,
                                        const Tensor& Y,
                                        const Tensor* gamma,
                                        const Tensor* beta,
                                        Tensor* running_mean,
                                        Tensor* running_var,
                                        Tensor* save_mean,
                                        Tensor* save_invstd,
                                        const BatchNormAttrs& a,
                                        /*out*/ BNParsed& dims)
{
  if (!a.training) return Status::InvalidArgument;

  AI_RETURN_IF_ERROR(BNValidateForward(X, Y, gamma, beta, running_mean, running_var, a, dims));

  if (!save_mean || !save_invstd) return Status::MissingInput;
  AI_RETURN_IF_ERROR(expect_rowmajor_1d(*save_mean));
  AI_RETURN_IF_ERROR(expect_rowmajor_1d(*save_invstd));
  AI_RETURN_IF_ERROR(expect_param_f32(*save_mean));
  AI_RETURN_IF_ERROR(expect_param_f32(*save_invstd));
  AI_RETURN_IF_ERROR(expect_vec_len(*save_mean,  dims.C));
  AI_RETURN_IF_ERROR(expect_vec_len(*save_invstd,dims.C));

  return Status::Ok;
}

// -----------------------------------------------------------------------------
// Forward (inference) validator
//  - out_invstd buffer is recommended/required for capture-safety.
// -----------------------------------------------------------------------------
inline Status BNValidateForwardInference(const Tensor& X,
                                         const Tensor& Y,
                                         const Tensor* gamma,
                                         const Tensor* beta,
                                         const Tensor* running_mean,
                                         const Tensor* running_var,
                                         Tensor* out_invstd,  // destination for invstd
                                         const BatchNormAttrs& a,
                                         /*out*/ BNParsed& dims)
{
  if (a.training) return Status::InvalidArgument;

  // running_* are const in inference path; cast away const for reuse in common checks.
  AI_RETURN_IF_ERROR(BNValidateForward(X, Y, gamma, beta,
                                       const_cast<Tensor*>(running_mean),
                                       const_cast<Tensor*>(running_var),
                                       a, dims));

  // For CUDA Graph capture-safety, caller should provide a stable buffer for invstd.
  if (!out_invstd) return Status::MissingInput;
  AI_RETURN_IF_ERROR(expect_rowmajor_1d(*out_invstd));
  AI_RETURN_IF_ERROR(expect_param_f32(*out_invstd));
  AI_RETURN_IF_ERROR(expect_vec_len(*out_invstd, dims.C));

  return Status::Ok;
}

// -----------------------------------------------------------------------------
// Backward common validator
// -----------------------------------------------------------------------------
inline Status BNValidateBackward(const Tensor& dY,
                                 const Tensor& X,
                                 const Tensor* gamma,
                                 const Tensor& save_mean,
                                 const Tensor& save_invstd,
                                 const Tensor* dX,
                                 const Tensor* dgamma,
                                 const Tensor* dbeta,
                                 const BatchNormAttrs& a,
                                 /*out*/ BNParsed& dims)
{
  // dY/X basics
  AI_RETURN_IF_ERROR(expect_rowmajor_4d(dY));
  AI_RETURN_IF_ERROR(expect_rowmajor_4d(X));
  AI_RETURN_IF_ERROR(expect_same_shape_layout(dY, X));
  AI_RETURN_IF_ERROR(expect_io_mixed_f16_f32(dY));
  AI_RETURN_IF_ERROR(expect_io_mixed_f16_f32(X));

  // Parse dims
  NCHW4 d{};
  AI_RETURN_IF_ERROR(get_dims_4d(X, a.channels_last, d));
  dims = {d.N, d.C, d.H, d.W};

  // Saved stats from forward (F32 vectors of length C)
  AI_RETURN_IF_ERROR(expect_rowmajor_1d(save_mean));
  AI_RETURN_IF_ERROR(expect_rowmajor_1d(save_invstd));
  AI_RETURN_IF_ERROR(expect_param_f32(save_mean));
  AI_RETURN_IF_ERROR(expect_param_f32(save_invstd));
  AI_RETURN_IF_ERROR(expect_vec_len(save_mean,  d.C));
  AI_RETURN_IF_ERROR(expect_vec_len(save_invstd,d.C));

  // Affine gamma (if enabled)
  if (a.with_affine) {
    if (!gamma) return Status::MissingInput;
    AI_RETURN_IF_ERROR(expect_rowmajor_1d(*gamma));
    AI_RETURN_IF_ERROR(expect_param_f32(*gamma));
    AI_RETURN_IF_ERROR(expect_vec_len(*gamma, d.C));
  } else if (gamma) {
    return Status::InvalidArgument;
  }

  // Optional outputs
  if (dX) {
    AI_RETURN_IF_ERROR(expect_rowmajor_4d(*dX));
    AI_RETURN_IF_ERROR(expect_same_shape_layout(*dX, X));
  }
  if (dgamma) {
    AI_RETURN_IF_ERROR(expect_rowmajor_1d(*dgamma));
    AI_RETURN_IF_ERROR(expect_param_f32(*dgamma));
    AI_RETURN_IF_ERROR(expect_vec_len(*dgamma, d.C));
  }
  if (dbeta) {
    AI_RETURN_IF_ERROR(expect_rowmajor_1d(*dbeta));
    AI_RETURN_IF_ERROR(expect_param_f32(*dbeta));
    AI_RETURN_IF_ERROR(expect_vec_len(*dbeta, d.C));
  }

  return Status::Ok;
}

} // namespace ai
