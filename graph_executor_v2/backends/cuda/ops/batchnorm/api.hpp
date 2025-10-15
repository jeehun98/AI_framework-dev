#pragma once

// ----------------------- Build-time includes -----------------------
#ifdef BUILD_STANDALONE_OPS
  #include "backends/cuda/ops/_common/shim/ai_shim.hpp"
#else
  #include "ai/tensor.hpp"
  #include "ai/dispatch.hpp"
#endif

// ==================================================================
// Batch Normalization CUDA API (NCHW / NHWC)
// ------------------------------------------------------------------
// - Training: per-channel mean/var reduction + running_* EMA update
//             + save_* (mean, invstd) outputs for backward.
// - Inference: uses running_* as fixed stats; no reduction/update.
// - Mixed Precision: inputs/outputs may be FP16/FP32; internal
//                   accumulation is FP32 by design.
// - CUDA Graph: capture-safe IF tensor shapes, attrs, and workspace
//               sizes remain invariant and no dynamic allocations.
// ==================================================================

namespace ai {

// ================================ Attributes ================================
/**
 * \brief Attributes for BatchNorm (BN). Compatible with PyTorch semantics.
 *
 * \note Momentum definition follows PyTorch:
 *       running = (1 - momentum) * running + momentum * batch_stat
 * \note Epsilon is added in the denominator: invstd = rsqrt(var + eps)
 * \note Layout: channels_last=false → NCHW, true → NHWC
 */
struct BatchNormAttrs {
  // Data layout
  bool  channels_last{false};   ///< false:NCHW, true:NHWC

  // Numerical / algorithmic knobs
  float eps{1e-5f};             ///< variance stabilization epsilon
  float momentum{0.1f};         ///< EMA coef for running stats (PyTorch style)
  bool  training{true};         ///< true: training path, false: inference

  // Optional affine parameters
  bool  with_affine{true};      ///< use gamma/beta (false → pure normalization)

  // Reduction algorithm
  bool  use_welford{true};      ///< use Welford for improved numeric stability

  // Extension hook (GroupNorm compatibility). For BN, must be 1.
  int   num_groups{1};
};

// ============================ Workspace (Forward) ============================
/**
 * \brief Optional forward workspace for capture-safety and scalable reductions.
 *
 * If provided, buffers must be preallocated by the caller and remain
 * valid for the kernel duration. Sizes must match queries from
 * GetFwdWorkspaceBytes(). Nullptrs are allowed if the implementation
 * does not need them for given shapes/attrs.
 */
struct BatchNormWorkspaceFwd {
  // Partial sums for per-channel reduction:
  //   [0..C-1] : sum(x), [C..2C-1] : sum(x^2)
  float* partial_sums{nullptr};  ///< size: 2*C in contiguous layout
  int    partial_sums_stride{0}; ///< row stride if using 2xC matrix view (0=contiguous)

  // Optional block-level temporary buffer for tiled reductions
  float* blockbuf{nullptr};
  size_t blockbuf_elems{0};
};

// ============================ Workspace (Backward) ===========================
/**
 * \brief Optional backward workspace.
 *
 * Holds partial sums for dgamma/dbeta reductions and (optionally)
 * temporary buffers for dX path (e.g., sums of dY, dY*X_hat).
 */
struct BatchNormWorkspaceBwd {
  // [0..C-1] : dbeta = Σ dY
  // [C..2C-1]: dgamma_partial = Σ (X-μ)*invstd*dY
  float* partial_sums{nullptr};  ///< size: 2*C
  int    partial_sums_stride{0};

  float* tempbuf{nullptr};
  size_t tempbuf_elems{0};
};

// ================================ Contracts =================================
// Shapes:
//   X, Y: [N,C,H,W] if !channels_last, else [N,H,W,C]
//   gamma, beta, running_mean, running_var, save_mean, save_invstd: [C]
// DTypes:
//   X/Y: f16 or f32. gamma/beta/running_*: f32.
//   Internal accumulation: f32.
// Aliasing:
//   Y must not alias X. running_* may be updated in-place during training.
// Affine:
//   if with_affine==false → gamma/beta must be nullptr; backward dgamma/dbeta nullptr recommended.
// Save tensors:
//   training=true → save_mean/save_invstd must be provided (non-null); used by backward.
//   training=false → ignored (nullptr allowed).
// CUDA Graph:
//   No dynamic allocations; fixed workspace sizes; deterministic shapes/attrs.

// ================================ Forward API ================================
/**
 * \brief BatchNorm forward.
 * \param X             Input tensor (NCHW or NHWC).
 * \param gamma         Scale [C] (nullable if !with_affine).
 * \param beta          Shift [C] (nullable if !with_affine).
 * \param running_mean  Running mean [C] (in/out if training; in if inference).
 * \param running_var   Running variance [C] (in/out if training; in if inference).
 * \param Y             Output tensor (same shape/layout as X).
 * \param attrs         Attributes controlling BN behavior.
 * \param stream        CUDA stream.
 * \param save_mean     [C] (required iff attrs.training==true; else may be nullptr).
 * \param save_invstd   [C] (required iff attrs.training==true; else may be nullptr).
 * \param ws_fwd        Optional forward workspace (capture-safe).
 * \return Status::OK on success, else InvalidArgument/NotSupported/CudaError/Unimplemented.
 *
 * \details
 * Training path:
 *   - Compute batch mean/var per channel (Welford if enabled).
 *   - Update running_* with EMA: run = (1 - m)*run + m*batch.
 *   - save_mean/save_invstd are written for backward.
 * Inference path:
 *   - Use running_* to normalize. No reductions or updates.
 */
Status BatchNormCudaLaunch(const Tensor& X,
                           const Tensor* gamma,          // [C] or nullptr if !with_affine
                           const Tensor* beta,           // [C] or nullptr if !with_affine
                           Tensor* running_mean,         // [C] (in/out when training, else in)
                           Tensor* running_var,          // [C] (in/out when training, else in)
                           Tensor& Y,                    // out: same shape as X
                           const BatchNormAttrs& attrs,
                           StreamHandle stream,
                           Tensor* save_mean /*=nullptr*/,     // [C] (required in training)
                           Tensor* save_invstd /*=nullptr*/,   // [C] (required in training)
                           const BatchNormWorkspaceFwd* ws_fwd /*=nullptr*/);

// ================================ Backward API ===============================
/**
 * \brief BatchNorm backward.
 * \param dY         Upstream gradient (same shape/layout as Y).
 * \param X          Forward input (or saved copy).
 * \param gamma      Scale [C] (nullable if !with_affine).
 * \param save_mean  From forward training pass [C].
 * \param save_invstd From forward training pass [C].
 * \param dX         (out, optional) gradient wrt X (nullable to skip).
 * \param dgamma     (out, optional) gradient wrt gamma [C] (nullable).
 * \param dbeta      (out, optional) gradient wrt beta  [C] (nullable).
 * \param attrs      Must match forward attrs except training must have been true.
 * \param stream     CUDA stream.
 * \param ws_bwd     Optional backward workspace.
 * \return Status::OK or an error status.
 *
 * \note If with_affine==false, dgamma/dbeta are meaningless and may be nullptr.
 */
Status BatchNormCudaBackwardLaunch(const Tensor& dY,
                                   const Tensor& X,
                                   const Tensor* gamma,           // [C] or nullptr if !with_affine
                                   const Tensor& save_mean,       // [C]
                                   const Tensor& save_invstd,     // [C]
                                   Tensor* dX,                    // out or nullptr
                                   Tensor* dgamma,                // out or nullptr
                                   Tensor* dbeta,                 // out or nullptr
                                   const BatchNormAttrs& attrs,
                                   StreamHandle stream,
                                   const BatchNormWorkspaceBwd* ws_bwd /*=nullptr*/);

// ============================== Workspace Queries ===========================
/**
 * \brief Return required forward workspace size in bytes for given tensors/attrs.
 *        Returns 0 if no extra workspace is needed.
 */
size_t GetFwdWorkspaceBytes(const Tensor& X, const BatchNormAttrs& attrs);

/**
 * \brief Return required backward workspace size in bytes for given tensors/attrs.
 *        Returns 0 if no extra workspace is needed.
 */
size_t GetBwdWorkspaceBytes(const Tensor& dY, const Tensor& X, const BatchNormAttrs& attrs);

} // namespace ai
