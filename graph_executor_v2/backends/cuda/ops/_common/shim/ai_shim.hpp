// backends/cuda/ops/_common/shim/ai_shim.hpp
#pragma once
// 기존 단일 ai_shim을 완전히 대체하는 집계 헤더 (기능 추가 없음)

// ---- Core (항상 필요) ----
#include "ai_status.hpp"
#include "ai_device.hpp"
#include "ai_stream.hpp"
#include "ai_capture.hpp"
#include "ai_memops.hpp"
#include "ai_cuda_check.hpp"
#include "ai_tensor.hpp"
#include "ai_validate.hpp"
#include "ai_nvtx.hpp"
#include "ai_ops_base.hpp"

// ---- New: 공용 enums/레이아웃/워크스페이스(가벼움) ----
#include "enums.hpp"       // ActKind, BiasKind (ABI: enum class : int)
#include "layout.hpp"      // valid_ld_rowmajor, resolve_ld
#include "workspace.hpp"   // is_workspace_aligned

// ---- 선택(무거움): 커널에서만 필요할 수 있음 ----
// 필요시 이 파일에서 켜도 되지만, kernels.cu에서 직접 include 권장
// #include "activations.hpp"    // apply_act_runtime, act_deriv 등
// #include "bias.hpp"           // expected_bias_elems
// #include "traits.hpp"         // vec_width, pack_t 등 (만들 예정이면)
// #include "vector_io.hpp"      // ldg_vec/stg_vec
// #include "broadcast.hpp"      // PerM/PerN 인덱싱
// #include "epilogue_functors.hpp" // (α·acc+β·C+bias)→act→(dropout)
