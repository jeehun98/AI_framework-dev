// backends/cuda/ops/_common/shim/ai_shim.hpp
#pragma once
//
// 경량 공용 우산 헤더(ODR/빌드시간 최소화).
// - heavy 헤더(activations/bias/traits/epilogue_functors)는 기본 미포함.
// - 커널/런처에서 필요한 것만 개별 include 권장.
//

// ----- Base (항상 먼저) -----
#include "ai_defs.hpp"      // __host__/__device__/AI_RD 등
#include "ai_status.hpp"    // Status

// ----- Core Types/Enums -----
#include "ai_device.hpp"    // Device/DType/Layout + dtype_size
#include "enums.hpp"        // ActKind, BiasKind (단일 소스)

// ----- Streams & Runtime Guards -----
#include "ai_stream.hpp"    // StreamHandle, as_cuda_stream
#include "ai_cuda_check.hpp"// AI_CUDA_CHECK/TRy/LAUNCH
#include "ai_capture.hpp"   // 캡처 상태/가드
#include "ai_memops.hpp"    // copy_{d2d,h2d,d2h}_async, set_d_async, alloc_d/free_d

// ----- Tensor / Validation -----
#include "numeric.hpp"         // fits_int32
#include "layout.hpp"          // valid_ld_rowmajor, resolve_ld
#include "tensor_layout.hpp"   // infer_ld_rowmajor_2d, validate_z_buffer
#include "ai_tensor.hpp"       // Tensor/Descriptor
#include "ai_validate.hpp"     // is_cuda_f32_rowmajor 등

// ----- Tooling -----
#include "ai_nvtx.hpp"      // NVTX Range/Mark
#include "workspace.hpp"    // is_workspace_aligned

// ----- Ops meta (경량) -----
#include "ai_ops_base.hpp"  // GemmAttrs (ActKind은 enums.hpp 참조)

// ----- Heavy (선택: 커널에서 직접 include 권장) -----
// #include "math_compat.hpp"        // expf_compat/tanhf_compat 등
// #include "activations.hpp"        // act_apply/apply_act_runtime
// #include "bias.hpp"               // load_bias/expected_bias_elems
// #include "traits.hpp"             // Epilogue<...> 정책 템플릿
// #include "epilogue_functors.hpp"  // 실행 시 에필로그 조립
