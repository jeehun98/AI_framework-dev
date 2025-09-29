// backends/cuda/register_ops.cpp
#include <mutex>
#include "backends/cuda/ops/_common/shim/ai_shim.hpp"

#include "backends/cuda/ops/gemm/api.hpp"  // ← 직접 원형 선언 대신 헤더로

namespace ai {

// =======================
// CUDA GEMM Forward 등록
// =======================
static void do_register_cuda_gemm_fwd() {
  auto& R = OpRegistry::inst();

  constexpr ActKind acts[] = {
    ActKind::None, ActKind::ReLU, ActKind::LeakyReLU,
    ActKind::GELU, ActKind::Sigmoid, ActKind::Tanh
  };

  // 현재 지원: F32 / RowMajor / No-Transpose
  for (bool with_bias : {false, true}) {
    for (auto a : acts) {
      OpKey k{
        OpKind::GEMM,
        Device::CUDA,
        DType::F32,
        Layout::RowMajor,
        a,
        with_bias
      };
      R.reg(k, &GemmCudaLaunch);
    }
  }
}

// =======================
// CUDA GEMM Backward 등록
// =======================
static void do_register_cuda_gemm_bwd() {
  // BWD 전용 레지스트리가 있다면 그것을 사용
  auto& Rb = BwdOpRegistry::inst();

  constexpr ActKind acts[] = {
    ActKind::None, ActKind::ReLU, ActKind::LeakyReLU,
    ActKind::GELU, ActKind::Sigmoid, ActKind::Tanh
  };

  for (bool with_bias : {false, true}) {
    for (auto a : acts) {
      BwdOpKey k{
        OpKind::GEMM_BWD,   // 프로젝트에 이 키가 없다면 OpKind::GEMM + dir=Backward로 대체
        Device::CUDA,
        DType::F32,
        Layout::RowMajor,
        a,
        with_bias
      };
      Rb.reg(k, &GemmCudaBackward);
    }
  }
}

// =======================
// 외부에서 반드시 한 번 호출 (집계 진입점에서 호출)
// =======================
extern "C"
#ifdef _WIN32
__declspec(dllexport)
#endif
void ai_backend_cuda_register_all() {
  static std::once_flag once;
  std::call_once(once, [] {
    do_register_cuda_gemm_fwd();
    do_register_cuda_gemm_bwd();
  });
}

} // namespace ai
