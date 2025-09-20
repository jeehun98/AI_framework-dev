// backends/cuda/register_ops.cpp
#include "ai/dispatch.hpp"
#include "ai/op_schema.hpp"

namespace ai {

// CUDA GEMM 런처(우리가 만든 래퍼) 선언
Status GemmCudaLaunch(const Tensor&, const Tensor&, const Tensor*, Tensor&,
                      const GemmAttrs&, StreamHandle);

// 내부 등록 함수
static void do_register_cuda_gemm() {
  auto& R = OpRegistry::inst();

  const ActKind acts[] = {
    ActKind::None, ActKind::ReLU, ActKind::LeakyReLU,
    ActKind::GELU, ActKind::Sigmoid, ActKind::Tanh
  };

  for (bool wb : {false, true}) {
    for (auto a : acts) {
      OpKey k{OpKind::GEMM, Device::CUDA, DType::F32, Layout::RowMajor, a, wb};
      R.reg(k, &GemmCudaLaunch);
    }
  }
}

// 외부에서 반드시 한 번 호출할 공개 함수 (정적 라이브러리 강제 포함 목적)
extern "C"
#ifdef _WIN32
__declspec(dllexport)
#endif
void ai_backend_cuda_register_all() {
  static bool once = false;
  if (!once) { do_register_cuda_gemm(); once = true; }
}

} // namespace ai
