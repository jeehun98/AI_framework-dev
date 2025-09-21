// backends/cuda/register_ops.cpp
#include "ai/dispatch.hpp"
#include "ai/op_schema.hpp"

namespace ai {

// =======================
// CUDA GEMM Forward 등록
// =======================

// 우리가 만든 FWD 런처(래퍼)
Status GemmCudaLaunch(const Tensor&, const Tensor&, const Tensor*, Tensor&,
                      const GemmAttrs&, StreamHandle);

static void do_register_cuda_gemm_fwd() {
  auto& R = OpRegistry::inst();

  const ActKind acts[] = {
    ActKind::None, ActKind::ReLU, ActKind::LeakyReLU,
    ActKind::GELU, ActKind::Sigmoid, ActKind::Tanh
  };

  for (bool with_bias : {false, true}) {
    for (auto a : acts) {
      OpKey k{OpKind::GEMM, Device::CUDA, DType::F32, Layout::RowMajor, a, with_bias};
      R.reg(k, &GemmCudaLaunch);
    }
  }
}

// =======================
// CUDA GEMM Backward 등록
// =======================

// 우리가 만든 BWD 런처(래퍼)
// 시그니처는 backward.cu의 구현과 일치해야 합니다.
Status GemmCudaBackward(const Tensor& A, const Tensor& B, const Tensor* C,
                        const Tensor& gY, const Tensor& Z,
                        Tensor* gA, Tensor* gB, Tensor* gC, Tensor* gBias,
                        const GemmAttrs& attrs, StreamHandle stream);

static void do_register_cuda_gemm_bwd() {
  // BWD 전용 레지스트리 (dispatch.hpp에 선언되어 있어야 함)
  auto& Rb = BwdOpRegistry::inst();

  const ActKind acts[] = {
    ActKind::None, ActKind::ReLU, ActKind::LeakyReLU,
    ActKind::GELU, ActKind::Sigmoid, ActKind::Tanh
  };

  // with_bias는 gBias 축적 유무에 따라 true/false 모두 등록(간단 매칭용)
  for (bool with_bias : {false, true}) {
    for (auto a : acts) {
      BwdOpKey k{
        OpKind::GEMM_BWD,      // ← BWD 구분
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
// 외부에서 반드시 한 번 호출
// (정적 라이브러리 강제 포함 목적)
// =======================
extern "C"
#ifdef _WIN32
__declspec(dllexport)
#endif
void ai_backend_cuda_register_all() {
  static bool once = false;
  if (!once) {
    do_register_cuda_gemm_fwd();
    do_register_cuda_gemm_bwd();
    once = true;
  }
}

} // namespace ai
