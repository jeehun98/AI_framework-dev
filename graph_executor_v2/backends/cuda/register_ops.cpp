// backends/cuda/register_ops.cpp
#include "ai/dispatch.hpp"
#include "ai/op_schema.hpp"

namespace ai {

// launcher.cu에서 제공
Status GemmCudaLaunch(const Tensor&, const Tensor&, const Tensor*, Tensor&,
                      const GemmAttrs&, StreamHandle);

static bool _reg = []{
  auto& R = OpRegistry::inst();

  const ActKind acts[] = {
    ActKind::None, ActKind::ReLU, ActKind::LeakyReLU,
    ActKind::GELU, ActKind::Sigmoid, ActKind::Tanh
  };

  // F32, RowMajor, with_bias/no_bias 모두 등록
  for (bool wb : {false, true}) {
    for (auto a : acts) {
      OpKey k{OpKind::GEMM, Device::CUDA, DType::F32, Layout::RowMajor, a, wb};
      R.reg(k, &GemmCudaLaunch);
    }
  }

  // (선택) F16 등은 나중에 추가
  return true;
}();

} // namespace ai
