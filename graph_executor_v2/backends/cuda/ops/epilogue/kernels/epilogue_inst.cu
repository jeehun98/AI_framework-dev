#include "epilogue/apply.hpp"

namespace epilogue {
  // 필요 조합만 전개해 바이너리 크기 제어
  template struct Apply<ActKind::None,   BiasKind::None,   false, false, float, float>;
  template struct Apply<ActKind::ReLU,   BiasKind::PerN,   false, false, float, float>;
  template struct Apply<ActKind::GELU,   BiasKind::PerM,   false, true , float, float>;
  // ...필요 시 추가
}
