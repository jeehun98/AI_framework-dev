#pragma once

// BiasMode(및 ActKind 등)은 traits.hpp가 단일 소스입니다.
#include "backends/cuda/ops/gemm/detail/traits.hpp"

// 이 헤더는 과거 BiasMode를 재정의했는데, 중복 정의를 제거하기 위해
// 지금은 traits.hpp를 끌어오는 얇은 어댑터로만 유지합니다.

// (선택) 여기서 편의 alias나 얇은 변환 헬퍼를 둘 수 있지만
// BiasMode 자체를 재선언/재정의하면 안 됩니다.
// 예)
// namespace regemm {
//   using ::regemm::BiasMode; // (불필요하지만 명시적으로 보여주고 싶다면)
// }
