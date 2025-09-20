# graph_executor_v2 — 아키텍처 개요 & 통합 가이드

> **마지막 업데이트:** 2025-09-19 23:41 KST

이 문서는 기존 `ge_v2` + `regemm_epilogue`를 **단일 실행 프레임워크**로 재구성한 현재 리포 구조와 실행 경로, 확장 방법을 요약합니다. 새 채팅에서 코드 설명할 때 이 문서를 링크/붙여서 참고하세요.

---

## 0) 핵심 요약

- 상위는 **그래프/오토그래드/디스패치(Registry)**, 하위는 **백엔드(CUDA 등) 커널**로 분리.
- 연산별 **Op 스키마(입력/속성/검증)** ↔ **디스패치(키 매칭)** ↔ **백엔드 런처** ↔ **커널**.
- 기존 `regemm` 커널/런처는 **그대로** 사용하고, 상위에서 **래퍼로 파라미터 매핑**.
- (옵션) 레거시 `ge2_launch(name, bufs, stream)` 경로도 **그대로 유지 가능**(브리지 사용 시).

---

## 1) 디렉터리 레이아웃

```
graph_executor_v2/
├─ include/ai/                 # 공용 C++ API
│  ├─ tensor.hpp               # Tensor/TensorDesc/Device/DType/Layout
│  ├─ op_schema.hpp            # ActKind, GemmAttrs(=leaky_slope 포함)
│  ├─ dispatch.hpp             # OpKey/OpRegistry(디스패치)
│  ├─ graph.hpp, autograd.hpp  # (확장 예정)
├─ src/
│  ├─ ops/gemm.cpp             # GEMM 스키마/검증 + gemm_run()
│  ├─ dispatch/registry.cpp    # Registry 구현
│  ├─ dispatch/selector_rules.cpp
│  ├─ executor/*               # 스케줄러/메모리/오토그래드(토대)
│  └─ bindings/py_api.cpp      # 파이썬 바인딩(데모)
├─ backends/cuda/
│  ├─ register_ops.cpp         # 이 백엔드가 지원하는 조합을 레지스트리에 등록
│  └─ ops/gemm/
│     ├─ launcher.cu           # ai::Tensor → regemm::GemmBiasActParamsEx 매핑(FWD)
│     └─ backward.cu           # ai::Tensor → regemm::GemmBiasActBwdParams 매핑(BWD)
├─ kernels/regemm_epilogue/    # 기존 regemm 모듈(그대로 사용)
│  ├─ include/regemm/*.h       # api.h, activations.h, bias.h, config.h, nvtx_shim.h
│  └─ src/*.cu                 # regemm_gemm_bias_act.cu, regemm_backward.cu, launcher.cu
├─ tests/{unit,backend_cuda,python}  # 테스트 스켈레톤
└─ python/graph_executor_v2/_core.pyd  # 파이썬 확장 빌드 산출물
```

---

## 2) 실행 경로 (데이터 플로우)

### Forward (현재 구현)
```
Python → py_api.cpp → ai::ops::gemm_run()
  → OpRegistry.find_best(OpKey{kind=GEMM, dev=CUDA, dtype=F32, layout=RowMajor, act=..., with_bias=...})
    → backends/cuda/ops/gemm/launcher.cu : GemmCudaLaunch(...)
      → regemm::gemm_bias_act_f32_ex(GemmBiasActParamsEx, cudaStream_t)
        → launch_gemm_bias_act_f32_{smoke|tiled}_ex(...) → CUDA kernels
```

### Backward (래퍼 제공, 상위 연결만 하면 됨)
```
Autograd Engine (테이프/노드)에서:
  gZ = gY * act'(Z) (커널 내에서)
  gA = alpha * gZ @ B^T    (cuBLAS)
  gB = alpha * A^T @ gZ    (cuBLAS)
  (옵션) gC = beta * gZ
  (옵션) gBias 축적
→ backends/cuda/ops/gemm/backward.cu : GemmCudaBackward(...)
  → regemm::gemm_bias_act_bwd_f32(GemmBiasActBwdParams, stream)
```

> 현재 상위 스키마에는 `alpha/beta` 노출이 없고 기본값(α=1, β=0)으로 설정되어 있습니다. 필요하면 `GemmAttrs`에 확장하세요.

---

## 3) 디스패치(Registry) 키 & 등록

- **OpKey** = `(OpKind, Device, DType, Layout, ActKind, with_bias)`
- **ActKind**는 `None, ReLU, LeakyReLU, GELU, Sigmoid, Tanh`까지 지원.  
  `GemmAttrs`에 `leaky_slope` 포함(기본 0.01).

**등록 예 — `backends/cuda/register_ops.cpp`:**
```cpp
for (bool wb : {false, true}) {
  for (auto a : {ActKind::None, ActKind::ReLU, ActKind::LeakyReLU,
                 ActKind::GELU, ActKind::Sigmoid, ActKind::Tanh}) {
    R.reg({OpKind::GEMM, Device::CUDA, DType::F32, Layout::RowMajor, a, wb},
          &GemmCudaLaunch);
  }
}
```

**선택 규칙** — `find_best`는 완전일치 우선, 실패 시 `act=None` → `layout=RowMajor` 순으로 폴백.

---

## 4) CUDA 백엔드 래퍼 (핵심)

- `launcher.cu` : `ai::Tensor`/`GemmAttrs` → `regemm::GemmBiasActParamsEx` 매핑 후  
  `regemm::gemm_bias_act_f32_ex(...)` 호출 (**공식 엔트리** 직접 호출).
- `backward.cu` : 동일 컨셉으로 `regemm::gemm_bias_act_bwd_f32(...)` 호출.

**제약(현재 버전)**
- F32, RowMajor, 비전치(`trans_a=false`, `trans_b=false`)만 연결.
- Bias는 1D 텐서 길이로 `Scalar | PerM | PerN` 자동 판정.

---

## 5) 레거시 `ge2_launch` 호환 (선택)

- 필요 시 `ge2_launch(name, bufs, stream)` 브리지를 통해 **기존 이름/버퍼 레이아웃**을 그대로 사용.
- 래퍼에서 bufs와 `params_ex`를 구성해 `ge2_launch("gemm_bias_act_f32_ex", ...)` 호출 → `launch_table.cpp` 분기 재사용.
- 기본 경로는 **브리지 없이** `regemm` 공식 API 직호출(코드 단순/안정).

---

## 6) 오토그래드(연결 가이드)

- Forward에서 **Z stash**가 필요하면 `GemmAttrs`에 `save_preact`/출력 `Z`를 추가하고, `launcher.cu`에서 `p.Z, p.save_preact` 설정.
- Backward 호출 시 입력으로 `A, B, (C), gY, Z`와 출력 `gA, gB, (gC), (gBias)`를 연결.
- `regemm_backward.cu`는 내부에서 `gZ`를 만들고, cuBLAS로 `gA/gB`를 계산.

---

## 7) Python 바인딩(데모)

- `src/bindings/py_api.cpp`에 간단한 `gemm_bias_act(a, b, bias=None, act="relu")` 예제가 있음.
- 실제 제품화에서는 **디바이스/스트림/메모리 소유권**을 명시적으로 처리 권장(현재 코드는 데모).

---

## 8) 빌드 & 링크

- 루트 `CMakeLists.txt`에서:
  - `ai_core` : 디스패치/실행기/ops/파이썬 바인딩
  - `ai_backend_cuda` : `backends/cuda/*` + `kernels/regemm_epilogue/src/*.cu`
  - `target_include_directories(ai_backend_cuda PUBLIC kernels/regemm_epilogue/include include)`
  - CUDA arch, 컴파일 옵션(TF32, fast-math 등)은 GPU에 맞춰 조정.

---

## 9) 확장 방법

### A. 새 연산 추가(예: LayerNorm)
1) `include/ai/op_schema.hpp`에 속성 struct 정의(ε 등)  
2) `src/ops/`에 스키마/검증/shape infer 추가  
3) `backends/<dev>/ops/<op>/`에 런처/커널 또는 기존 커널 래퍼 구현  
4) `backends/<dev>/register_ops.cpp`에서 조합 등록  
5) (옵션) 오토그래드 규칙 정의 → bwd 래퍼 추가

### B. FP16/TC 경로 추가
- `DType::F16` 키로 레지스트리 등록 추가
- `launcher.cu`에서 FP16 파라미터로 매핑 후 `regemm`의 FP16 경로(존재 시) 호출
- 정확도 정책(TF32/FP16/bf16)은 `GemmAttrs`에 플래그로 확장

### C. Multiple layouts / Transpose
- `Layout::ColMajor`/`trans_a/b=true` 조합에 맞춰 leading dim/shape 계산 확장
- 레지스트리 키에 해당 조합 등록

---

## 10) 제약/현재 상태

- ✅ GEMM fwd/bwd(F32, RowMajor, No-Transpose) 동작
- ✅ ActKind: None/ReLU/LeakyReLU/GELU/Sigmoid/Tanh
- ✅ Bias: Scalar/PerM/PerN
- ⏳ `alpha/beta` 상위 노출 미정(기본 α=1, β=0), 필요 시 `GemmAttrs` 확장
- ⏳ Z-stash 상위 노출(Autograd 엔진과 연계) — 옵션 추가 예정
- ⏳ 다른 연산(Conv/Norm/Attention 등) — 스키마→디스패치→백엔드 패턴으로 동일 확장

---

## 11) 용어

- **Op 스키마**: 연산의 타입/shape/속성 정의와 검증 레이어  
- **디스패치(Registry)**: `(op, device, dtype, layout, act, with_bias)` → **함수 포인터** 선택  
- **백엔드**: 디바이스별 구현체(CUDA/CPU/ROCm…)  
- **Epilogue**: GEMM 결과에 bias/activation 등 후처리 적용 단계  
- **Z-stash**: activation 이전의 pre-activation 값을 저장해 backward에서 사용



## 12) elwkdls rbclr / Bias 해석

// backends/cuda/ops/gemm/launcher.cu
//
// 역할:
//  - 상위 ai::Tensor/GemmAttrs를 regemm::GemmBiasActParamsEx로 매핑
//  - (현재) f32, row-major, 비전치 경로 지원
//  - Bias 1D 축 판정: Scalar(1) > PerN(len==N) > PerM(len==M)
//    * M==N인 경우 PerN을 기본으로 우선
//    * N==1 또는 M==1일 때도 Scalar가 먼저 잡히도록 순서가 중요!
//
// 반환코드 약속(0=OK, <0 오류):
//  -1 : 디바이스가 CUDA가 아님
//  -2 : dtype(f32) 불일치
//  -3 : 레이아웃(row-major) 불일치
//  -4 : transpose 경로 미지원
//  -5 : shape 차원 불일치(2D 아님 등)
//  -6 : 행렬 크기 불일치(M,K,N 검사 실패)
//  -7 : leading dim(ld*) 유효성 실패
//  -8 : 정수 변환 범위 초과(int32)
//
// 주의:
//  - stream은 상위에서 void*로 전달되며 여기서 cudaStream_t로 재해석.
//  - regemm EX 파라미터는 Z-stash도 지원하지만, 현재 save_preact=0으로 비활성.
//

#include <cuda_runtime.h>
#include <cstring>
#include <stdexcept>
#include <limits>

#include "ai/tensor.hpp"
#include "ai/dispatch.hpp"
#include "ai/op_schema.hpp"

#include "regemm/api.h"  // GemmBiasActParamsEx / gemm_bias_act_f32_ex

namespace {

// --- 유틸: row-major 2D 텐서의 leading dimension 추론 ---
// 우선 stride[0]이 있으면 그 값을 사용, 없으면 기본 contiguous로 shape[1].
inline int64_t infer_ld_rowmajor_2d(const ai::Tensor& t) {
  if (t.desc.shape.size() != 2) return 0;
  if (t.desc.stride.size() >= 2) return t.desc.stride[0];
  return t.desc.shape[1];
}

// --- ai::ActKind → regemm::ActKind 매핑 ---
inline regemm::ActKind to_regemm_act(ai::ActKind a) {
  using A = ai::ActKind;
  using R = regemm::ActKind;
  switch (a) {
    case A::None:      return R::None;
    case A::ReLU:      return R::ReLU;
    case A::LeakyReLU: return R::LeakyReLU;
    case A::GELU:      return R::GELU;
    case A::Sigmoid:   return R::Sigmoid;
    case A::Tanh:      return R::Tanh;
  }
  return R::None;
}

// --- Bias 축 판정 규칙 ---
//  * 길이 1 ⇒ Scalar (항상 최우선; N==1/M==1 케이스 보호)
//  * 길이==N ⇒ PerN (M==N 동률인 경우에도 PerN 우선)
//  * 길이==M ⇒ PerM
//  * 그 외/2D 이상 ⇒ None (보수적 무시)
inline regemm::BiasKind infer_bias_kind(const ai::Tensor* Bias, int64_t M, int64_t N) {
  if (!Bias || !Bias->data) return regemm::BiasKind::None;
  const auto& d = Bias->desc;
  if (d.shape.size() != 1) return regemm::BiasKind::None;

  const int64_t len = d.shape[0];
  if (len == 1) return regemm::BiasKind::Scalar; // ★ scalar 먼저!
  if (len == N) return regemm::BiasKind::PerN;   // ★ M==N이면 PerN 우선
  if (len == M) return regemm::BiasKind::PerM;
  return regemm::BiasKind::None;
}

// --- int64→int32 안전 변환 체크 ---
// regemm 파라미터는 int32 필드이므로 범위를 초과하면 에러로 처리.
inline bool fits_int32(int64_t x) {
  return x >= std::numeric_limits<int>::min() && x <= std::numeric_limits<int>::max();
}

} // anonymous namespace

namespace ai {

// 0=OK, <0 error
Status GemmCudaLaunch(const Tensor& A, const Tensor& B, const Tensor* Bias,
                      Tensor& Y, const GemmAttrs& attrs, StreamHandle stream) {
  // 1) 기본 가드: 디바이스/타입/레이아웃/transpose 지원여부
  if (!A.is_cuda() || !B.is_cuda() || !Y.is_cuda()) return -1;
  if (A.desc.dtype != DType::F32 || B.desc.dtype != DType::F32 || Y.desc.dtype != DType::F32) return -2;
  if (A.desc.layout != Layout::RowMajor || B.desc.layout != Layout::RowMajor || Y.desc.layout != Layout::RowMajor) return -3;
  if (attrs.trans_a || attrs.trans_b) return -4; // 현재 비전치만 지원

  // 2) shape 검증
  if (A.desc.shape.size()!=2 || B.desc.shape.size()!=2 || Y.desc.shape.size()!=2) return -5;
  const int64_t M = A.desc.shape[0];
  const int64_t K = A.desc.shape[1];
  const int64_t Kb= B.desc.shape[0];
  const int64_t N = B.desc.shape[1];
  if (K!=Kb || Y.desc.shape[0]!=M || Y.desc.shape[1]!=N) return -6;

  // 3) leading dim 추론 및 유효성 체크
  const int64_t lda = infer_ld_rowmajor_2d(A);
  const int64_t ldb = infer_ld_rowmajor_2d(B);
  const int64_t ldd = infer_ld_rowmajor_2d(Y);
  if (lda < K || ldb < N || ldd < N) return -7;

  // 4) regemm 파라미터의 int32 제한 확인
  if (!fits_int32(M) || !fits_int32(N) || !fits_int32(K) ||
      !fits_int32(lda) || !fits_int32(ldb) || !fits_int32(ldd)) {
    return -8;
  }

  // 5) regemm 확장 파라미터 구성
  regemm::GemmBiasActParamsEx p{};
  p.M = static_cast<int>(M);
  p.N = static_cast<int>(N);
  p.K = static_cast<int>(K);

  // A, B, (C 미사용), D
  p.A   = A.data; p.lda = static_cast<int>(lda);
  p.B   = B.data; p.ldb = static_cast<int>(ldb);
  p.C   = nullptr; p.ldc = 0;          // C는 현재 미사용 (beta=0)
  p.D   = Y.data; p.ldd = static_cast<int>(ldd);

  // 스케일 (상위에서 alpha/beta 노출 X → alpha=1, beta=0)
  p.alpha = 1.0f;
  p.beta  = 0.0f;

  // Bias 포인터 + 축 판정
  p.bias      = (Bias && Bias->data) ? Bias->data : nullptr;
  p.bias_kind = infer_bias_kind(Bias, M, N); // ★ 규칙 반영 (Scalar > PerN > PerM)

  // Activation
  p.act         = to_regemm_act(attrs.act);
  p.leaky_slope = attrs.leaky_slope;

  // Z stash (EX 기능) — 현재 비활성. 오토그래드 연동 시 여기 활성화.
  p.Z           = nullptr;
  p.ldZ         = 0;    // 0이면 내부에서 ldd로 간주
  p.save_preact = 0;    // 1이면 pre-activation(Z) 저장

  // 6) 실행 — stream은 void* → cudaStream_t 재해석
  regemm::gemm_bias_act_f32_ex(p, reinterpret_cast<cudaStream_t>(stream));
  return 0;
}

} // namespace ai
