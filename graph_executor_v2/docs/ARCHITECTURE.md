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
