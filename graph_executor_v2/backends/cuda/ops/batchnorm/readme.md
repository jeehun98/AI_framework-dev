🧩 Module: backends/cuda/ops/batchnorm

Batch Normalization (BN) operator implemented as an independent CUDA module.

구성:

📦 backends/cuda/ops/batchnorm
 ┣ 📜 api.hpp
 ┣ 📜 launcher.cu
 ┗ 📜 kernels.cu

1️⃣ Overview

이 모듈은 Batch Normalization (BN) 의 forward / backward 연산을 CUDA 상에서 수행한다.
주요 특징:

NCHW / NHWC 레이아웃 모두 지원 (attrs.channels_last).

학습 / 추론 경로 모두 포함.

혼합 정밀도 지원 구조를 가정하지만, 현재 커널은 FP32 전용.

CUDA Graph 캡처 세이프, 내부 동적 할당 없음.

1 channel = 1 CTA 구조로 단순하고 결정적(deterministic) 연산.

외부에서 주어지는 Tensor 포인터 기반 API, 내부에서 메모리 할당하지 않음.

2️⃣ 파일 구조별 역할
파일	역할 요약
api.hpp	외부에서 사용하는 공식 API 시그니처 및 속성 정의 (BatchNormAttrs, BatchNormCudaLaunch, BatchNormCudaBackwardLaunch)
launcher.cu	shape/type 검증, 각 CUDA kernel 실행 순서 관리, 학습/추론 분기 및 EMA(running stats) 업데이트 수행
kernels.cu	실제 CUDA device-level 계산 커널 정의 (mean/var reduce, normalize+affine, backward gradients)
3️⃣ Data Layout / Precision

입력/출력: FP32 tensor (Tensor.desc.dtype == F32)

내부 누적: FP32 (부분 합, 분산 계산 포함)

레이아웃:

channels_last == false → NCHW ([N,C,H,W])

channels_last == true → NHWC ([N,H,W,C])

텐서 확인 유틸:

is4_f32_cuda(t) → 4D FP32 CUDA tensor

is1_f32_cuda(t) → 1D FP32 CUDA tensor (e.g., mean/var/gamma/beta)

4️⃣ Forward Path (BatchNormCudaLaunch)
경로 선택

학습 모드 (attrs.training = true)

Mean / Var 계산

welford_reduce_meanvar_launcher() 호출

각 채널에 대해 한 CTA(block)가 Σx, Σx² 계산

running_var 버퍼를 임시 var 저장용으로 사용

invstd 계산

compute_invstd_kernel() → invstd = rsqrt(var + eps)

결과 save_invstd에 저장

정규화 + Affine 변환

bn_forward_normalize_affine_launcher()

Y = ((X - mean) * invstd) * gamma + beta

Running statistics 업데이트 (EMA)

bn_update_running_kernel()

running_mean ← (1-m)*running_mean + m*batch_mean

running_var ← (1-m)*running_var + m*batch_var

batch_var는 running_var 버퍼를 그대로 재사용함

**save_mean, save_invstd**는 backward를 위한 출력

추론 모드 (attrs.training = false)

running_var로부터 invstd 계산 (compute_invstd_kernel)

정규화 및 affine 적용 (bn_forward_normalize_affine_launcher)

현재 구현에서는 save_invstd == nullptr이면 MissingInput으로 종료
→ 별도의 임시 버퍼를 내부에서 생성하지 않음 (캡처 세이프 설계)

5️⃣ Backward Path (BatchNormCudaBackwardLaunch)

dgamma, dbeta 계산

bn_backward_reduce_dbeta_dgamma_launcher()

채널별 1 CTA로 다음을 계산:

dbeta[c] += Σ dY

dgamma[c] += Σ dY * x̂ (단, with_affine==true일 때만)

dbeta, dgamma는 런처 호출 전 0으로 초기화되어야 함 (+= 누적)

dX 계산

bn_backward_dx_launcher()

내부 수식:

dyγ = dY * (gamma or 1)
S1 = Σ dyγ
S2 = Σ dyγ * x̂
dX = (1/M) * invstd * (M*dyγ - S1 - x̂*S2)


각 채널마다 CTA 1개로 2-phase 수행 (reduce → write)

6️⃣ Kernel 구성 (kernels.cu)
커널 이름	역할	CTA 구조	Shared Memory	Precision	설명
reduce_mean_var_kernel	X로부터 mean, var 계산	channel당 1 CTA	2×blockDim floats (sum, sumsq)	accum FP64, reduce FP32	biased var (1/M), per-channel reduction
bn_forward_norm_affine_kernel	normalize + affine	channel당 1 CTA	없음	FP32	Y=(X-μ)*invstd*γ+β
bn_bwd_dbeta_dgamma_kernel	dbeta/dgamma 감소	channel당 1 CTA	2×blockDim floats	FP32	dbeta=ΣdY, dgamma=ΣdY*x̂
bn_bwd_dx_kernel	dX 계산	channel당 1 CTA	2×blockDim floats	FP32	(2-phase reduction + write)
bn_update_running_kernel	running stats EMA	grid over C	없음	FP32	(1-m)*running + m*batch
compute_invstd_kernel	invstd 계산	grid over C	없음	FP32	invstd = 1/sqrt(var + eps)

공통 설정

blockDim = 256

gridDim = C (1 CTA per channel)

모든 커널은 cudaGraph capture-safe (동적 메모리 없음)

deterministic (block 수 = C, atomic 없음)

7️⃣ Numerical Behavior

Var 계산: biased (1/M), unbiased(1/(M-1)) 아님

Reduction precision: thread-local FP64, shared FP32

Accumulation order deterministic (1 CTA per channel)

Epsilon: 적용 위치 invstd = 1/sqrt(var + eps)

Running stat update: PyTorch-style momentum

running = (1 - m)*running + m*batch

8️⃣ Supported / Unsupported
항목	지원 상태
Layout	NCHW / NHWC
DType	FP32 only
Mixed precision (FP16/BF16)	⚠️ 구조만 존재, 미구현
Capture-safe execution	✅
GroupNorm (num_groups>1)	❌ (현재 BN only)
Deterministic	✅ (channel당 CTA 1개, atomic 없음)
Gradient wrt running stats	❌ (통계는 EMA로만 업데이트)
Workspace (ws_fwd/ws_bwd)	구조만 존재, 실제 사용 없음
9️⃣ Typical Kernel Launch Flow (Training)
# in launcher.cu
--------------------------------------------
welford_reduce_meanvar_launcher()      # mean,var
compute_invstd_kernel()                # invstd = rsqrt(var + eps)
bn_forward_normalize_affine_launcher() # normalize + affine
bn_update_running_kernel()             # EMA update
--------------------------------------------

🔟 API 규약 요약 (api.hpp)
항목	의미
BatchNormAttrs::channels_last	true → NHWC, false → NCHW
eps	분산 안정화 ε
momentum	running stats EMA 계수
training	true: 학습, false: 추론
with_affine	gamma/beta 적용 여부
use_welford	수치 안정성 옵션 (현재 Σx² 구현)
BatchNormWorkspaceFwd/Bwd	선택적 버퍼 (현재 사용 안 함)
Status::Ok	정상 수행
기타 Status::*	shape/dtype mismatch, missing input 등 오류
11️⃣ Known Behaviors / Limitations (현재 구현 기준)
구분	내용
🔹 Precision	내부 누적 FP32, reduce_mean_var_kernel는 FP64 누적 사용
🔹 Variance	1/M(biased) 방식
🔹 NHWC 성능	C가 큰 경우 메모리 비연속 접근으로 성능 저하 가능
🔹 running_var 재사용	batch var를 running_var 버퍼에 계산 후 EMA update에서 같은 포인터 재사용
🔹 invstd 버퍼	inference 시 save_invstd 없으면 MissingInput 오류
🔹 Mixed precision	FP16/BF16 입력 시 API validation 실패 (is4_f32_cuda)
🔹 Gradient determinism	deterministic (atomic 없음, CTA=C)
🔹 Workspace	구조체 정의만 존재, 실제 미사용
🔹 BN fusion	별도의 Conv-BN-Fuse 기능 없음
12️⃣ CUDA Graph Capture Safety

모든 커널은 cudaMalloc, cudaFree 등 동적 호출 없음

shape, attr, ws 크기가 고정되면 graph-safe

캡처된 그래프는 동일 텐서 shape 재사용 시 그대로 실행 가능

13️⃣ Example Usage (학습)
ai::Tensor X, Y, gamma, beta, running_mean, running_var, save_mean, save_invstd;
ai::BatchNormAttrs attrs;
attrs.training = true;
attrs.with_affine = true;
attrs.channels_last = false;
attrs.eps = 1e-5f;
attrs.momentum = 0.1f;

ai::Status st = ai::BatchNormCudaLaunch(
  X, &gamma, &beta,
  &running_mean, &running_var,
  Y, attrs,
  stream,
  &save_mean, &save_invstd,
  nullptr
);

14️⃣ Example Usage (역전파)
ai::Tensor dY, X, dX, dgamma, dbeta, save_mean, save_invstd;
ai::BatchNormAttrs attrs;
attrs.with_affine = true;
attrs.channels_last = false;

ai::Status st = ai::BatchNormCudaBackwardLaunch(
  dY, X, &gamma,
  save_mean, save_invstd,
  &dX, &dgamma, &dbeta,
  attrs, stream, nullptr
);

15️⃣ 전체 데이터 흐름 요약
Forward (Training)
 ┌────────────┐
 │   Input X  │
 └──────┬─────┘
        ▼
  mean,var reduce  → save_mean,var
        ▼
  invstd = 1/sqrt(var + eps)
        ▼
  Y = ((X - mean)*invstd)*γ + β
        ▼
  update running_mean,var (EMA)
        ▼
     Output Y

Backward
 ┌────────────┐
 │   dY, X    │
 └──────┬─────┘
        ▼
  Σ dY, Σ dY*x̂  → dbeta, dgamma
        ▼
  dX = invstd/M * (M*dyγ - S1 - x̂*S2)
        ▼
     Output dX


✅ 요약

상태: FP32-only, NCHW/NHWC 지원, Capture-safe

구조: Simple, deterministic, 1-CTA-per-channel

주요 특징: External tensor-driven API, no malloc, PyTorch-style momentum

미지원: FP16/BF16, workspace 활용, GroupNorm, Fusion

안정성: 오류처리/검증 철저, 커널 자체는 캡처세이프