📘 Epilogue 모듈 개발 문서
(GraphExecutor CUDA Backend / Independent Module v0.1)
🧩 개요

epilogue 모듈은 GEMM, Conv2D, RNN 등 주요 연산의 결과에 대해
공통적으로 수행되는 후처리 연산(post-processing) 을
독립적으로 구현하고 테스트하기 위한 단일 CUDA 모듈이다.

즉, 각 연산 모듈(GEMM, Conv, RNN)과 직접 연결되지 않아도
epilogue 자체만으로 “bias + activation + scale + store” 흐름을 완성할 수 있도록 설계함.

🧱 디렉터리 구조
📦 epilogue
 ┣ 📂api                  # ABI 고정용 API 헤더
 ┃ ┣ 📜 dtype.h
 ┃ ┗ 📜 epilogue.h
 ┣ 📂kernels              # CUDA 커널 및 functor 정의
 ┃ ┣ 📜 epilogue_functors.cuh
 ┃ ┣ 📜 epilogue_kernel.cu
 ┃ ┣ 📜 epilogue_params.cuh
 ┃ ┗ 📜 philox.cuh
 ┣ 📂launcher             # 런처 (파라미터 패킹 및 커널 디스패치)
 ┃ ┗ 📜 epilogue_launcher.cu
 ┣ 📂pybind               # (선택) Python 바인딩
 ┃ ┗ 📜 epilogue_bind.cpp
 ┣ 📂tests                # 독립 실행 테스트
 ┃ ┗ 📜 test_epilogue_min.cpp
 ┣ 📜 CMakeLists.txt      # CUDA 빌드 구성
 ┗ 📜 build_guide_and_architecture.md   ← (이 문서)

⚙️ 구현 범위 (v0.1)
✅ 현재 구현된 기능
항목	내용
연산 대상	FP32 (float)
Layout	RowMajor
기능 조합	Bias(PerN) + Activation(ReLU)
커널 구조	단일 CUDA kernel (epilogue_kernel_f32_rowmajor)
호출 경로	C++ 런처(epi::run) → 커널 디스패치
테스트	독립 실행(epi_test.exe) – CPU 결과 검증
빌드	CMake + CUDA 12.6 (VS2022 / Ninja)
🧩 핵심 구조
1️⃣ API 정의 (api/epilogue.h)
namespace epi {
struct Attrs {
  ActKind  act{ActKind::ReLU};
  BiasKind bias{BiasKind::PerN};
  float    alpha{1.f}, beta{0.f};
};

struct Tensors {
  void* x; void* y; const void* bias;
  int M, N; Layout x_layout, y_layout;
  int ld_x, ld_y;
};

struct Plan { Attrs attrs; int sm_target{0}; };

Status run(const Plan& plan, const Tensors& ts,
           DType xdt, DType ydt, DType bdt, void* stream=nullptr);
}


run() API는 이후 다른 연산 모듈(GEMM/Conv 등)에서 그대로 재사용 가능하도록
ABI 안정성을 유지한 형태로 설계됨.

2️⃣ 런처 (launcher/epilogue_launcher.cu)

책임: 입력 포인터/stride/attr를 받아 EpParams로 패킹 후 커널 호출

주의: 커널은 선언(extern "C")만 포함, 정의는 커널 파일에만 존재

extern "C" __global__ void epilogue_kernel_f32_rowmajor(EpParams);


디스패치 흐름

dim3 block(256);
dim3 grid((M*N + 255) / 256);
epilogue_kernel_f32_rowmajor<<<grid,block>>>(params);

3️⃣ 커널 (kernels/epilogue_kernel.cu)

단일 스레드에서 (m,n)별 element-wise 연산 수행
(Bias → Act → Blend → Store)

extern "C" __global__
void epilogue_kernel_f32_rowmajor(EpParams P) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = idx; i < P.M*P.N; i += gridDim.x*blockDim.x) {
    int m = i / P.N, n = i % P.N;
    float v = P.x[m*P.ld_x + n];
    if (P.has_bias) v += P.bias[n];
    if (P.act == 1) v = v > 0.f ? v : 0.f; // ReLU
    P.y[m*P.ld_y + n] = P.alpha * v + P.beta * P.y[m*P.ld_y + n];
  }
}

4️⃣ 파라미터 구조 (kernels/epilogue_params.cuh)
struct EpParams {
  int M, N, ld_x, ld_y;
  const float* x; float* y; const float* bias;
  float alpha, beta;
  uint8_t act, has_bias;
};

5️⃣ CMake 설정

핵심 포인트

STATIC 라이브러리로 구성 (epi)

CUDA_SEPARABLE_COMPILATION + CUDA_RESOLVE_DEVICE_SYMBOLS 활성화

인코딩: /utf-8 (CXX) / -Xcompiler=/utf-8 (CUDA)

set_target_properties(epi PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

🧪 테스트 (tests/test_epilogue_min.cpp)

검증: CPU-side 연산과 GPU 결과 비교

케이스: ReLU + Bias(PerN) 조합

for(int m=0;m<M;++m)
  for(int n=0;n<N;++n){
    float ref = hx[m*N+n] + hb[n];
    ref = ref>0.f ? ref : 0.f;
    if (fabsf(ref - hy[m*N+n]) > 1e-6f) errors++;
  }


결과:

OK. errors=0

🔧 빌드 방법

VS2022 환경 (CUDA 12.6 기준)

chcp 65001
set VSLANG=1033
rmdir /s /q build
cmake -S . -B build -G "Ninja" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build -j
build\epi_test.exe

🚀 향후 확장 로드맵
단계	내용	주요 수정 위치
v0.2	FP16 / BF16 dtype 지원	epilogue_params.cuh, launcher.cu dtype dispatch
v0.3	Dropout (Philox RNG)	epilogue_functors.cuh, philox.cuh
v0.4	Residual / SaveZ 지원	EpParams, ep_apply_scalar()
v0.5	Clamp / Quantization	epilogue_functors.cuh
v0.6	Multi-layout (ColMajor/Strided)	launcher.cu stride 계산
v1.0	GEMM 내부 epilogue hook 연동	regemm_kernel.cu (ep_apply 호출)
💡 설계 의도 요약
목표	설계 포인트
독립성	어떤 모듈에도 종속되지 않고 단독 컴파일/테스트 가능
재사용성	GEMM, RNN, Conv 등 다양한 모듈이 동일 API(epi::run)로 호출 가능
확장성	Functor 기반 조합 (Bias + Act + Dropout + Residual)을 쉽게 추가
CUDA Graph Capture 호환성	모든 파라미터를 POD 구조(EpParams)로 고정
Debug/Benchmark 용이성	단일 모듈로 벤치마킹 및 성능 비교 가능
📍 체크리스트

 커널 정의는 단 하나의 .cu 에만 존재

 런처에서는 선언만 포함 (extern "C")

 CMake에서 STATIC + CUDA_RESOLVE_DEVICE_SYMBOLS ON

 /utf-8 인코딩 분리 적용

 테스트 검증 통과 (errors=0)

현재 버전: epilogue_v0.1
작성자: @쩝쩝박사
목표: GraphExecutor CUDA 백엔드 내 모든 post-op epilogue의 기반 모듈로 사용
다음 단계: FP16 / Dropout / Residual 확장 및 CUTLASS-style functor pattern 적용