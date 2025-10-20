    폴더별/파일별 역할 요약
📂api

dtype.h

내부 타입 시스템. DType(F16/F32) 열거와 DTypeOf<T>, CTypeOf<DType> 매핑.

확장 포인트: BF16 추가 시 여기서 DType::BF16 및 매핑을 추가.

epilogue.h

외부 공개 API.

ActKind, Attrs(act, dropout_p, seed, save_mask),

Plan(rows/cols/stride), Tensors(x/bias/y/mask_out),

run(plan, tensors, dtype) 선언.

확장 포인트: 새로운 attr(예: residual blend, alpha/beta)을 추가하고 Plan.attrs로 전달.

epilogue_stub.cpp

run()의 dtype 분기 → 각 dtype 런처로 위임.

주의: 여기선 로직 없음. 실패 코드·입력검증은 커널/디스패치 쪽에서.

📂kernels/policy

ep_apply.cuh

“정책 기반” 파이프라인의 미들웨어.

Policy가 제공하는 BiasF/ActF/DropF/BlendF를 호출해
X + Bias → Act → Dropout → Blend(+ Resid)를 원소/벡터 단위로 실행.

확장 포인트:

새 정책(타일/벡터화/브로드캐스트 방식)을 만들고 EpApply<NewPolicy> 사용.

Dropout off 경로/Residual on/off 등의 컴파일타임 분기(UseDrop, UseResid) 추가.

ep_functors.cuh

활성화/바이어스 처리 펑터 집합. (ActNone/ReLU/GELU, BiasAct 등)

확장 포인트: SiLU/Swish/LeakyReLU 추가, Per-row/Per-head bias 처리용 펑터 추가.

ep_kernel_policy.cuh

런치 구성 헬퍼. compute_grid(), compute_block() 등.

최적화 포인트: grid-stride loop 전환, 2D 타일링, occupancy 기반 TPB 자동화.

ep_math.cuh

수학 유틸/캐스팅. to_f32/from_f32, GELU(ReLU) 스칼라 구현, Math<T>::add.

최적화 포인트: fast-math(+approx), __half2용 벡터 연산 유틸 추가.

ep_policy.cuh

기본 타일·벡터화 정책 상수. TilePolicy::TPB, VEC 등.

최적화 포인트: 하드웨어/문제 크기에 맞춰 값 튜닝, half2 경로 활성화.

ep_traits.cuh

지원 타입 트레이트(IsSupported<T>).

확장 포인트: BF16 지원, 사용자 정의 포맷 추가 시 여기서 허용.

📂kernels

epilogue_params.cuh

디바이스 측 파라미터 집합 EpParams<T>와 make_params().

포인터(x/bias/y/mask), 크기/스트라이드, dropout 설정(seed, p, save_mask) 등.

정책 호환 필드(p_drop/keep_scale/alpha/beta/resid alias) 포함.

확장 포인트: blend/residual, scaling, 다양한 브로드캐스트 형태 필드 추가.

epilogue_kernels_policy.cu

실제 CUDA 커널 본체. kBiasActDropout<T,Act>(혹은 정책화된 호출)에서
로드 → BiasAct → Activation → (Inverted)Dropout(+mask) → 저장.

최적화 포인트:

grid-stride loop, coalesced vector load/store(float4/half2),

마스크 bit-pack(8:1), 분기 제거( predication ),

warp-level primitives 활용(특히 Blend/Residual 결합 시).

epilogue_dispatch.cu

입력 검증, ActKind 분기 → 템플릿 인스턴스 선택.

확장 포인트: 새 활성화/새 dtype 추가 시 case 확장. 입력 스키마 검증 강화.

philox.cuh

RNG 계층. 현재는 **무상태 SplitMix 해시 기반 rand01(seed, idx)**로 결정적·캡처세이프.

확장 포인트:

통계적 보장이 더 필요하면 cuRAND Philox로 교체(링크: CUDA::curand).

seed offset/step counter로 step-wise 랜덤 마스크 생성.

📂launcher

epilogue_launcher_policy.cu

외부에서 호출되는 dtype별 런처 심볼(launch_policy_f32/f16) 구현.

확장 포인트: 여러 정책(벡터화 on/off, bitpack on/off) 변형을 조건부로 노출.

📂pybind

epilogue_pybind.cpp

Python 바인딩. Plan/Attrs/Tensors 노출, run() 호출.

포인터 필드는 정수 주소 프로퍼티로 노출(캡슐 없이 ctypes 포인터 대입 가능).

확장 포인트: 예외/상태 반환 개선, NumPy/CuPy 포인터 래핑 유틸 추가.

📜문서.md

개요/사용 예/빌드 가이드/확장 포인트.

팁: 협업 시 변경 이력(정책 변경, RNG 변경, API 추가)을 여기에 버저닝해두면 좋아.

설계의 핵심 포인트 (확장/최적화 기준)

단일 커널 Fused Epilogue
Bias → Activation → Dropout → (mask store)를 한 런치로 처리. 메모리 왕복/커널 오버헤드 최소화.

정책화 계층(Policy/Functor/Apply)
타일/벡터화/브로드캐스트/활성화/드롭아웃/블렌드/리지듀얼을 조합 가능하게 설계.
→ 새 요구사항은 “펑터/정책”만 추가·교체하면 됨(커널 본체는 그대로 재사용).

무상태 RNG & Graph-capture Safe
rand01(seed, global_idx)로 재현성과 캡처 안정성 확보.
→ 리플레이마다 다른 마스크가 필요하면 seed만 바꾸면 됨.

성능 레버(실전 튜닝 순서)

TPB/grid-stride loop로 occupancy 맞춤

N축 정렬 시 float4/half2 vectorized load/store

조건분기 제거(predication)로 warp divergence 완화

마스크 bit-pack(메모리 대역폭 절감)

활성화 수학식 fast-math/近似(특히 GELU)

(필요 시) cuRAND Philox로 난수 품질 강화

기능 확장 패턴

새 활성화: ep_functors.cuh에 functor 추가 → dispatch.cu에 case 추가

새 브로드캐스트: BiasF 펑터/정책 추가(Per-row/Per-channel/Per-head)

Residual/Blend: EpParams에 필드 추가 → BlendF/UseResid 분기 활성화

새 dtype: dtype.h/traits/dispatch/pybind에 항목 추가