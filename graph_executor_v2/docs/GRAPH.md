그래프 캡처 학습 스택 문서
전체 개요
[Model (Sequential, Dense...)]
        │   forward_into / backward_into  (레이어 구현)
        ▼
capture_plan.make_plan_for_sequential()     ← 레이어별 버퍼/WS 사전할당
        │
        ▼
graph_exec.record_step_graph()              ← fwd→loss→bwd→opt CUDA Graph 녹화
        │
        ├── Graph 지원 : graph.instantiate() → gexec
        └── 폴백     : GraphExecLike(launch_fn)
        │
        ▼
TrainGraph(X_buf, y_buf, logits, gexec, stream)
        │
        └─ set_batch(X,y) + launch()        ← 테스트/트레이너에서 호출

graph_executor_v2/graph/capture_plan.py
모듈 목적

Sequential 계열 모델의 그래프 캡처용 사전 할당 계획(CapturePlan)을 만든다.

레이어별 순전파 출력/프리액티베이션 버퍼(y,z), 역전파 버퍼(gA,gW,gB), 백엔드 워크스페이스(work) 를 준비.

주요 타입

PerLayerBufs

name: str — 디버그용 레이어명(L{idx}:{Class}).

y: cp.ndarray — 해당 레이어의 출력 버퍼.

z: Optional[cp.ndarray] — pre-activation 버퍼(활성화가 있을 때).

gA: cp.ndarray — 입력에 대한 기울기 버퍼.

gW: Optional[cp.ndarray] — 가중치 기울기(파라미터 레이어일 때).

gB: Optional[cp.ndarray] — 바이어스 기울기(있을 때).

work: Any — 백엔드 전용 워크스페이스(gemm ws 등) 핸들.

LossBufs

dY: Optional[cp.ndarray] — 로짓에 대한 손실 미분 버퍼(softmax CE일 때).

out_shape: Tuple[int, ...] — 마지막 레이어 출력 shape.

CapturePlan

input_shape: Tuple[int, ...]

per_layer: List[PerLayerBufs]

loss: LossBufs

공개 함수

make_plan_for_sequential(model, input_shape, *, loss_kind="softmax_ce", lt_bytes=(8<<20)) -> CapturePlan

입력: 모델(Sequential 스타일), 입력 shape, 손실 종류, GEMM LT 워크스페이스 바이트 힌트

출력: CapturePlan

동작:

레이어별 compute_output_shape 로 출력 shape 추론.

y/z 버퍼 확보.

파라미터 레이어(Dense)면 gA/gW/gB + gemm.ensure_workspaces 준비.

마지막 출력 shape로 LossBufs.dY 할당(softmax_ce일 때).

에러/주의:

레이어가 compute_output_shape를 제공하지 않거나 실패하면 RuntimeError.

확장 포인트:

Conv/Norm 등 새 레이어 유형에 맞춰 gW/gB 및 work 할당 로직 추가.

graph_executor_v2/graph/graph_exec.py
모듈 목적

한 스텝 학습(fwd→loss→bwd→opt) 을 CUDA Graph에 녹화하고 실행자를 반환.

CUDA Graph 미지원 환경에선 동일 동작을 수행하는 폴백 실행자 제공.

TrainGraph로 고정 입출력(X_buf, y_buf) + 실행 핸들을 패키징.

핵심 내부 유틸

_run_fwd(model, plan, X, stream_ptr)

X에서 시작해 각 레이어의 forward_into 호출.

_run_bwd(model, plan, g_in, stream_ptr)

g_in에서 역순으로 backward_into 호출.

_zero_bwd_buffers(plan)

누적 방지를 위해 gA/gW/gB를 0으로 초기화.

공개 함수

record_step_graph(model, loss_fn, optimizer_step_fn, plan, *, X_buf, y_buf, stream=None)

입력:

model — forward_into/backward_into 지원 모델

loss_fn — forward(logits, y, return_scalar=False) 지원

optimizer_step_fn — 옵티마이저 스텝 콜러블(예: opt.step_into)

plan: CapturePlan

X_buf, y_buf — 고정 입력/라벨 버퍼(그래프 내에서 참조)

stream — 캡처/실행 스트림(없으면 내부 생성)

출력: CUDA GraphExec 또는 GraphExecLike(폴백)

동작:

워밍업 1회: FWD → LOSS → BWD(zero) → OPT 로 커널 파이프/스트림 상태 고정.

지원 시 CUDA Graph 캡처(cp.cuda.graph.capture_stream)로 동일 시퀀스 녹화.

미지원이면 _one_step 람다로 폴백 실행자 생성.

주의:

반드시 X_buf, y_buf를 인자로 받아 시작 → K mismatch 방지.

손실의 return_scalar=False를 통해 (device_loss, dY)를 얻는 설계 전제.

클래스

GraphExecLike

launch() 하나만 있는 간단 대체 실행자. 내부 스트림에 고정.

TrainGraph

속성: X_buf, y_buf, logits

메서드:

set_batch(X, y) — 고정 버퍼에 복사(포인터 불변)

launch() — 캡처된 그래프(or 폴백) 실행

graph_executor_v2/optim/rebind.py
모듈 목적

옵티마이저 비종속으로, 캡처 플랜의 gW/gB 버퍼를 옵티마이저가 읽는 grad 포인터로 재바인딩하는 유틸.

공개 함수

collect_params_from_plan(model, plan) -> List[(param, grad, exempt)]

AdamW가 제공하는 수집 함수가 있으면 우선 사용.

없으면 Dense 휴리스틱: lyr.W ↔ per.gW, lyr.b ↔ per.gB로 매칭.

try_rebind_grads(model, optimizer, plan) -> None

옵티마이저가 rebind_grads를 지원하면 triplet으로 재바인딩 수행.

주의/확장

다른 옵티마이저(SGD 등)도 rebind_grads만 구현하면 재바인딩 경로에 그대로 편입됨.

레이어가 W/b 외 파라미터를 갖는다면, 해당 레이어 타입에 맞춘 매칭 로직을 추가.

graph_executor_v2/losses/utils.py
모듈 목적

손실의 dY 스케일이 batch-sum인지 batch-mean인지 추정하여 권장 grad_scale 을 리턴.

공개 함수

infer_grad_scale(loss_fn, model, X, y) -> (float, "sum"|"mean")

sum(|dY|)가 배치 크기에 비례하면 1/B, 아니면 1.0.

로그로 스케일 확인, learning rate/grad_norm 튜닝에 활용.

graph_executor_v2/utils/streams.py
모듈 목적

CuPy 스트림 기반 간단 타이머.

클래스

Timer(stream=None)

with Timer(stream) as t: ...; t.ms() 형태로 구간 ms 측정.

Event.record + get_elapsed_time 사용.

graph_executor_v2/train/cuda_graph_trainer.py
모듈 목적

사용자 입장에서 최소 호출로 그래프 캡처 학습을 돌릴 수 있는 E2E 트레이너.

compile() 한 번 → 이후 one_step(X,y) 반복 호출.

공개 API

CudaGraphTrainer(model, loss_fn, optimizer, *, lt_bytes=(8<<20))

모델/손실/옵티마이저를 주입.

내부에서 스트림 고정.

compile(input_shape: Tuple[int,int]) -> None

model.build → make_plan_for_sequential → try_rebind_grads → record_step_graph

고정 X_buf, y_buf 생성 및 TrainGraph 구축.

one_step(X, y) -> float

고정 버퍼에 복사 후 그래프 실행, 현재 모델 파라미터로 손실 재계산하여 반환.

tg: TrainGraph

필요 시 고정 버퍼 직접 접근/디버깅 가능.

사용 예 (통합 테스트)
loss = SoftmaxCrossEntropy()
model = make_model(M,D,H,C)
opt = AdamWOpt([], lr=1e-3, wd=1e-4)
if hasattr(opt, "ensure_initialized"): opt.ensure_initialized()
if hasattr(opt, "set_grad_scale"):     opt.set_grad_scale(1.0)

trainer = CudaGraphTrainer(model, loss, opt)
trainer.compile((M, D))

L0, _ = loss.forward(model(X), y)
trainer.one_step(X, y)
L1, _ = loss.forward(model(X), y)
print(L0, L1)

graph_executor_v2/layers/sequential.py 의 compile 래퍼
목적

레거시 사용처를 위해 Sequential.compile(...)을 슬림 래퍼로 남김.

내부는 make_plan_for_sequential + try_rebind_grads + record_step_graph 호출.

시그니처
compile(self, input_shape, *, loss, optimizer, lt_bytes=(8<<20), stream=None) -> TrainGraph

동작

build() 확인 → supports_capture() 보장

plan 생성 → 옵티마이저 grad 재바인딩

X_buf, y_buf 생성 → record_step_graph(..., X_buf, y_buf, stream)

TrainGraph 반환

데이터 흐름/형상 규약

입력 X: (B, K)

첫 Dense W: (K, H) → K 일치 필수

출력 y(logits): (B, C)

Loss(softmax_ce):

forward(logits, y, return_scalar=False) → (device_loss_scalar, dY)

캡처 그래프에서는 고정 y_buf 를 전달

에러/주의 사항 요약

K mismatch: record_step_graph 캡처 시, 반드시 **고정 입력 X_buf**를 넘겨야 함(수정 완료).

compute_output_shape 실패: 새 레이어는 해당 메서드 정확히 구현 필요.

forward_into/backward_into 미구현: supports_capture()에서 막힘. 각 레이어 구현 확인.

옵티마이저: rebind_grads 미구현이면 동작은 하지만 plan의 gW/gB → layer.grad 복사가 필요할 수 있음(현재 경로는 plan 버퍼를 직접 참조하도록 구성되었고, AdamW에선 제공됨).

확장 포인트

레이어 추가: capture_plan.make_plan_for_sequential에 해당 레이어의 파라미터/버퍼/WS 로직만 추가하면 동일 경로로 캡처 가능.

다른 손실: loss_kind 분기/LossBufs 확장. MSE 등은 dY shape가 로짓 shape와 동일한지만 보장하면 됨.

다중 입력 모델: X_buf를 dict/tuple로 일반화하고 _run_fwd에서 레이어별 forward_into 시그니처를 확장.

테스트/벤치마크 팁

정상성 체크: 그래프 1스텝 후 loss_after < loss_before.

업데이트 방향: Σ(Δθ·g) < 0 확인(옵션).

성능: 동일 스트림에서 set_batch→launch 하도록 유지. H→D 복사와 실행이 한 스트림에 직렬화되어 오버헤드 최소화.

타이밍: utils.streams.Timer로 구간 측정. CUDA 이벤트 기반.