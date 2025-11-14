좋아—아래 순서대로 하면 캡처 → 열기 → NVTX로 범위 해석 → 수치 요약까지 한 번에 정리됩니다. (Windows 기준)

1) 기본 캡처 (당신이 쓴 명령어 그대로)
nsys profile --trace=cuda,nvtx -o run_trace --force-overwrite=true python smoke_static_dynamic_optim_and_pool.py


생성물:

run_trace.nsys-rep : GUI/CLI 분석용 리포트 (필수)

(옵션) run_trace.sqlite : 통계/스크립팅용 DB

경고:

관리자 권한이 아니면 CPU sampling/context switches 비활성 경고가 뜰 수 있음(무시해도 CUDA/NVTX 타임라인에는 영향 없음). 필요 시 PowerShell을 관리자로 실행.

2) 바로 열어보기 (GUI / CLI)
GUI (권장)

실행 파일 경로 예:
C:\Program Files\NVIDIA Corporation\Nsight Systems 20xx.x.x\host-windows-x64\nsys-ui.exe

실행 후 File → Open Report → run_trace.nsys-rep.

탭 탐색 요령:

CUDA GPU Trace: 스트림별 커널 타임라인

NVTX Trace: forward / loss / backward / optimizer 같은 사용자 범위

CUDA Kernel Statistics: 커널별 호출/시간

CUDA API Statistics: API 호출(메모리, 런치, 이벤트) 비용

CLI 요약
nsys stats --report summary run_trace.nsys-rep
nsys stats --report gpukernsum run_trace.nsys-rep
nsys stats --report nvtxsum run_trace.nsys-rep

3) NVTX 태깅 (코드에 구간 라벨 붙이기)

당신 레포에 있는 nvtx_shim(예: graph_executor_v2.backends.cuda.ops.gemm.detail.nvtx_shim)을 쓴다고 가정:

from graph_executor_v2.backends.cuda.ops.gemm.detail.nvtx_shim import nvtx_range

with nvtx_range("forward"):
    out = model(x)

with nvtx_range("loss"):
    loss = criterion(out, y)

with nvtx_range("backward"):
    loss.backward()

with nvtx_range("optimizer"):
    optimizer.step()


팁:

중첩/순서가 명확해야 타임라인이 깔끔합니다(열고 닫고를 반드시 1:1).

CUDA Graph 캡처 내부에서도 NVTX는 캡처 안전하므로 자유롭게 감싸도 됩니다.

레인지 이름은 항상 일관된 텍스트를 사용(통계 집계에 중요).

4) 캡처 범위를 NVTX로 “필요한 구간만” 줄이기

훈련 스크립트가 길거나 warmup이 길면, 전체를 캡처하지 말고 NVTX 구간만 캡처하세요.

(A) 코드에 “캡처용” 범위 표시
with nvtx_range("capture"):   # 이름은 임의이지만 아래 옵션과 일치해야 함
    # warmup 이후 단 1step만 캡처하고 싶다면 여기에 forward~optimizer를 배치
    step_once()

(B) nsys 옵션으로 “범위 기반 캡처”
nsys profile --trace=cuda,nvtx -o run_trace --force-overwrite=true `
  --capture-range=nvtx --capture-range-end=stop `
  python smoke_static_dynamic_optim_and_pool.py


동작:

프로세스 전체가 아니라 NVTX("capture") 범위가 열려 있는 동안만 타임라인을 기록.

범위가 종료되면 자동으로 캡처 종료.

대안: --delay <sec> + --duration <sec> 로 시간 기반 캡처도 가능하지만, NVTX 범위 기반이 훈련 루프에 더 안정적입니다.

5) 더 풍부한 캡처(필요할 때만 추가)

cuBLAS/cuSOLVER/cuSPARSE 호출까지 보고 싶을 때:

nsys profile --trace=cuda,nvtx,cublas,cusolver,cusparse `
  -o run_trace --force-overwrite=true `
  python smoke_static_dynamic_optim_and_pool.py


SQLite까지 내보내기(자동화/스クリ핑):

nsys profile --trace=cuda,nvtx --export=sqlite `
  -o run_trace --force-overwrite=true `
  python smoke_static_dynamic_optim_and_pool.py


stdout을 즉시 보고 싶다면:

nsys profile --trace=cuda,nvtx --show-output=true `
  -o run_trace --force-overwrite=true `
  python smoke_static_dynamic_optim_and_pool.py

6) 무엇을 어디서 볼까 (실전 체크리스트)

GUI “CUDA GPU Trace / NVTX Trace”에서:

forward 구간

GEMM/에필로그/드롭아웃이 연속 1~2개의 커널로 붙어 있는지(분리돼 있으면 fuse 여지)

불필요한 transpose/copy가 끼어있는지

backward 구간

dGEMM + 활성 미분/축약 에필로그로 잘 묶였는지

optimizer 구간

AdamW가 멀티텐서 1~2런치로 묶였는지

빈틈(gap) 존재 여부

커널 사이 간격이 길면 런치 오버헤드/암묵 sync 의심

스트림 동시성

독립 작업이 서로 다른 스트림에서 겹쳐 실행되는지

CLI “요약”에서:

nsys stats --report gpukernsum,nvtxsum run_trace.nsys-rep


커널 상위 Top-N / NVTX 구간별 총합 시간으로 병목 파악.

이전 리포트와 숫자 비교로 튜닝 효과 확인.

7) 반복 실험을 위한 미니 스크립트 (복붙용)
PowerShell: profile.ps1
param(
  [string]$name = "run_trace",
  [switch]$sqlite
)

$export = $sqlite.IsPresent ? "--export=sqlite" : ""
nsys profile --trace=cuda,nvtx $export --force-overwrite=true `
  -o $name python smoke_static_dynamic_optim_and_pool.py

Write-Host "Generated: $PWD\$name.nsys-rep"
if ($sqlite.IsPresent) { Write-Host "Also: $PWD\$name.sqlite" }


사용:

.\profile.ps1 -name run_trace_baseline
.\profile.ps1 -name run_trace_fused -sqlite

Python: NVTX 헬퍼 (자동 레인지)
from contextlib import contextmanager
from graph_executor_v2.backends.cuda.ops.gemm.detail.nvtx_shim import nvtx_range

@contextmanager
def train_step_ranges():
    with nvtx_range("forward"):   yield "forward"
    # 다음으로 loss, backward, optimizer 등은 호출부에서 각각 감싸도 OK

# 사용 예시 
with nvtx_range("capture"):     # --capture-range=nvtx 를 쓸 계획이라면
    with nvtx_range("forward"):
        out = model(x)
    with nvtx_range("loss"):
        loss = criterion(out, y)
    with nvtx_range("backward"):
        loss.backward()
    with nvtx_range("optimizer"):
        optimizer.step()

8) 흔한 이슈 & 해결

nsys-ui 명령을 못 찾음
→ nsys-ui.exe의 경로를 PATH에 추가하거나 exe를 직접 실행
예: "C:\Program Files\NVIDIA Corporation\Nsight Systems 2024.x.x\host-windows-x64\nsys-ui.exe"

CPU 관련 경고
→ 관리자 권한이 아니면 뜹니다. GPU/NVTX 분석만 할 땐 무시해도 됩니다.

리포트가 너무 커짐
→ --capture-range=nvtx 로 캡처 구간 축소, 혹은 --duration 제한.

NVTX가 타임라인에 안 보임
→ --trace에 nvtx 포함했는지, NVTX 범위를 해당 스레드에서 제대로 push/pop 했는지 확인.

9) 한 컷 요약 (Minimal → Advanced)

Minimal

nsys profile --trace=cuda,nvtx -o run_trace --force-overwrite=true `
  python smoke_static_dynamic_optim_and_pool.py


NVTX만 캡처

코드에 with nvtx_range("capture"): 추가

실행:

nsys profile --trace=cuda,nvtx --capture-range=nvtx --capture-range-end=stop `
  -o run_trace --force-overwrite=true `
  python smoke_static_dynamic_optim_and_pool.py


분석(요약 수치)

nsys stats --report gpukernsum,nvtxsum run_trace.nsys-rep


GUI에서 타임라인

nsys-ui.exe → Open run_trace.nsys-rep → NVTX/CUDA Trace

필요하면 여기에 당신 프레임워크(Sequential/TrainGraph) 공용 NVTX 데코레이터를 끼워넣는 패치도 바로 만들어줄게.
(예: one_step_dynamic 내부에서 forward/loss/backward/optimizer를 자동으로 태깅)