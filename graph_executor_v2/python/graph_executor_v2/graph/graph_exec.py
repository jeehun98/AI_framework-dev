# File: python/graph_executor_v2/graph/graph_exec.py
from __future__ import annotations
from typing import Any, Optional, Sequence
import cupy as cp

from .capture_plan import CapturePlan
from .execution_planner import ExecPlanner, ExecPlan
from .runtime import GraphRuntime  # run_step 해석/실행 담당

# Conv2D / WS 유틸
from graph_executor_v2.layers.conv2d import Conv2D
from graph_executor_v2.ops import conv2d as convops

# (선택) BN2d 타입 감지용
try:
    from graph_executor_v2.layers.batchnorm import BatchNorm2d as _BN2d
except Exception:
    _BN2d = None

# ===== NVTX (optional) =====
try:
    from graph_executor_v2.backends.cuda.ops.gemm.detail.nvtx_shim import nvtx_range  # type: ignore
except Exception:
    class _DummyNvtx:
        def __call__(self, *_a, **_k):
            class _Ctx:
                def __enter__(self): return None
                def __exit__(self, *args): return False
            return _Ctx()
    nvtx_range = _DummyNvtx()  # type: ignore


class GraphExecLike:
    """CUDA Graph 미사용(또는 불가) 환경에서의 폴백 실행자.

    - graphExec(instantiated graph)와 인터페이스를 맞춘다.
    - 내부에 한 스텝을 수행하는 클로저(_launch)를 보관하고 .launch(stream_ptr)로 호출한다.
    """
    def __init__(self, launch_fn, stream: cp.cuda.Stream):
        self._launch = launch_fn
        self._stream = stream

    def launch(self, stream_ptr=None):
        # stream_ptr는 호환성용 인자(무시). 내부에서 고정 스트림 사용.
        with self._stream:
            self._launch()


# ---------------- record / instantiate ----------------
def record_step_graph(
    model,
    loss_fn,
    optimizer_step_fn,
    plan: CapturePlan,
    * ,
    X_buf: cp.ndarray,
    y_buf: cp.ndarray,
    stream: Optional[cp.cuda.Stream] = None,
    loss_out: Optional[cp.ndarray] = None,          # ✅ 그래프 내부에서 갱신될 손실 스칼라 버퍼(디바이스, shape=())
    # ---- 확장 인자 (동적 경로/플래너) ----
    layers_override: Optional[Sequence[Any]] = None, # 동적 경로 전개 결과(없으면 model.layers)
    exec_plan: Optional[ExecPlan] = None,            # Execution Planner 결과
):
    """fwd → loss → bwd → opt '한 스텝'을 CUDA Graph로 캡처하여 실행자 반환.

    동작 개요:
      1) (워밍업 1회) 동일 순서로 한 번 실행하여 버퍼/워크스페이스/시그니처를 고정
         - loss_out이 주어졌다면 디바이스 스칼라를 여기에 기록
      2) CUDA Graph 캡처 (지원 시)
         - capture_stream(stream) 구간 안에서 동일 시퀀스를 수행
      3) instantiate() 하여 graphExec 반환
      4) CUDA Graph 미지원이면 GraphExecLike 폴백 반환

    확장 포인트:
      - layers_override: 동적 경로 전개(Sequential._linearize_path)의 레이어 시퀀스 지원
      - exec_plan: Execution Planner 결과(스트림/이벤트 스케줄 등)
        → GraphRuntime가 해석하여 실행 (현재는 선형 스케줄)

    주의:
      - BN2d backward는 X_saved(prev_y) 필요. 첫 레이어가 BN인 경우 대비해
        모델 입력 버퍼를 model._graph_input_buf에 기억해둠.
    """
    if stream is None:
        stream = cp.cuda.Stream(non_blocking=True)

    # ExecPlan 준비(없으면 기본 Planner로 선형 스케줄 생성)
    if exec_plan is None:
        exec_plan = ExecPlanner().build(plan=plan, max_streams=1)
    # CapturePlan에 exec_plan을 연결(런타임에서 참조)
    setattr(plan, "exec_plan", exec_plan)

    # 런타임 준비
    rt = GraphRuntime(stream=stream)

    # 레이어 시퀀스 선택(정적: model.layers / 동적: layers_override)
    layers_seq: Sequence[Any] = layers_override if layers_override is not None else list(getattr(model, "layers", []))
    assert len(layers_seq) == len(plan.per_layer), \
        f"[record_step_graph] layers vs plan length mismatch: {len(layers_seq)} vs {len(plan.per_layer)}"

    # ------ 워밍업 1회 ------
    with nvtx_range("[CAPTURE] warmup"):
        with stream:
            # BN bwd fallback 대비 입력 버퍼 포인터 보관
            setattr(model, "_graph_input_buf", X_buf)
            rt.run_step(
                layers=layers_seq,
                plan=plan,
                loss_fn=loss_fn,
                optimizer_step_fn=optimizer_step_fn,
                X_buf=X_buf,
                y_buf=y_buf,
                loss_out=loss_out,
                capture=False,
            )

    has_graph = hasattr(cp.cuda, "graph") and hasattr(cp.cuda.graph, "capture_stream")

    # ------ CUDA Graph 캡처 ------
    if has_graph:
        with nvtx_range("[CAPTURE] cudaGraphCapture"):
            with stream:
                with cp.cuda.graph.capture_stream(stream) as cap:
                    rt.run_step(
                        layers=layers_seq,
                        plan=plan,
                        loss_fn=loss_fn,
                        optimizer_step_fn=optimizer_step_fn,
                        X_buf=X_buf,
                        y_buf=y_buf,
                        loss_out=loss_out,
                        capture=True,
                    )
        gexec = cap.graph.instantiate()
        return gexec

    # ------ 폴백 (그래프 미지원) ------
    def _one_step():
        rt.run_step(
            layers=layers_seq,
            plan=plan,
            loss_fn=loss_fn,
            optimizer_step_fn=optimizer_step_fn,
            X_buf=X_buf,
            y_buf=y_buf,
            loss_out=loss_out,
            capture=False,
        )

    return GraphExecLike(_one_step, stream)


class TrainGraph:
    """캡처된 그래프 실행자 + I/O 버퍼 묶음.

    - set_batch(): 호스트/다른 디바이스 텐서를 고정 I/O 버퍼로 복사
    - launch(): CUDA Graph 인스턴스(or 폴백)의 .launch 호출
    """
    def __init__(self, gexec, io, stream):
        self._gexec = gexec
        self._io = io
        self._stream = stream

    @property
    def logits(self):
        return self._io["logits"]

    @property
    def X_buf(self):
        return self._io["X"]

    @property
    def y_buf(self):
        return self._io["y"]

    def set_batch(self, X_dev, y_dev):
        """현재 배치를 고정 I/O 버퍼(X/y)에 복사 (그래프와 동일 스트림)."""
        xb, yb = self._io["X"], self._io["y"]
        with self._stream:  # ✅ 그래프와 동일 스트림에서 H2D/D2D 수행
            xb[...] = cp.asarray(X_dev, dtype=xb.dtype)
            yb[...] = cp.asarray(y_dev, dtype=yb.dtype)

    def launch(self):
        """CUDA Graph 인스턴스(or 폴백) 실행."""
        self._gexec.launch(self._stream.ptr)
