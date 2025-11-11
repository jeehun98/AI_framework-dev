# File: python/graph_executor_v2/graph/graph_exec.py
from __future__ import annotations
from typing import Any, Optional, Sequence
import os
import cupy as cp

from .capture_plan import CapturePlan
from .execution_planner import ExecPlanner, ExecPlan
from .runtime import GraphRuntime  # run_step 해석/실행 담당

# Conv2D / WS 유틸 (미래 확장용; 현재 파일 내 직접 사용 안 할 수 있음)
from graph_executor_v2.layers.conv2d import Conv2D  # noqa: F401
from graph_executor_v2.ops import conv2d as convops  # noqa: F401

# (선택) BN2d 타입 감지용
try:
    from graph_executor_v2.layers.batchnorm import BatchNorm2d as _BN2d  # noqa: F401
except Exception:
    _BN2d = None  # type: ignore

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
    *,
    X_buf: cp.ndarray,
    y_buf: cp.ndarray,
    stream: Optional[cp.cuda.Stream] = None,
    loss_out: Optional[cp.ndarray] = None,          # ✅ 그래프 내부에서 갱신될 손실 스칼라 버퍼(디바이스, shape=())
    # ---- 확장 인자 (동적 경로/플래너) ----
    layers_override: Optional[Sequence[Any]] = None, # 동적 경로 전개 결과(없으면 model.layers)
    exec_plan: Optional[ExecPlan] = None,            # Execution Planner 결과
    # ---- 메타/디버그 ----
    graph_key: Optional[Any] = None,                 # GraphPool에서 만든 키(있으면 TrainGraph에 전달 가능)
    ctx: Optional[dict] = None,                      # 🔥 NEW: RNG/브랜치 등 런타임 컨텍스트
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

    # 🔥 NEW: RNG 메타 주입(컨텍스트가 오면 plan에 고정)
    try:
        rng = (ctx or {}).get("rng", {}) or {}
        if getattr(plan, "seed", None) is None and "seed" in rng:
            setattr(plan, "seed", int(rng["seed"]))
        if getattr(plan, "rng_step", None) is None and "step" in rng:
            setattr(plan, "rng_step", int(rng["step"]))
    except Exception:
        pass

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

    디버그 표면:
      - 환경변수 GEV2_EXPOSE_DEBUG=1 일 때만 plan/key/tags 노출
      - io 바인딩은 문서/테스트 편의를 위해 항상 읽기용으로 공개
    """
    def __init__(self, gexec, io, stream,
                 *,
                 plan: Optional[CapturePlan] = None,
                 graph_key: Optional[Any] = None,
                 tags: Optional[dict] = None):
        self._gexec = gexec
        self._io = io
        self._stream = stream

        # 디버그/문서용 노출은 게이트로 보호
        self._expose_debug = os.getenv("GEV2_EXPOSE_DEBUG", "0") == "1"
        self._plan = plan if self._expose_debug else None
        self._key = graph_key if self._expose_debug else None

        # 🔎 RNG 메타를 태그에 복사해 타임라인에서 보기 쉽게(디버그 ON일 때만)
        t = dict(tags or {})
        try:
            if self._plan is not None:
                if getattr(self._plan, "seed", None) is not None:
                    t.setdefault("rng_seed", int(getattr(self._plan, "seed")))
                if getattr(self._plan, "rng_step", None) is not None:
                    t.setdefault("rng_step", int(getattr(self._plan, "rng_step")))
        except Exception:
            pass
        self._tags = t if self._expose_debug else {}

    # ---- 공개 표면(테스트/문서 호환) ----
    @property
    def io(self):
        """I/O 바인딩 테이블(문서·테스트 호환을 위해 공개)."""
        return self._io

    @property
    def logits(self):
        return self._io["logits"]

    @property
    def X_buf(self):
        return self._io["X"]

    @property
    def y_buf(self):
        return self._io["y"]

    # ---- 선택적 디버그 표면 ----
    @property
    def plan(self):
        """CapturePlan 핸들(환경변수 GEV2_EXPOSE_DEBUG=1일 때만 노출)."""
        return self._plan

    @property
    def key(self):
        """GraphPool 키(환경변수 GEV2_EXPOSE_DEBUG=1일 때만 노출)."""
        return self._key

    @property
    def tags(self):
        """NVTX 등 캡처/리플레이 태그(환경변수 GEV2_EXPOSE_DEBUG=1일 때만 노출)."""
        return self._tags

    # pytest/문서 스크립트가 탐색 호출할 수 있는 얇은 헬퍼
    def debug_capture_plan(self):
        return self.plan

    def debug_dump_ir(self):
        # IR은 보통 Sequential/Builder 쪽에서 보관하므로 여기서는 None 반환
        return None

    def set_batch(self, X_dev, y_dev):
        """현재 배치를 고정 I/O 버퍼(X/y)에 복사 (그래프와 동일 스트림)."""
        xb, yb = self._io["X"], self._io["y"]
        with self._stream:  # ✅ 그래프와 동일 스트림에서 H2D/D2D 수행
            xb[...] = cp.asarray(X_dev, dtype=xb.dtype)
            yb[...] = cp.asarray(y_dev, dtype=yb.dtype)

    def launch(self):
        """CUDA Graph 인스턴스(or 폴백) 실행."""
        self._gexec.launch(self._stream.ptr)
