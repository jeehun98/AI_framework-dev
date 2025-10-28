# File: python/graph_executor_v2/layers/sequential.py
from __future__ import annotations
from typing import (
    List, Tuple, Any, Iterable, Optional, Dict, Sequence, TYPE_CHECKING
)
import cupy as cp

# ============================================================================
# 이 파일의 목적
# ----------------------------------------------------------------------------
# - 고수준 Sequential 컨테이너:
#     * Eager 경로: call()/backward() 로 즉시 실행
#     * Graph Capture 경로:
#         - 정적(Static): compile() → one_step()
#         - 동적(Dynamic): one_step_dynamic()  (If/Repeat/EarlyExit 등 분기 포함)
#
# - 외부 모듈 연결:
#     * graph.capture_plan:
#         - make_plan_for_sequential(): 정적 전체 모델 플랜
#         - make_plan_for_path(): 동적 "평탄화된 경로" 전용 플랜
#         - advance_dropout(): 반복시 시드/마스크 전진
#     * graph.graph_exec:
#         - record_step_graph(): fwd→loss→bwd→opt 1 step을 CUDA Graph로 캡처
#         - TrainGraph: set_batch()/launch() 로 재생(replay)
#     * graph.graph_executor:
#         - GraphSignature/GraphKey/MultiGraphPool, graph_pool 인스턴스
#           (동적 경로별 TrainGraph 캐시/재사용에 쓰임)
#
# - NVTX 태깅:
#     * 타임라인 분석을 위한 통일된 네이밍 사용
#     * [CAPTURE][static] / [REPLAY][static] / [DYN] ... 등
#
# - 향후 확장(설계 여지):
#     * Execution Planner(토폴로지→스트림/이벤트 스케줄) 삽입 지점:
#         - 정적: compile()에서 make_plan_for_sequential(...) 직후
#         - 동적: _get_or_capture_dynamic_entry()에서 make_plan_for_path(...) 직후
#     * Graph Runtime(Allocator/RNG/Stream/Tape 통합) 주도 캡처:
#         - graph_exec.record_step_graph(...) 내부
# ============================================================================

from graph_executor_v2.graph.capture_plan import (
    make_plan_for_sequential,
    make_plan_for_path,
    advance_dropout,
)
from graph_executor_v2.graph.graph_exec import record_step_graph, TrainGraph
from graph_executor_v2.optim.rebind import try_rebind_grads
from .base import Layer

import inspect
import time

# ===== NVTX (optional) =====
# 통일된 네이밍으로 타임라인 분석을 쉽게 하기 위해 래퍼를 사용합니다.
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

# ✅ 런타임 임포트 우선: 실제 클래스 로드 실패시 스텁으로 폴백
#    - 동적 경로 평탄화(_linearize_path)에서 If/Repeat/EarlyExit를 "덕 타이핑"으로만 식별하므로
#      이 임포트가 실패해도 기능상 문제는 없음(체크는 getattr 기반).
try:
    from graph_executor_v2.layers.conditional import If, Repeat, EarlyExit  # 실제 컨트롤 레이어
except Exception:
    class _Missing: ...
    If = Repeat = EarlyExit = _Missing  # fallback stubs

# ============================
# Typing-only imports & stubs
# ============================
if TYPE_CHECKING:
    from graph_executor_v2.graph.graph_executor import (
        GraphSignature, GraphKey, MultiGraphPool
    )
else:
    # 런타임에 타입이 없더라도 파일은 동작해야 하므로 Any로 폴백
    from typing import Any as _AnyType
    GraphSignature = _AnyType  # type: ignore[assignment]
    GraphKey = _AnyType        # type: ignore[assignment]
    MultiGraphPool = _AnyType  # type: ignore[assignment]

# 런타임 인스턴스 로딩 (없어도 동작하도록 폴백 준비)
# - 동적 경로 그래프 캐시(풀)가 존재하면 우선 사용, 없으면 로컬 dict로 대체
try:
    from graph_executor_v2.graph.graph_executor import graph_pool  # type: ignore
except Exception:
    graph_pool = None  # type: ignore

# 폴백: 프로세스 내 간단한 캐시(dict) + LRU
_FALLBACK_POOL: Dict[Any, Any] = {}

# parameters()에서 (p, g) 자동 탐색 시 사용하는 후보 속성명들
CANDIDATE_PARAM_GRAD_NAMES = [
    ("W", "dW"),
    ("weight", "dweight"),
    ("b", "db"),
    ("bias", "dbias"),
]

# 동적 경로 호환용: 옛 graph_exec에도 동작하도록 하는 프록시
class _ModelLayersProxy:
    """model의 다른 속성은 그대로 위임하고, layers만 path_layers로 바꿔치기.

    - 일부 record_step_graph 버전이 layers_override를 지원하지 않는 경우 사용.
    - self._base에 모든 접근을 위임하되, 'layers' 접근/설정만 override한다.
    """
    def __init__(self, base, layers):
        self._base = base
        self.layers = list(layers)

    def __getattr__(self, name):
        if name == "layers":
            return self.layers
        return getattr(self._base, name)

    def __setattr__(self, name, value):
        if name in ("_base", "layers"):
            object.__setattr__(self, name, value)
        else:
            setattr(self._base, name, value)


class Sequential(Layer):
    """고수준 순차 모델 컨테이너.

    ▶ 지원 모드
      - Eager: call()/backward()
      - Graph(정적): compile() → one_step()
      - Graph(동적): one_step_dynamic()  (If/Repeat/EarlyExit 포함 경로별 캡처/캐시)

    ▶ 외부 연동
      - capture_plan: make_plan_for_*(), advance_dropout()
      - graph_exec: record_step_graph(), TrainGraph
      - graph_executor: GraphKey/GraphSignature/graph_pool
    """
    # 폴백 풀 상한/LRU 제어용
    _FALLBACK_POOL_MAX = 8

    def __init__(self, *layers: Layer, name: Optional[str] = None):
        super().__init__(name=name)
        self.layers: List[Layer] = list(layers)
        self.training: bool = True

        # 정적 캡처 결과 핸들
        self._tg: Optional[TrainGraph] = None
        self._loss_buf: Optional[cp.ndarray] = None
        self._stream: Optional[cp.cuda.Stream] = None

        # LRU tick (미사용이지만 향후 시간기반 정책 확장 용이)
        self._pool_ticks: int = 0

    def _tick(self) -> int:
        self._pool_ticks += 1
        return self._pool_ticks

    # -------------------------------------------------------------------------
    # 구성/빌드
    # -------------------------------------------------------------------------
    def add(self, layer: Layer) -> None:
        """레이어를 추가하고, 이미 빌드된 상태라면 간단히 출력 shape를 추적 갱신."""
        self.layers.append(layer)
        if self.built and self.output_shape is not None:
            ish = self.output_shape
            try:
                layer.build(ish)
                osh = layer.compute_output_shape(ish)
            except Exception:
                osh = None
            if osh is not None:
                self.output_shape = tuple(map(int, osh))

    def build(
        self,
        input_shape: Tuple[int, ...],
        *,
        strict: bool = True,
        verify_output: bool = True
    ) -> None:
        """모든 하위 레이어에 대해 build/compute_output_shape를 순차 수행.

        - strict=True: 중간 레이어에서 예외 발생 시 즉시 실패
        - verify_output=True: 전체 빌드 종료 후 결과 검증/오류 리포트
        """
        cur = tuple(map(int, input_shape))
        errors = []

        for i, lyr in enumerate(self.layers):
            lname = f"{lyr.__class__.__name__}:{i}"
            try:
                if hasattr(lyr, "build"):
                    lyr.build(cur)
            except Exception as e:
                msg = f"[Sequential.build] build failed at {lname} with in_shape={cur}: {e}"
                if strict:
                    raise RuntimeError(msg) from e
                errors.append(msg)
            try:
                if hasattr(lyr, "compute_output_shape"):
                    cur = tuple(map(int, lyr.compute_output_shape(cur)))
            except Exception as e:
                msg = f"[Sequential.build] compute_output_shape failed at {lname} with in_shape={cur}: {e}"
                if strict:
                    raise RuntimeError(msg) from e
                errors.append(msg)
                cur = None
                break

        self.input_shape = tuple(map(int, input_shape))
        self.output_shape = cur if isinstance(cur, tuple) else None
        self.built = (len(errors) == 0) and (self.output_shape is not None)

        if verify_output and not self.built:
            detail = "\n".join(errors) if errors else "unknown error"
            raise RuntimeError(
                f"[Sequential.build] build incomplete. output_shape={self.output_shape}, "
                f"errors:\n{detail}"
            )

    # -------------------------------------------------------------------------
    # Eager 실행 (참고/디버깅/테스트용)
    # -------------------------------------------------------------------------
    def call(self, x: Any):
        """즉시 실행 forward (캡처 없이). 디버깅/테스트에 유용."""
        out = x
        for lyr in self.layers:
            if hasattr(lyr, "training"):
                lyr.training = self.training
            out = lyr(out)
        return out

    def backward(self, grad_output: Any):
        """즉시 실행 backward (캡처 없이)."""
        g = grad_output
        for lyr in reversed(self.layers):
            g = lyr.backward(g)
        return g

    def compute_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """레이어들의 compute_output_shape()를 순차 호출해 최종 출력을 계산."""
        cur = tuple(map(int, input_shape))
        for lyr in self.layers:
            cur = lyr.compute_output_shape(cur)
        return cur

    def summary(self, indent: int = 2) -> str:
        """간단한 요약 문자열 생성 (shape/파라미터 수 등)."""
        lines = []
        pad = " " * indent
        lines.append(f"Sequential(name={self.name})")
        if self.input_shape:
            lines.append(f"{pad}Input:  {self.input_shape}")
        cur = self.input_shape
        total_params = 0
        for i, lyr in enumerate(self.layers):
            cls = lyr.__class__.__name__
            shp = None
            if cur is not None:
                try:
                    shp = lyr.compute_output_shape(cur)
                    cur = shp
                except Exception:
                    shp = "?"
                    cur = None
            pcount = 0
            try:
                for (p, _, _) in lyr.parameters():  # type: ignore
                    try:
                        pcount += int(p.size) if hasattr(p, "size") else int(p.size())
                    except Exception:
                        pass
            except Exception:
                pass
            total_params += pcount
            lines.append(f"{pad}[{i:02d}] {cls:>20} -> {shp}  (params={pcount})")
        if cur is not None:
            lines.append(f"{pad}Output: {cur}")
        lines.append(f"{pad}Total params: {total_params}")
        return "\n".join(lines)

    # -------------------------------------------------------------------------
    # 학습 유틸
    # -------------------------------------------------------------------------
    def train(self, mode: bool = True):
        """train/eval 모드 플래그를 하위 레이어에 전파."""
        self.training = bool(mode)
        for lyr in self.layers:
            if hasattr(lyr, "training"):
                lyr.training = self.training
        return self

    def eval(self):
        """eval 모드 진입 (train(False))"""
        return self.train(False)

    def parameters(self) -> Iterable[Tuple[Any, Any, str]]:
        """(param, grad, tag) 를 순회하며 방출.

        - 레이어가 `parameters()`를 제공하면 그것을 우선 사용
        - 없으면 후보 속성명(CANDIDATE_PARAM_GRAD_NAMES)로 덕타이핑 수집
        """
        for idx, lyr in enumerate(self.layers):
            lname = f"{lyr.__class__.__name__}:{idx}"
            if hasattr(lyr, "parameters") and callable(getattr(lyr, "parameters")):
                for t in lyr.parameters():  # type: ignore
                    if isinstance(t, tuple) and len(t) == 3:
                        yield t
                    elif isinstance(t, tuple) and len(t) == 2:
                        p, g = t
                        yield (p, g, lname)
                continue
            for p_name, g_name in CANDIDATE_PARAM_GRAD_NAMES:
                if hasattr(lyr, p_name) and hasattr(lyr, g_name):
                    p = getattr(lyr, p_name)
                    g = getattr(lyr, g_name)
                    yield (p, g, f"{lname}.{p_name}")

    def zero_grad(self):
        """모든 파라미터 그래드를 0으로 설정(가능하면 in-place)."""
        for (p, _, _) in self.parameters():
            g = getattr(p, "grad", None)
            if g is not None:
                try:
                    g[...] = 0
                except Exception:
                    try:
                        if hasattr(g, "zero_"):
                            g.zero_()
                        else:
                            setattr(p, "grad", None)
                    except Exception:
                        pass
        for lyr in self.layers:
            if hasattr(lyr, "zero_grad") and callable(getattr(lyr, "zero_grad")):
                try:
                    lyr.zero_grad()  # type: ignore
                except Exception:
                    pass
                continue
            for _, g_name in CANDIDATE_PARAM_GRAD_NAMES:
                if hasattr(lyr, g_name):
                    g = getattr(lyr, g_name)
                    try:
                        g[...] = 0
                    except Exception:
                        try:
                            if hasattr(g, "zero_"):
                                g.zero_()
                            else:
                                setattr(lyr, g_name, None)
                        except Exception:
                            pass

    def attach_grads(self):
        """(p, g) 쌍이 제공되는 경우 p.grad에 g를 연결(역호환)."""
        for (p, g, _) in self.parameters():
            if g is not None:
                try:
                    setattr(p, "grad", g)
                except Exception:
                    pass

    # =========================
    # ===== Graph Capture =====
    # =========================
    def supports_capture(self) -> bool:
        """모든 레이어가 capture-safe 인터페이스(forward_into/backward_into)를 지원하는가?"""
        ok = True
        for lyr in self.layers:
            f_ok = hasattr(lyr, "forward_into") and callable(getattr(lyr, "forward_into"))
            b_ok = hasattr(lyr, "backward_into") and callable(getattr(lyr, "backward_into"))
            ok = ok and f_ok and b_ok
        return ok

    def compile(
        self,
        input_shape: Tuple[int, ...],
        *,
        loss,
        optimizer,
        lt_bytes: int = (8 << 20),
        stream: Optional[cp.cuda.Stream] = None,
    ) -> "TrainGraph":
        """정적(Graph) 경로: 전체 모델 1-step을 CUDA Graph로 캡처해 재생 준비.

        흐름:
          1) (필요시) build()
          2) capture-safe 가드(supports_capture)
          3) make_plan_for_sequential(...)  → 전체 DAG/바인딩 준비
          4) try_rebind_grads(...)         → 옵티마이저에 그래드 버퍼 재바인딩
          5) 고정 I/O 버퍼(X/y/loss) 생성  → capture-safe를 위해 shape/dtype 고정
          6) record_step_graph(...)        → fwd→loss→bwd→opt 1-step 캡처
          7) TrainGraph 생성/보관          → 이후 one_step()에서 replay

        TODO(Planner 통합):
          - ExecPlanner().build(plan.dag, ...) 를 3) 이후에 호출
          - record_step_graph(..., exec_plan=exec_plan) 으로 전달
        """
        in_shape = tuple(map(int, input_shape))
        if not self.built:
            self.build(in_shape)

        assert self.supports_capture(), "All layers must implement forward_into/backward_into for capture"

        if stream is None:
            stream = cp.cuda.Stream(non_blocking=True)

        # 3) 전체 모델용 캡처 플랜 생성
        plan = make_plan_for_sequential(
            self, in_shape, loss_kind="softmax_ce", lt_bytes=lt_bytes
        )

        # 4) 옵티마이저-그래드 버퍼 리바인드 (캡처 전 일관화)
        try_rebind_grads(self, optimizer, plan)

        # 5) 캡처-세이프 I/O 버퍼 (커널 제약 고려해 fp32/labels=int32)
        X_buf = cp.zeros(in_shape, dtype=cp.float32)
        N = int(in_shape[0])
        y_buf = cp.zeros((N,), dtype=cp.int32)
        loss_buf = cp.zeros((), dtype=cp.float32)

        # 6) CUDA Graph 캡처
        with nvtx_range("[CAPTURE][static]"):
            gexec = record_step_graph(
                self,
                loss,
                optimizer.step_into,
                plan,
                X_buf=X_buf,
                y_buf=y_buf,
                stream=stream,
                loss_out=loss_buf,
                # exec_plan=... (Planner 통합 시 전달)
            )

        io = {"X": X_buf, "y": y_buf, "logits": plan.per_layer[-1].y}
        tg = TrainGraph(gexec, io, stream)

        # 7) 내부 핸들 보관
        self._tg = tg
        self._loss_buf = loss_buf
        self._stream = stream
        return tg

    def one_step(self, X, y) -> float:
        """정적(Graph) 경로의 1 step 재생(replay).

        전제: compile()로 캡처/준비 완료 상태.
        - 고정 버퍼에 현재 배치 복사 → cudaGraphLaunch → loss 읽기
        """
        assert self._tg is not None, "call compile() first"
        assert self._loss_buf is not None, "loss buffer not initialized"

        xb, yb = self._tg.X_buf, self._tg.y_buf
        x_arr = cp.asarray(X)
        y_arr = cp.asarray(y)

        # 입출력 가드 (정적 그래프는 shape/dtype 불변이 원칙)
        assert tuple(xb.shape) == tuple(x_arr.shape), f"X shape mismatch: {x_arr.shape} vs {xb.shape}"
        assert yb.shape == (xb.shape[0],), f"y shape must be (N,), got {yb.shape} vs N={xb.shape[0]}"
        assert yb.dtype == cp.int32, f"labels must be int32 for current CE kernel (got {yb.dtype})"

        self._tg.set_batch(x_arr, y_arr)
        with nvtx_range("[REPLAY][static]"):
            self._tg.launch()
        return float(self._loss_buf.get())

    @property
    def tg(self) -> TrainGraph:
        """현재 활성 TrainGraph 핸들(정적 또는 최근 동적 경로)을 반환."""
        assert self._tg is not None, "call compile() first"
        return self._tg

    # =========================================================
    # ========== Dynamic Path Handling (분기/반복) ============
    # =========================================================

    def _infer_signature(self, X, ctx: Dict[str, Any]) -> "GraphSignature":
        """GraphSignature 생성 (shape/dtype/layout 등 최소 정보).

        - GraphSignature 타입이 런타임에 없을 수도 있으므로 Any 폴백 클래스 사용.
        - 서명은 GraphKey 구성의 일부로 사용되어 그래프 풀 캐시 키를 안정화.
        """
        from typing import Any as _AnyType
        if GraphSignature is _AnyType:  # type: ignore[comparison-overlap]
            class _Sig:
                __slots__ = ("shape", "dtype", "layout")
                def __init__(self, shape, dtype, layout):
                    self.shape = tuple(map(int, shape))
                    self.dtype = str(dtype)
                    self.layout = str(layout)
            dtype = getattr(X, "dtype", None)
            dtype_str = str(dtype) if dtype is not None else "fp32"
            shape = tuple(getattr(X, "shape", ()))
            layout = ctx.get("layout", "rowmajor")
            return _Sig(shape, dtype_str, layout)  # type: ignore[return-value]
        dtype = getattr(X, "dtype", None)
        dtype_str = str(dtype) if dtype is not None else "fp32"
        shape = tuple(getattr(X, "shape", ()))
        layout = ctx.get("layout", "rowmajor")
        return GraphSignature(shape=shape, dtype=dtype_str, layout=layout)  # type: ignore[call-arg]

    def _pool_get(self, key: Any) -> Optional[Any]:
        """그래프 풀(있으면) 또는 로컬 폴백에서 엔트리 조회."""
        if graph_pool is not None and hasattr(graph_pool, "get"):
            try:
                entry = graph_pool.get(key)  # type: ignore[attr-defined]
                return entry
            except Exception:
                return None
        entry = _FALLBACK_POOL.get(key)
        if entry is not None:
            entry["last_used"] = time.monotonic()
        return entry

    def _pool_put(self, key: Any, entry: Any) -> None:
        """그래프 풀(있으면) 또는 로컬 폴백에 엔트리 저장 (LRU 상한 관리)."""
        if graph_pool is not None and hasattr(graph_pool, "put"):
            try:
                graph_pool.put(key, entry)  # type: ignore[attr-defined]
                return
            except Exception:
                pass
        # Fallback with LRU cap
        entry["last_used"] = time.monotonic()
        _FALLBACK_POOL[key] = entry
        if len(_FALLBACK_POOL) > self._FALLBACK_POOL_MAX:
            # evict LRU
            victim = min(_FALLBACK_POOL.items(), key=lambda kv: kv[1].get("last_used", 0.0))[0]
            _FALLBACK_POOL.pop(victim, None)

    def _make_pool_key(self, sig: Any, ctx: Dict[str, Any], *, loss) -> Any:
        """GraphPool 키 생성.

        - 누적 경로(branch_path)를 우선 사용 → 동일 모델 내 복수 분기 충돌 방지
        - variant: training/amp/loss_kind/dtype/경로지문(path_fingerprint) 등 불변화하여 튜플로 고정
        - GraphKey 타입이 있으면 사용, 없으면 해시안정 튜플로 폴백
        """
        # 누적 경로 우선
        branch_path = ctx.get("branch_path")
        if branch_path:
            branch_id = "->".join(map(str, branch_path))
        else:
            branch_id = ctx.get("branch", "default")

        vdict = dict(ctx.get("variant", {}))
        vdict["path_fp"] = tuple(ctx.get("path_fingerprint", ()))
        vdict["training"] = bool(self.training)
        vdict["dtype"] = str(getattr(sig, "dtype", "fp32"))
        vdict["loss_kind"] = getattr(loss, "name", "softmax_ce")
        vdict["amp"] = bool(ctx.get("amp", False))
        variant = tuple(sorted((str(k), self._freeze_value(v)) for k, v in vdict.items()))
        try:
            if GraphKey not in (None, object):  # 약식 가드
                return GraphKey(signature=sig, branch_id=str(branch_id), variant=variant)  # type: ignore[call-arg]
        except Exception:
            pass
        return ("dyn",
                tuple(getattr(sig, "shape", ()) ),
                str(getattr(sig, "dtype", "")),
                str(getattr(sig, "layout", "")),
                str(branch_id),
                variant)

    @staticmethod
    def _freeze_value(v: Any) -> Any:
        """변형 가능한 값들을 해시가능한 불변 값으로 고정."""
        if isinstance(v, (str, int, float, bool, type(None))):
            return v
        if isinstance(v, (tuple, list)):
            return tuple(Sequential._freeze_value(x) for x in v)
        if isinstance(v, dict):
            return tuple(sorted((str(k), Sequential._freeze_value(val)) for k, val in v.items()))
        return str(v)

    def _linearize_path(self, X, ctx: Dict[str, Any]) -> List[Layer]:
        """동적 제어 레이어(If/Repeat/EarlyExit)를 '실행된 경로'로 평탄화.

        처리 규칙:
          - If: l.decide(X, ctx) → (branch, block)
                · ctx['branch_path'] += (branch,)
                · block이 Sequential이면 내부 layers를 전개
          - Repeat: T = l.steps(X, ctx)
                · ctx['repeat_steps'] = T (캡처는 1step, 재생 T회)
                · body 전개
          - EarlyExit: stages를 순차 전개; 각 stage 전개 후 exit_fn(ctx) True면 종료
                · ctx['branch_path'] += (f"ee:{k}",)
                · ctx['earlyexit'] = True
          - 컨트롤 레이어가 전개 후에도 남아있으면 예외 (평탄화 누락 가드)
          - ctx['path_fingerprint'] = (레이어 클래스 명 시퀀스) 기록
        """
        def _is_if(obj):
            return callable(getattr(obj, "decide", None)) and \
                hasattr(obj, "then_block") and hasattr(obj, "else_block")

        def _is_repeat(obj):
            return callable(getattr(obj, "steps", None)) and hasattr(obj, "body")

        def _is_early(obj):
            return hasattr(obj, "stages") and isinstance(getattr(obj, "stages"), (list, tuple))

        # 누적 분기 경로 컨테이너 초기화
        if "branch_path" not in ctx:
            ctx["branch_path"] = tuple()

        linear: List[Layer] = []
        for l in self.layers:
            if _is_if(l):
                branch, block = l.decide(X, ctx)
                ctx["branch_path"] = tuple(ctx["branch_path"]) + (branch,)
                ctx["branch"] = branch  # 단일 키도 유지 (레거시)
                if isinstance(block, Sequential):
                    linear.extend(block.layers)
                else:
                    linear.append(block)

            elif _is_repeat(l):
                T = int(l.steps(X, ctx))
                ctx["repeat_steps"] = T
                body = l.body
                if isinstance(body, Sequential):
                    linear.extend(body.layers)
                else:
                    linear.append(body)

            elif _is_early(l):
                # 각 stage를 순차 전개, stage마다 exit_fn(ctx) 검사하여 조기 종료
                stages = list(l.stages)
                for k, s in enumerate(stages):
                    if isinstance(s, Sequential):
                        linear.extend(s.layers)
                    else:
                        linear.append(s)
                    if callable(getattr(l, "exit_fn", None)) and l.exit_fn(ctx):
                        ctx["branch_path"] = tuple(ctx["branch_path"]) + (f"ee:{k}",)
                        break
                ctx["earlyexit"] = True

            else:
                linear.append(l)

        # ✅ 컨트롤 레이어 잔존 가드 (평탄화 누락 방지)
        leftovers = []
        def _is_ctrl(x):
            return _is_if(x) or _is_repeat(x) or _is_early(x)
        for x in linear:
            if _is_ctrl(x):
                leftovers.append(type(x).__name__)
        if leftovers:
            raise RuntimeError(
                f"[dynamic] control layers must be flattened, but found in path: {leftovers}"
            )

        # 경로 fingerprint 저장 (레이어 클래스 시퀀스)
        ctx["path_fingerprint"] = tuple(type(l).__name__ for l in linear)
        return linear

    @staticmethod
    def _ensure_path_captureable(layers: Sequence[Layer]) -> None:
        """경로 내 모든 레이어가 capture-safe 인터페이스를 지원하는지 확인."""
        for lyr in layers:
            if not (hasattr(lyr, "forward_into") and hasattr(lyr, "backward_into")):
                raise AssertionError(f"Layer {type(lyr).__name__} not capture-ready")

    def _get_or_capture_dynamic_entry(
        self,
        X: cp.ndarray,
        y: cp.ndarray,
        *,
        loss,
        optimizer,
        ctx: Dict[str, Any],
        lt_bytes: int,
        stream: Optional[cp.cuda.Stream],
    ) -> Dict[str, Any]:
        """동적 경로의 핵심 진입점: 평탄화→키생성→캐시조회→(미스)캡처→엔트리반환.

        반환 엔트리:
          {
            "tg": TrainGraph,         # 경로별 TrainGraph
            "loss_buf": ndarray,      # 손실 스칼라 버퍼
            "stream": Stream,         # 사용 스트림
            "optimizer": Optimizer,   # 경로 전용 옵티마이저(리바인드 끝난)
            "plan": CapturePlan,      # advance_dropout 등에 사용
          }
        """
        # 1) 경로 평탄화
        with nvtx_range("[DYN] path_linearize"):
            path_layers = self._linearize_path(X, ctx)
        self._ensure_path_captureable(path_layers)

        # 2) 키 구성 (GraphSignature + branch_path 등)
        with nvtx_range("[DYN] make_pool_key"):
            sig = self._infer_signature(X, ctx)
            key = self._make_pool_key(sig, ctx, loss=loss)

        # 3) 풀 조회 (GraphPool → Fallback dict)
        with nvtx_range("[DYN] get_from_pool"):
            entry = self._pool_get(key)
            if entry is not None:
                return entry

        # 4) 신규 캡처 (미스 시)
        in_shape = tuple(map(int, getattr(sig, "shape", tuple(X.shape))))

        # 경로 레이어 재빌드(배치/타임 변화 대응)
        def _rebuild_path_layers(layers, ish):
            cur = tuple(ish)
            for lyr in layers:
                try:
                    if hasattr(lyr, "build"):
                        lyr.build(cur)
                except Exception:
                    pass
                try:
                    if hasattr(lyr, "compute_output_shape"):
                        cur = tuple(map(int, lyr.compute_output_shape(cur)))
                except Exception:
                    pass

        if not self.built:
            self.build(in_shape)
        else:
            if tuple(self.input_shape or ()) != in_shape:
                _rebuild_path_layers(path_layers, in_shape)

        if stream is None:
            stream = cp.cuda.Stream(non_blocking=True)

        # 동적 경로 전용 플랜
        with nvtx_range("[DYN] make_capture_plan"):
            plan = make_plan_for_path(
                path_layers, in_shape, loss_kind=getattr(loss, "name", "softmax_ce"), lt_bytes=lt_bytes
            )

        # TODO(Planner 통합): 여기서 ExecPlanner().build(plan.dag, ...) 호출 → exec_plan
        # TODO(GraphRuntime 통합): record_step_graph(..., exec_plan=exec_plan)

        # ---- 경로 전용 (param, grad) 트리플 수집: 정확 매핑 + 중복 방지 ----
        def _collect_triplets_from_path(plan, layers):
            """캡처 플랜(per_layer.* grad 버퍼)을 경로 레이어 파라미터에 정확 매핑."""
            triplets = []
            seen = set()
            def push(p, g, tag):
                # Tensor-like라 가정: (data.ptr) 또는 id 기반으로 유일성 판단
                key = (
                    int(getattr(getattr(p, "data", p), "ptr", id(p))),
                    int(getattr(getattr(g, "data", g), "ptr", id(g)))
                )
                if key not in seen:
                    triplets.append((p, g, tag))
                    seen.add(key)

            for i, lyr in enumerate(layers):
                per = plan.per_layer[i]
                # Dense/Conv 공통
                if hasattr(lyr, "W") and per.gW is not None:
                    push(getattr(lyr, "W"), per.gW, f"{type(lyr).__name__}:{i}.W")
                for b_name in ("b", "bias", "B"):
                    if hasattr(lyr, b_name) and getattr(lyr, b_name) is not None and per.gB is not None:
                        push(getattr(lyr, b_name), per.gB, f"{type(lyr).__name__}:{i}.{b_name}")
                        break
                # BN
                if hasattr(lyr, "gamma") and per.gW is not None:
                    try:
                        if tuple(getattr(lyr, "gamma").shape) == tuple(per.gW.shape):
                            push(getattr(lyr, "gamma"), per.gW, f"BN2d:{i}.gamma")
                    except Exception:
                        pass
                if hasattr(lyr, "beta") and per.gB is not None:
                    try:
                        if tuple(getattr(lyr, "beta").shape) == tuple(per.gB.shape):
                            push(getattr(lyr, "beta"), per.gB, f"BN2d:{i}.beta")
                    except Exception:
                        pass
                # RNN
                for w_name, g_name, tag in (("Wx", "gWx", "Wx"), ("Wh", "gWh", "Wh")):
                    if hasattr(lyr, w_name) and getattr(per, g_name, None) is not None:
                        push(getattr(lyr, w_name), getattr(per, g_name), f"RNN:{i}.{tag}")
                if hasattr(lyr, "b") and getattr(per, "gB", None) is not None and getattr(lyr, "b") is not None:
                    push(getattr(lyr, "b"), per.gB, f"RNN:{i}.b")
            return triplets

        triplets = _collect_triplets_from_path(plan, path_layers)

        # ---- 옵티마이저 바인딩 (경로별 옵티마이저 생성/캐시 or 재바인드) ----
        opt_for_path = optimizer

        def _new_opt_like(base_opt):
            """기본 옵티마이저 하이퍼파라미터를 복사해 경로전용 인스턴스를 생성."""
            OptCls = base_opt.__class__
            hyper = {}
            for k in ("lr", "wd", "weight_decay", "beta1", "beta2", "betas", "eps"):
                if hasattr(base_opt, k):
                    hyper[k] = getattr(base_opt, k)
            try:
                return OptCls([], **hyper)
            except TypeError:
                return OptCls([])

        with nvtx_range("[DYN] rebind"):
            try:
                if hasattr(opt_for_path, "rebind_grads"):
                    opt_for_path.rebind_grads(triplets)
                else:
                    raise AssertionError("optimizer has no rebind_grads")
            except AssertionError:
                # 원본이 rebind를 지원하지 않으면 경로전용 옵티마이저를 새로 생성
                opt_for_path = _new_opt_like(optimizer)
                if hasattr(opt_for_path, "ensure_initialized"):
                    try:
                        opt_for_path.ensure_initialized()
                    except Exception:
                        pass
                opt_for_path.rebind_grads(triplets)

        # 5) 고정 I/O 버퍼 (현재 커널 제약상 fp32/int32가 안전)
        X_buf = cp.zeros(in_shape, dtype=cp.float32)
        N = int(in_shape[0])
        y_buf = cp.zeros((N,), dtype=cp.int32)
        loss_buf = cp.zeros((), dtype=cp.float32)

        # ---- record_step_graph 하위호환 처리 ----
        try:
            sig_rs = inspect.signature(record_step_graph)
            has_layers_override = ("layers_override" in sig_rs.parameters)
        except Exception:
            has_layers_override = False

        with nvtx_range(f"[DYN] record_step_graph path={ctx.get('path_fingerprint')}"):
            if has_layers_override:
                gexec = record_step_graph(
                    self,
                    loss,
                    opt_for_path.step_into,
                    plan,
                    X_buf=X_buf,
                    y_buf=y_buf,
                    stream=stream,
                    loss_out=loss_buf,
                    layers_override=path_layers,
                    # exec_plan=... (Planner 통합 시)
                )
            else:
                # layers_override 미지원 record_step_graph에 대한 호환
                proxy_model = _ModelLayersProxy(self, path_layers)
                gexec = record_step_graph(
                    proxy_model,
                    loss,
                    opt_for_path.step_into,
                    plan,
                    X_buf=X_buf,
                    y_buf=y_buf,
                    stream=stream,
                    loss_out=loss_buf,
                    # exec_plan=... (Planner 통합 시)
                )

        io = {"X": X_buf, "y": y_buf, "logits": plan.per_layer[-1].y}
        tg = TrainGraph(gexec, io, stream)

        entry = {
            "tg": tg,
            "loss_buf": loss_buf,
            "stream": stream,
            "optimizer": opt_for_path,
            "plan": plan,  # Dropout counter advance 등에 사용
        }
        self._pool_put(key, entry)
        return entry

    def one_step_dynamic(
        self,
        X,
        y,
        *,
        loss,
        optimizer,
        ctx: Optional[Dict[str, Any]] = None,
        lt_bytes: int = (8 << 20),
        stream: Optional[cp.cuda.Stream] = None,
    ) -> float:
        """동적(Graph) 경로: If/Repeat/EarlyExit 포함한 '현재 실행된 경로'를 캡처/재생.

        흐름:
          1) _get_or_capture_dynamic_entry(...) 호출
             - 평탄화 → 키 구성 → 캐시 조회 → (미스) 플랜/옵티마이저 리바인드 → record_step_graph
          2) entry.tg.set_batch(...) 로 고정버퍼에 배치 복사
          3) Repeat: ctx['repeat_steps']=T 이면 advance_dropout(plan, t) 후 tg.launch() T회
          4) loss_buf 읽어 반환
        """
        ctx = dict(ctx or {})
        x_arr = cp.asarray(X)
        y_arr = cp.asarray(y)

        entry = self._get_or_capture_dynamic_entry(
            x_arr, y_arr, loss=loss, optimizer=optimizer,
            ctx=ctx, lt_bytes=lt_bytes, stream=stream
        )

        tg: TrainGraph = entry["tg"]
        loss_buf: cp.ndarray = entry["loss_buf"]
        plan = entry.get("plan", None)

        # ✅ 현재 동적 경로 그래프 핸들을 모델 수준 핸들로 노출 (외부 사용 용이)
        self._tg = tg
        self._loss_buf = loss_buf
        self._stream = entry.get("stream", self._stream)

        # 모양/타입 가드
        assert tuple(tg.X_buf.shape) == tuple(x_arr.shape), \
            f"[dynamic] X shape mismatch: {x_arr.shape} vs {tg.X_buf.shape}"
        assert tg.y_buf.shape == (tg.X_buf.shape[0],), \
            f"[dynamic] y shape must be (N,), got {tg.y_buf.shape} vs N={tg.X_buf.shape[0]}"
        assert tg.y_buf.dtype == cp.int32, \
            f"[dynamic] labels must be int32 (got {tg.y_buf.dtype})"

        # 고정 버퍼에 배치 복사
        tg.set_batch(x_arr, y_arr)

        # Repeat: 캡처는 1 step 기준, 실행 시 T회 launch
        T = int(ctx.get("repeat_steps", 1))
        rep_batches = ctx.get("repeat_batches", None)  # [(X_t, y_t), ...] 가능

        with nvtx_range(f"[DYN] replay path={ctx.get('path_fingerprint')} x{T}"):
            if isinstance(rep_batches, (list, tuple)) and len(rep_batches) >= T:
                for t in range(T):
                    # 드롭아웃: 회차마다 카운터/시드 전진 (정책적으로 끌 수도 있음)
                    if plan is not None:
                        advance_dropout(plan, seed_bump=t)
                    xb_t = cp.asarray(rep_batches[t][0])
                    yb_t = cp.asarray(rep_batches[t][1])
                    assert tuple(tg.X_buf.shape) == tuple(xb_t.shape), "[dynamic] repeat batch X shape mismatch"
                    assert tg.y_buf.shape == (tg.X_buf.shape[0],), "[dynamic] repeat batch y shape mismatch"
                    tg.set_batch(xb_t, yb_t)
                    tg.launch()
            else:
                for t in range(max(1, T)):
                    if plan is not None:
                        advance_dropout(plan, seed_bump=t)
                    tg.launch()

        # 손실 스칼라 반환
        return float(loss_buf.get())
