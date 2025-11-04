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
from graph_executor_v2.graph.rewriter import run as rewrite

from .base import Layer

# 🔽 패턴 패스 (현재 no-op) — 이후 최적화/퓨전 추가 시 활성화
from graph_executor_v2.graph.pattern_registry import run_patterns

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
        self._tg: Optional[TrainGraph] = None
        self._loss_buf: Optional[cp.ndarray] = None
        self._stream: Optional[cp.cuda.Stream] = None
        self._pool_ticks: int = 0
        # === NEW: local telemetry counters ===
        self._tm = {
            "capture_count": 0,
            "replay_count": 0,
            "pool_hit": 0,
            "pool_miss": 0,
            "pool_put": 0,
            "pool_evict_fallback": 0,
        }

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
        """즉시 실행 forward (캡처 없이). 디버깅/테스트용 + None 가드."""
        out = x
        for i, lyr in enumerate(self.layers):
            if hasattr(lyr, "training"):
                lyr.training = self.training
            out = lyr(out)
            if out is None:
                lname = f"{type(lyr).__name__}:{i}"
                raise RuntimeError(
                    f"[Sequential.call] layer '{lname}' returned None in forward. "
                    f"Check its call() implementation to ensure it returns a tensor."
                )
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
        """(param, grad, tag)를 순회하며 방출.

        우선순위:
          1) 레이어가 parameters()를 제공하면 그대로 사용
          2) 후보 속성명 쌍(CANDIDATE_PARAM_GRAD_NAMES)
          3) ★ 일반 탐색: 파라미터 객체의 `.grad` 존재 여부로 수집
        """
        for idx, lyr in enumerate(self.layers):
            lname = f"{lyr.__class__.__name__}:{idx}"

            # 1) 레이어 자체 제공
            if hasattr(lyr, "parameters") and callable(getattr(lyr, "parameters")):
                for t in lyr.parameters():  # type: ignore
                    if isinstance(t, tuple) and len(t) == 3:
                        yield t
                    elif isinstance(t, tuple) and len(t) == 2:
                        p, g = t
                        yield (p, g, lname)
                continue

            # 2) 이름 쌍 덕 타이핑
            found_named = False
            for p_name, g_name in CANDIDATE_PARAM_GRAD_NAMES:
                if hasattr(lyr, p_name) and hasattr(lyr, g_name):
                    p = getattr(lyr, p_name)
                    g = getattr(lyr, g_name)
                    yield (p, g, f"{lname}.{p_name}")
                    found_named = True
            if found_named:
                continue

            # 3) ★ 일반 탐색: 자주 쓰는 파라미터 이름에서 .grad 붙은 객체 수집
            generic_names = ("W", "weight", "kernel", "b", "bias", "gamma", "beta")
            for p_name in generic_names:
                if hasattr(lyr, p_name):
                    p = getattr(lyr, p_name)
                    g = getattr(p, "grad", None)
                    if g is not None:
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
        """정적(Graph) 경로: 전체 모델 1-step을 CUDA Graph로 캡처해 재생 준비."""
        in_shape = tuple(map(int, input_shape))
        if not self.built:
            self.build(in_shape)

        assert self.supports_capture(), "All layers must implement forward_into/backward_into for capture"

        if stream is None:
            stream = cp.cuda.Stream(non_blocking=True)

        # ==== Pattern pass (no-op) ====
        layers_opt = run_patterns(self.layers)
        model_for_plan = self if layers_opt is self.layers else _ModelLayersProxy(self, layers_opt)

        # 3) 전체 모델용 캡처 플랜 생성
        plan = make_plan_for_sequential(
            model_for_plan, in_shape, loss_kind="softmax_ce", lt_bytes=lt_bytes
        )

        # 4) 옵티마이저-그래드 버퍼 리바인드 (캡처 전 일관화)
        try_rebind_grads(model_for_plan, optimizer, plan)

        # 5) 캡처-세이프 I/O 버퍼 (커널 제약 고려해 fp32/labels=int32)
        X_buf = cp.zeros(in_shape, dtype=cp.float32)
        N = int(in_shape[0])
        y_buf = cp.zeros((N,), dtype=cp.int32)
        loss_buf = cp.zeros((), dtype=cp.float32)

        # 6) CUDA Graph 캡처
        with nvtx_range("[CAPTURE][static]"):
            gexec = record_step_graph(
                model_for_plan,
                loss,
                optimizer.step_into,
                plan,
                X_buf=X_buf,
                y_buf=y_buf,
                stream=stream,
                loss_out=loss_buf,
            )

        io = {"X": X_buf, "y": y_buf, "logits": plan.per_layer[-1].y}

        # ⬇️ 문서/테스트용 태그 전달(키는 정적 경로에선 None 유지)
        tags = {"nvtx_capture_tag": "[CAPTURE][static]", "nvtx_replay_tag": "[REPLAY]"}
        tg = TrainGraph(gexec, io, stream, plan=plan, graph_key=None, tags=tags)

        # 7) 내부 핸들 보관
        self._tg = tg
        self._loss_buf = loss_buf
        self._stream = stream
        return tg

    def one_step(self, X, y) -> float:
        """정적(Graph) 경로의 1 step 재생(replay)."""
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
        
        self._tm["replay_count"] += 1  # === NEW ===
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
        """GraphSignature 생성 (shape/dtype/layout 등 최소 정보)."""
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
                entry = graph_pool.get(key)  # global pool에서 hit/miss 자체 카운트
                if entry is not None:
                    self._tm["pool_hit"] += 1
                else:
                    self._tm["pool_miss"] += 1
                return entry
            except Exception:
                pass
        entry = _FALLBACK_POOL.get(key)
        if entry is not None:
            entry["last_used"] = time.monotonic()
            self._tm["pool_hit"] += 1
        else:
            self._tm["pool_miss"] += 1
        return entry

    def _pool_put(self, key: Any, entry: Any) -> None:
        """그래프 풀(있으면) 또는 로컬 폴백에 엔트리 저장 (LRU 상한 관리)."""
        if graph_pool is not None and hasattr(graph_pool, "put"):
            try:
                graph_pool.put(key, entry)
                self._tm["pool_put"] += 1
                return
            except Exception:
                pass
        # Fallback with LRU cap
        entry["last_used"] = time.monotonic()
        _FALLBACK_POOL[key] = entry
        self._tm["pool_put"] += 1
        if len(_FALLBACK_POOL) > self._FALLBACK_POOL_MAX:
            # evict LRU
            victim = min(_FALLBACK_POOL.items(), key=lambda kv: kv[1].get("last_used", 0.0))[0]
            _FALLBACK_POOL.pop(victim, None)
            self._tm["pool_evict_fallback"] += 1

    def _make_pool_key(self, sig: Any, ctx: Dict[str, Any], *, loss) -> Any:
        """GraphPool 키 생성."""
        branch_path = ctx.get("branch_path")
        if branch_path:
            branch_id = "->".join(map(str, branch_path))
        else:
            branch_id = ctx.get("branch", "default")

        # === NEW === variant 구성 보정 (amp 문자열 반영 / variant 우선)
        vdict = dict(ctx.get("variant", {}))  # variant 우선
        if "amp" not in vdict and "amp" in ctx:
            vdict["amp"] = ctx.get("amp")
        vdict.setdefault("amp", "fp32")  # 기본값

        vdict["path_fp"] = tuple(ctx.get("path_fingerprint", ()))
        vdict["training"] = bool(self.training)
        vdict["dtype"] = str(getattr(sig, "dtype", "fp32"))
        vdict["loss_kind"] = getattr(loss, "name", "softmax_ce")

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
        """동적 제어 레이어(If/Repeat/EarlyExit)를 '실행된 경로'로 평탄화."""
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
        """동적 경로의 핵심 진입점: 평탄화→패턴→키→캐시→(미스)캡처."""
        # 1) 경로 평탄화
        with nvtx_range("[DYN] path_linearize"):
            path_layers = self._linearize_path(X, ctx)
        self._ensure_path_captureable(path_layers)

        # ==== Pattern pass (no-op) ====
        with nvtx_range("[DYN] patterns"):
            path_layers = rewrite(path_layers)

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

        # ---- 경로 전용 (param, grad) 트리플 수집: 정확 매핑 + 중복 방지 ----
        def _collect_triplets_from_path(plan, layers):
            triplets = []
            seen = set()
            def push(p, g, tag):
                key2 = (
                    int(getattr(getattr(p, "data", p), "ptr", id(p))),
                    int(getattr(getattr(g, "data", g), "ptr", id(g)))
                )
                if key2 not in seen:
                    triplets.append((p, g, tag))
                    seen.add(key2)

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
                    # graph_key는 TrainGraph로 전달만 하고 record_step_graph 내부에선 사용하지 않아도 OK
                    graph_key=key,
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
                    graph_key=key,
                )

        io = {"X": X_buf, "y": y_buf, "logits": plan.per_layer[-1].y}

        # ⬇️ 문서/테스트용 태그 전달
        tags = {
            "nvtx_capture_tag": "[DYN][CAPTURE]",
            "nvtx_replay_tag": "[DYN][REPLAY]",
            "path_fingerprint": tuple(ctx.get("path_fingerprint", ())),
            "branch_path": tuple(ctx.get("branch_path", ())),
        }
        tg = TrainGraph(gexec, io, stream, plan=plan, graph_key=key, tags=tags)

        entry = {
            "tg": tg,
            "loss_buf": loss_buf,
            "stream": stream,
            "optimizer": opt_for_path,
            "plan": plan,  # Dropout counter advance 등에 사용
        }
        self._tm["capture_count"] += 1  # === NEW ===
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
        """동적(Graph) 경로: If/Repeat/EarlyExit 포함한 '현재 실행된 경로'를 캡처/재생."""
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
        self._tm["replay_count"] += max(1, int(ctx.get("repeat_steps", 1)))  # === NEW ===

        return float(loss_buf.get())

    # ======== NEW: Frontend convenience APIs (fit/warmup/replay & pool tools) ========

    def fit(
        self,
        data_loader,
        *,
        loss,
        optimizer,
        ctx: Optional[Dict[str, Any]] = None,
        epochs: int = 1,
        use_dynamic: bool = True,
        static_input_shape: Optional[Tuple[int, ...]] = None,
        prewarm_samples: Optional[Sequence[Tuple[Any, Any, Dict[str, Any]]]] = None,
        report_every: int = 100,
    ):
        """
        케라스/파이토치 느낌의 고수준 학습 루프.
        - use_dynamic=True: one_step_dynamic 경로 사용(분기/반복 지원, on-demand capture)
        - use_dynamic=False: 정적 compile/one_step 사용(입력 shape 고정 필요)
        - prewarm_samples: [(X, y, ctx), ...] 형태로 미리 GraphKey를 캡처해 hit율↑
        """
        ctx = dict(ctx or {})

        if not use_dynamic:
            assert static_input_shape is not None, "static_input_shape is required for static fit"
            self.compile(static_input_shape, loss=loss, optimizer=optimizer)

        if prewarm_samples:
            for Xw, yw, cw in prewarm_samples:
                _ = self.one_step_dynamic(Xw, yw, loss=loss, optimizer=optimizer, ctx=cw)

        step = 0
        for ep in range(epochs):
            for X, y in data_loader:
                if use_dynamic:
                    loss_val = self.one_step_dynamic(X, y, loss=loss, optimizer=optimizer, ctx=ctx)
                else:
                    loss_val = self.one_step(X, y)
                step += 1
                if report_every and (step % report_every == 0):
                    print(f"[fit] epoch={ep} step={step} loss={loss_val:.6f}")

    def warmup(
        self,
        samples: Sequence[Tuple[Any, Any, Dict[str, Any]]],
        *,
        loss,
        optimizer,
    ) -> Dict[Tuple[Tuple[str, Any], ...], "TrainGraph"]:
        """
        여러 (X,y,ctx) 조합으로 GraphKey를 미리 캡처해 둠.
        반환: { variant_kv_tuple: TrainGraph }
        """
        out = {}
        for X, y, ctx in samples:
            _ = self.one_step_dynamic(X, y, loss=loss, optimizer=optimizer, ctx=ctx)
            var = tuple(sorted((str(k), self._freeze_value(v)) for k, v in dict(ctx.get("variant", {})).items()))
            out[var] = self.tg
        return out

    def replay_loop(
        self,
        batches: Iterable[Tuple[Any, Any]],
        *,
        steps: Optional[int] = None,
    ):
        """
        이미 캡처된 self.tg(TrainGraph)로 핫루프 실행.
        - set_batch() + launch()만 수행 → Python 오버헤드 최소화
        - 사전에 self.tg가 존재해야 함 (compile() 또는 warmup/one_step_dynamic()으로 생성)
        """
        assert self._tg is not None, "No captured graph. Call compile() or warmup/one_step_dynamic() first."
        n = 0
        for X, y in batches:
            self._tg.set_batch(cp.asarray(X), cp.asarray(y))
            self._tg.launch()
            n += 1
            if steps and n >= steps:
                break

    def pool_stats(self) -> Dict[str, Any]:
        """
        현재 그래프 풀 요약.
        - 글로벌 풀 사용 시: 크기만 노출(내부 구조 은닉 가정)
        - 폴백 풀 사용 시: 키 수/최종 사용 시각 등 요약
        """
        stats = {"global": False, "fallback_size": 0, "fallback_cap": self._FALLBACK_POOL_MAX}
        try:
            if graph_pool is not None and hasattr(graph_pool, "_store"):
                stats["global"] = True
                stats["global_size"] = len(getattr(graph_pool, "_store"))
        except Exception:
            pass
        try:
            from time import monotonic
            stats["fallback_size"] = len(_FALLBACK_POOL)
            if _FALLBACK_POOL:
                last_used = [v.get("last_used", 0.0) for v in _FALLBACK_POOL.values()]
                stats["fallback_oldest_sec"] = max(0.0, (monotonic() - min(last_used)))
        except Exception:
            pass

        stats["local_tm"] = dict(self._tm)

        return stats

    def get_graph_key_preview(self, X, *, ctx: Optional[Dict[str, Any]] = None, loss=None):
        """
        실제 캡처 없이 '현재 입력+컨텍스트'로 생성될 GraphKey를 미리 산출.
        - 디버그/로깅/메트릭에 유용
        """
        ctx = dict(ctx or {})
        sig = self._infer_signature(cp.asarray(X), ctx)
        return self._make_pool_key(sig, ctx, loss=loss)

    def evict_pool(self, *, predicate=None, max_remove: Optional[int] = None):
        """
        폴백 LRU 풀에서 조건부로 엔트리를 제거. (글로벌 풀은 운영 정책에 따름)
        predicate(key, entry) → bool 이 True인 항목만 제거.
        """
        removed = 0
        keys = list(_FALLBACK_POOL.keys())
        for k in keys:
            if predicate is None or predicate(k, _FALLBACK_POOL[k]):
                _FALLBACK_POOL.pop(k, None)
                removed += 1
                if max_remove and removed >= max_remove:
                    break
        return removed
    
    # === NEW ===
    def telemetry(self) -> Dict[str, int]:
        """로컬 Sequential 단위 텔레메트리 카운터 반환."""
        return dict(self._tm)

    # === NEW ===
    def reset_telemetry(self) -> None:
        """로컬 Sequential 텔레메트리 초기화."""
        for k in self._tm.keys():
            self._tm[k] = 0
