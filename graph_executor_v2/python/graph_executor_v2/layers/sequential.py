# python/graph_executor_v2/layers/sequential.py
from __future__ import annotations
from typing import (
    List, Tuple, Any, Iterable, Optional, Dict, Sequence, TYPE_CHECKING
)
import cupy as cp

from graph_executor_v2.graph.capture_plan import (
    make_plan_for_sequential,
    make_plan_for_path,
)
from graph_executor_v2.graph.graph_exec import record_step_graph, TrainGraph
from graph_executor_v2.optim.rebind import try_rebind_grads
from .base import Layer

import inspect

# ✅ 런타임 임포트 우선: 실제 클래스 로드 실패시 스텁으로 폴백
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
    from typing import Any as _AnyType
    GraphSignature = _AnyType  # type: ignore[assignment]
    GraphKey = _AnyType        # type: ignore[assignment]
    MultiGraphPool = _AnyType  # type: ignore[assignment]

# 런타임 인스턴스 로딩 (없어도 동작하도록 폴백 준비)
try:
    from graph_executor_v2.graph.graph_executor import graph_pool  # type: ignore
except Exception:
    graph_pool = None  # type: ignore

# 폴백: 프로세스 내 간단한 캐시(dict)
_FALLBACK_POOL: Dict[Any, Any] = {}

# ✅ 추가: 캡처 플랜 기반 grad 재바인딩 유틸(선택)
try:
    from graph_executor_v2.optim.adamw import collect_params_from_plan  # type: ignore
except Exception:
    collect_params_from_plan = None  # 런타임에 체크

CANDIDATE_PARAM_GRAD_NAMES = [
    ("W", "dW"),
    ("weight", "dweight"),
    ("b", "db"),
    ("bias", "dbias"),
]

# 동적 경로 호환용: 옛 graph_exec에도 동작하도록 하는 프록시
class _ModelLayersProxy:
    """model의 다른 속성은 그대로 위임하고, layers만 path_layers로 바꿔치기."""
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
    def __init__(self, *layers: Layer, name: Optional[str] = None):
        super().__init__(name=name)
        self.layers: List[Layer] = list(layers)
        self.training: bool = True

        # captured session (static)
        self._tg: Optional[TrainGraph] = None
        self._loss_buf: Optional[cp.ndarray] = None
        self._stream: Optional[cp.cuda.Stream] = None

    def add(self, layer: Layer) -> None:
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

    # === Eager ===
    def call(self, x: Any):
        out = x
        for lyr in self.layers:
            if hasattr(lyr, "training"):
                lyr.training = self.training
            out = lyr(out)
        return out

    def backward(self, grad_output: Any):
        g = grad_output
        for lyr in reversed(self.layers):
            g = lyr.backward(g)
        return g

    def compute_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        cur = tuple(map(int, input_shape))
        for lyr in self.layers:
            cur = lyr.compute_output_shape(cur)
        return cur

    def summary(self, indent: int = 2) -> str:
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

    # === Training helpers ===
    def train(self, mode: bool = True):
        self.training = bool(mode)
        for lyr in self.layers:
            if hasattr(lyr, "training"):
                lyr.training = self.training
        return self

    def eval(self):
        return self.train(False)

    def parameters(self) -> Iterable[Tuple[Any, Any, str]]:
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
        in_shape = tuple(map(int, input_shape))
        if not self.built:
            self.build(in_shape)

        assert self.supports_capture(), "All layers must implement forward_into/backward_into for capture"

        if stream is None:
            stream = cp.cuda.Stream(non_blocking=True)

        plan = make_plan_for_sequential(
            self, in_shape, loss_kind="softmax_ce", lt_bytes=lt_bytes
        )

        try_rebind_grads(self, optimizer, plan)

        X_buf = cp.zeros(in_shape, dtype=cp.float32)
        N = int(in_shape[0])
        y_buf = cp.zeros((N,), dtype=cp.int32)
        loss_buf = cp.zeros((), dtype=cp.float32)

        gexec = record_step_graph(
            self,
            loss,
            optimizer.step_into,
            plan,
            X_buf=X_buf,
            y_buf=y_buf,
            stream=stream,
            loss_out=loss_buf,
        )

        io = {"X": X_buf, "y": y_buf, "logits": plan.per_layer[-1].y}
        tg = TrainGraph(gexec, io, stream)

        self._tg = tg
        self._loss_buf = loss_buf
        self._stream = stream
        return tg

    def one_step(self, X, y) -> float:
        assert self._tg is not None, "call compile() first"
        assert self._loss_buf is not None, "loss buffer not initialized"

        xb, yb = self._tg.X_buf, self._tg.y_buf
        x_arr = cp.asarray(X)
        y_arr = cp.asarray(y)

        assert tuple(xb.shape) == tuple(x_arr.shape), f"X shape mismatch: {x_arr.shape} vs {xb.shape}"
        assert yb.shape == (xb.shape[0],), f"y shape must be (N,), got {yb.shape} vs N={xb.shape[0]}"
        assert yb.dtype == cp.int32, f"labels must be int32 for current CE kernel (got {yb.dtype})"

        self._tg.set_batch(x_arr, y_arr)
        self._tg.launch()
        return float(self._loss_buf.get())

    @property
    def tg(self) -> TrainGraph:
        assert self._tg is not None, "call compile() first"
        return self._tg

    # =========================================================
    # ========== Dynamic Path Handling (분기/반복) ============
    # =========================================================

    def _infer_signature(self, X, ctx: Dict[str, Any]) -> "GraphSignature":
        from typing import Any as _AnyType
        if GraphSignature is _AnyType:  # type: ignore[comparison-overlap]
            # 그래프 풀 타입 모듈이 없어도 동작하도록: 서명은 단순 dict로 표현
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
        if graph_pool is not None and hasattr(graph_pool, "get"):
            try:
                return graph_pool.get(key)  # type: ignore[attr-defined]
            except Exception:
                return None
        return _FALLBACK_POOL.get(key)

    def _pool_put(self, key: Any, entry: Any) -> None:
        if graph_pool is not None and hasattr(graph_pool, "put"):
            try:
                graph_pool.put(key, entry)  # type: ignore[attr-defined]
                return
            except Exception:
                pass
        _FALLBACK_POOL[key] = entry

    def _make_pool_key(self, sig: Any, ctx: Dict[str, Any]) -> Any:
        branch_id = ctx.get("branch", "default")
        # variant는 정렬된 튜플로 동결
        vdict = dict(ctx.get("variant", {}))
        variant = tuple(sorted((str(k), self._freeze_value(v)) for k, v in vdict.items()))
        # GraphKey가 있으면 써도 되고, 없어도 단순 튜플로 충분
        try:
            if GraphKey not in (None, object):  # 약식 가드
                return GraphKey(signature=sig, branch_id=str(branch_id), variant=variant)  # type: ignore[call-arg]
        except Exception:
            pass
        # 폴백 키(해시 가능한 튜플)
        return ("dyn", tuple(getattr(sig, "shape", ())), str(getattr(sig, "dtype", "")),
                str(getattr(sig, "layout", "")), str(branch_id), variant)

    @staticmethod
    def _freeze_value(v: Any) -> Any:
        if isinstance(v, (str, int, float, bool, type(None))):
            return v
        if isinstance(v, (tuple, list)):
            return tuple(Sequential._freeze_value(x) for x in v)
        if isinstance(v, dict):
            return tuple(sorted((str(k), Sequential._freeze_value(val)) for k, val in v.items()))
        return str(v)

    def _linearize_path(self, X, ctx: Dict[str, Any]) -> List[Layer]:
        """
        동적 제어 레이어(If/Repeat/EarlyExit)를 덕타이핑으로 감지해
        실제 연산 레이어 시퀀스로 평탄화한다.
        - If:      decide(X, ctx) -> (branch_id, block)
        - Repeat:  steps(X, ctx) -> T, 본문 body는 1step만 캡처하고 실행 시 T회 launch
        - EarlyExit: stages를 순차적으로 연결 (exit_fn은 외부 정책)
        """
        def _is_if(obj):
            return callable(getattr(obj, "decide", None)) and \
                hasattr(obj, "then_block") and hasattr(obj, "else_block")

        def _is_repeat(obj):
            return callable(getattr(obj, "steps", None)) and hasattr(obj, "body")

        def _is_early(obj):
            return hasattr(obj, "stages") and isinstance(getattr(obj, "stages"), (list, tuple))

        linear: List[Layer] = []
        for l in self.layers:
            if _is_if(l):
                branch, block = l.decide(X, ctx)
                ctx["branch"] = branch
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
                for s in l.stages:
                    if isinstance(s, Sequential):
                        linear.extend(s.layers)
                    else:
                        linear.append(s)
                ctx["earlyexit"] = True

            else:
                linear.append(l)

        # ✅ 컨트롤 레이어 잔존 가드 (평탄화 누락 방지)
        leftovers = []
        for x in linear:
            if _is_if(x) or _is_repeat(x) or _is_early(x):
                leftovers.append(type(x).__name__)
        if leftovers:
            raise RuntimeError(
                f"[dynamic] control layers must be flattened, but found in path: {leftovers}"
            )
        return linear

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
        # 1) 경로 평탄화
        path_layers = self._linearize_path(X, ctx)

        # 2) 키 구성
        sig = self._infer_signature(X, ctx)
        key = self._make_pool_key(sig, ctx)

        # 3) 풀 조회
        entry = self._pool_get(key)
        if entry is not None:
            return entry

        # 4) 신규 캡처
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
        plan = make_plan_for_path(path_layers, in_shape, loss_kind="softmax_ce", lt_bytes=lt_bytes)

        # ---- 경로 전용 (param, grad) 트리플 수집: 정확 매핑 ----
        def _collect_triplets_from_path(plan, layers):
            triplets = []
            for i, lyr in enumerate(layers):
                per = plan.per_layer[i]
                # Dense/Conv 공통
                if hasattr(lyr, "W") and per.gW is not None:
                    triplets.append((getattr(lyr, "W"), per.gW, f"{type(lyr).__name__}:{i}.W"))
                for b_name in ("b", "bias", "B"):
                    if hasattr(lyr, b_name) and getattr(lyr, b_name) is not None and per.gB is not None:
                        triplets.append((getattr(lyr, b_name), per.gB, f"{type(lyr).__name__}:{i}.{b_name}"))
                        break
                # BN
                if hasattr(lyr, "gamma") and per.gW is not None:
                    try:
                        if tuple(getattr(lyr, "gamma").shape) == tuple(per.gW.shape):
                            triplets.append((getattr(lyr, "gamma"), per.gW, f"BN2d:{i}.gamma"))
                    except Exception:
                        pass
                if hasattr(lyr, "beta") and per.gB is not None:
                    try:
                        if tuple(getattr(lyr, "beta").shape) == tuple(per.gB.shape):
                            triplets.append((getattr(lyr, "beta"), per.gB, f"BN2d:{i}.beta"))
                    except Exception:
                        pass
                # RNN
                for w_name, g_name, tag in (("Wx", "gWx", "Wx"), ("Wh", "gWh", "Wh")):
                    if hasattr(lyr, w_name) and getattr(per, g_name, None) is not None:
                        triplets.append((getattr(lyr, w_name), getattr(per, g_name), f"RNN:{i}.{tag}"))
                if hasattr(lyr, "b") and getattr(per, "gB", None) is not None and getattr(lyr, "b") is not None:
                    triplets.append((getattr(lyr, "b"), per.gB, f"RNN:{i}.b"))
            return triplets

        triplets = _collect_triplets_from_path(plan, path_layers)

        # ---- 옵티마이저 바인딩 (경로별 옵티마이저 생성/캐시) ----
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

        # 5) 고정 I/O 버퍼
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
            )
        else:
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
            )

        io = {"X": X_buf, "y": y_buf, "logits": plan.per_layer[-1].y}
        tg = TrainGraph(gexec, io, stream)

        entry = {"tg": tg, "loss_buf": loss_buf, "stream": stream, "optimizer": opt_for_path}
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
        """
        동적 경로(If/Repeat/EarlyExit 포함)를 입력/컨텍스트에 따라 평탄화-캡처-재생한다.
        - 경로별 그래프를 내부 풀에 캐시하며, 실행 후 해당 경로의 TrainGraph를
          self._tg / self._loss_buf / self._stream 에 연결한다.
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

        # ✅ 현재 동적 경로 그래프 핸들을 모델 수준 핸들로 노출
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
        for _ in range(max(1, T)):
            tg.launch()

        # 손실 스칼라 반환
        return float(loss_buf.get())
