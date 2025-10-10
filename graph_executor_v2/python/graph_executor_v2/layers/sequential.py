# graph_executor_v2/layers/sequential.py
from __future__ import annotations
from typing import List, Tuple, Any, Iterable, Optional, Dict
import cupy as cp

from graph_executor_v2.graph.capture_plan import make_plan_for_sequential
from graph_executor_v2.graph.graph_exec import record_step_graph, TrainGraph
from graph_executor_v2.optim.rebind import try_rebind_grads

from .base import Layer

# ✅ 추가: 캡처 플랜 기반 grad 재바인딩 유틸
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

class Sequential(Layer):
    """
    - layers: [Layer, ...]
    - call(x): 순서대로 forward
    - backward(grad): 역순 전파
    - parameters(): (param, grad, name) 이터레이터
    - zero_grad(): grad 초기화
    - state_dict()/load_state_dict(): 저장/로드
    - train()/eval(): 서브 레이어까지 모드 전파
    - attach_grads(): 레이어 내부 dW/db 등을 param.grad로 연결(가능한 경우)
    - (추가) supports_capture(): 모든 레이어가 *_into를 지원하는지 확인
    - (추가) plan_capture(...): 캡처용 버퍼/워크스페이스 사전할당
    - (추가) record_graph_step(...): CUDA Graph 녹화 후 GraphExec 반환
    - (추가) compile(...): 전체 학습 1스텝(fwd→loss→bwd→opt) 그래프 컴파일
    """
    def __init__(self, *layers: Layer, name: Optional[str] = None):
        super().__init__(name=name)
        self.layers: List[Layer] = list(layers)
        self.training: bool = True  # 기본 학습 모드

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

    def build(self, input_shape: Tuple[int, ...]) -> None:
        cur = tuple(map(int, input_shape))
        for lyr in self.layers:
            try:
                lyr.build(cur)
            except Exception:
                pass
            try:
                cur = tuple(map(int, lyr.compute_output_shape(cur)))
            except Exception:
                pass
        self.input_shape = tuple(map(int, input_shape))
        self.output_shape = cur if isinstance(cur, tuple) else None
        self.built = True

    # === Runtime ===
    def call(self, x: Any):
        out = x
        for lyr in self.layers:
            # 드롭아웃/BN 등은 lyr.training 플래그를 참조하도록 구현
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
            # 파라미터 수 집계(가능한 경우)
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
        """model.train()/eval() 호환"""
        self.training = bool(mode)
        for lyr in self.layers:
            if hasattr(lyr, "training"):
                lyr.training = self.training
        return self

    def eval(self):
        return self.train(False)

    # 표준화된 파라미터 인터페이스
    def parameters(self) -> Iterable[Tuple[Any, Any, str]]:
        for idx, lyr in enumerate(self.layers):
            lname = f"{lyr.__class__.__name__}:{idx}"
            if hasattr(lyr, "parameters") and callable(getattr(lyr, "parameters")):
                # (param, grad, name) or (param, grad)
                for t in lyr.parameters():  # type: ignore
                    if isinstance(t, tuple) and len(t) == 3:
                        yield t  # (p, g, name)
                    elif isinstance(t, tuple) and len(t) == 2:
                        p, g = t
                        yield (p, g, lname)
                continue
            # 휴리스틱 스캔 (W/dW, b/db 등)
            for p_name, g_name in CANDIDATE_PARAM_GRAD_NAMES:
                if hasattr(lyr, p_name) and hasattr(lyr, g_name):
                    p = getattr(lyr, p_name)
                    g = getattr(lyr, g_name)
                    yield (p, g, f"{lname}.{p_name}")

    def zero_grad(self):
        """param.grad 및 레이어 내부 dW/db 등을 0으로."""
        # param.grad 초기화
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
                        # cupy.ndarray 는 임의 속성 부여가 안 될 수 있음 → 조용히 패스
                        pass
        # 레이어 내부 그라드 캐시 초기화
        for lyr in self.layers:
            # 사용자 정의 zero_grad가 있으면 우선
            if hasattr(lyr, "zero_grad") and callable(getattr(lyr, "zero_grad")):
                try:
                    lyr.zero_grad()  # type: ignore
                except Exception:
                    pass
                continue
            # 없으면 휴리스틱
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
        """
        레이어 내부 dW/db 등을 param.grad에 연결.
        - Optimizer가 param.grad를 읽는 구조(AdamW/SGD 등) 지원.
        - 역전파 직후 호출.
        - cupy.ndarray는 임의 속성 부여 실패 가능 → 실패 시 조용히 스킵
        """
        for (p, g, _) in self.parameters():
            if g is not None:
                try:
                    setattr(p, "grad", g)
                except Exception:
                    # 옵티마이저가 (param, grad) 튜플을 직접 받는다면 이 단계가 불필요
                    pass

    # === 저장/로드 ===
    def state_dict(self) -> dict:
        """
        레이어가 state_dict() 제공하면 위임.
        없으면 (W,b 등) 파라미터를 추출해 저장.
        """
        sd = {"name": self.name, "layers": []}
        for lyr in self.layers:
            entry = {"class": lyr.__class__.__name__}
            if hasattr(lyr, "state_dict"):
                try:
                    entry["state"] = lyr.state_dict()  # type: ignore
                except Exception:
                    entry["state"] = None
            else:
                # 휴리스틱 파라미터 스냅샷
                params = {}
                for p_name, _ in CANDIDATE_PARAM_GRAD_NAMES:
                    if hasattr(lyr, p_name):
                        params[p_name] = getattr(lyr, p_name)
                entry["params"] = params
            sd["layers"].append(entry)
        return sd

    def load_state_dict(self, sd: dict):
        """
        저장한 순서/타입이 동일하다는 가정.
        """
        assert len(sd.get("layers", [])) == len(self.layers), "layer count mismatch"
        for lyr, entry in zip(self.layers, sd["layers"]):
            if hasattr(lyr, "load_state_dict") and entry.get("state", None) is not None:
                try:
                    lyr.load_state_dict(entry["state"])  # type: ignore
                    continue
                except Exception:
                    pass
            # 휴리스틱 로드
            params = entry.get("params", {})
            for k, v in params.items():
                if hasattr(lyr, k):
                    try:
                        getattr(lyr, k)[...] = v
                    except Exception:
                        setattr(lyr, k, v)
        return self

    # =========================
    # ===== Graph Capture =====
    # =========================
    def supports_capture(self) -> bool:
        """
        모든 레이어가 forward_into/backward_into 를 제공하는지 검사.
        (pass-through 레이어는 forward_into만 제공해도 무방하지만,
         학습 캡처를 위해선 역전파 레이어들도 backward_into 필요)
        """
        ok = True
        for lyr in self.layers:
            f_ok = hasattr(lyr, "forward_into") and callable(getattr(lyr, "forward_into"))
            b_ok = hasattr(lyr, "backward_into") and callable(getattr(lyr, "backward_into"))
            ok = ok and f_ok and b_ok
        return ok

    def plan_capture(
        self,
        input_shape: Tuple[int, ...],
        *,
        loss_kind: str = "softmax_ce",
        lt_bytes: int = (8 << 20),  # 예: 8MB
    ) -> Dict[str, Any]:
        """
        캡처 실행에 필요한 사전 버퍼/워크스페이스를 할당하고 shape를 정리.
        현재는 Dense 레이어(= GEMM 기반) 위주로 워크스페이스를 준비.
        반환: dict { 'buffers': ..., 'workspaces': ..., 'shapes': ... }
        """
        from graph_executor_v2.ops import gemm as gops  # 지연 임포트
        cur = tuple(map(int, input_shape))
        bufs: Dict[str, Any] = {"fwd": [], "bwd": []}
        wss: Dict[str, Any] = []
        shapes: Dict[str, Any] = {"per_layer": []}

        # 순전파 출력/프리액티베이션 저장 여부 판단
        for idx, lyr in enumerate(self.layers):
            oname = f"L{idx}:{lyr.__class__.__name__}"
            try:
                out_shp = tuple(map(int, lyr.compute_output_shape(cur)))
            except Exception:
                out_shp = None
            shapes["per_layer"].append((oname, {"in": cur, "out": out_shp}))
            # Dense일 경우: Z 버퍼가 필요할 수 있음(activation != 'none')
            need_z = getattr(lyr, "activation", "none") != "none"
            # 순전파 out 버퍼
            if out_shp is not None:
                y = cp.empty(out_shp, dtype=cp.float32)
            else:
                raise RuntimeError(f"cannot infer output shape for {oname}")
            # Z(옵션)
            z = cp.empty(out_shp, dtype=cp.float32) if need_z else None

            bufs["fwd"].append({"y": y, "z": z, "name": oname})

            # 역전파 버퍼 (gA, gW, gB 등) — Dense 기준 휴리스틱
            # 레이어 내부에서 검증하므로 여기선 모양만 추정
            if hasattr(lyr, "W") and hasattr(lyr, "b"):
                W = getattr(lyr, "W")
                in_dim, units = int(W.shape[0]), int(W.shape[1])
                gA = cp.empty(cur, dtype=cp.float32)
                gW = cp.empty_like(W)
                gB = cp.empty((1, units), dtype=cp.float32)
                bufs["bwd"].append({"gA": gA, "gW": gW, "gB": gB, "name": oname})
                # GEMM 워크스페이스
                ws = gops.ensure_workspaces(cur[0], units, lt_bytes=lt_bytes)
                wss.append({"work": ws, "name": oname})
            else:
                # 파라미터가 없는 레이어(예: 활성화, reshape 등)
                gA = cp.empty(cur, dtype=cp.float32)
                bufs["bwd"].append({"gA": gA, "gW": None, "gB": None, "name": oname})
                wss.append({"work": None, "name": oname})

            # 다음 입력 shape 갱신
            cur = out_shp

        # 최종 로짓/손실용 dY 버퍼 (softmax CE 가정)
        if loss_kind == "softmax_ce":
            dY = cp.empty(cur, dtype=cp.float32)
            bufs["loss"] = {"dY": dY, "out_shape": cur}
        else:
            bufs["loss"] = {"dY": None, "out_shape": cur}

        return {"buffers": bufs, "workspaces": wss, "shapes": shapes}


    def record_graph_step(
        self,
        X: cp.ndarray,
        y: Any,
        *,
        loss_fn,
        optimizer_step_fn,
        capture_plan: Dict[str, Any],
        stream: Optional[cp.cuda.Stream] = None,
    ):
        """
        1 스텝 학습을 CUDA Graph로 녹화하고 GraphExec 유사 객체를 반환.
        - CUDA Graph 캡처 API가 환경에 없으면 동일 동작의 폴백 객체를 반환(launch만 제공).
        """
        if stream is None:
            stream = cp.cuda.Stream(non_blocking=True)

        bufs = capture_plan["buffers"]
        fwd_bufs = bufs["fwd"]
        bwd_bufs = bufs["bwd"]
        dY = bufs["loss"]["dY"]
        workspaces = capture_plan["workspaces"]

        # ---- 워밍업 1회(모양/타입/스트림 고정) ----
        with stream:
            cur = X
            # FWD
            for idx, lyr in enumerate(self.layers):
                ybuf = fwd_bufs[idx]["y"]
                zbuf = fwd_bufs[idx]["z"]
                lyr.forward_into(cur, out=ybuf, z_out=zbuf, stream=stream.ptr)
                cur = ybuf
            # LOSS (⚠️ 호스트 복사 금지)
            loss_dev, dY_tmp = loss_fn.forward(cur, y, return_scalar=False)
            if dY is not None:
                dY[...] = dY_tmp

            # ✅ BWD 전에 캡처 grad 버퍼를 0으로 초기화 (누적 방지)
            for i in range(len(self.layers)):
                b = bwd_bufs[i]
                ga = b.get("gA", None)
                gw = b.get("gW", None)
                gb = b.get("gB", None)
                if ga is not None: ga.fill(0)
                if gw is not None: gw.fill(0)
                if gb is not None: gb.fill(0)

            # BWD (역순)
            g = dY if dY is not None else dY_tmp
            for ridx, lyr in enumerate(reversed(self.layers)):
                i = len(self.layers) - 1 - ridx
                b = bwd_bufs[i]
                ws = workspaces[i]["work"]
                if b["gW"] is not None:
                    lyr.backward_into(
                        g,
                        gA_out=b["gA"], gW_out=b["gW"], gB_out=b["gB"],
                        work_dZ=(ws.dZ if ws is not None else None),
                        lt_workspace=(ws.lt_ws if (ws is not None and ws.lt_ws is not None) else None),
                        stream=stream.ptr
                    )
                else:
                    lyr.backward_into(  # type: ignore
                        g, gA_out=b["gA"],
                        work_dZ=None, lt_workspace=None, stream=stream.ptr
                    )
                g = b["gA"]
            # OPT
            optimizer_step_fn()

        # ---- 그래프 캡처 지원 여부 확인 ----
        _has_capture_stream = hasattr(cp.cuda, "graph") and hasattr(cp.cuda.graph, "capture_stream")

        if _has_capture_stream:
            # ========== 진짜 CUDA Graph 경로 ==========
            with stream:
                with cp.cuda.graph.capture_stream(stream) as cap:
                    cur = X
                    # FWD
                    for idx, lyr in enumerate(self.layers):
                        ybuf = fwd_bufs[idx]["y"]
                        zbuf = fwd_bufs[idx]["z"]
                        lyr.forward_into(cur, out=ybuf, z_out=zbuf, stream=stream.ptr)
                        cur = ybuf
                    # LOSS (⚠️ 호스트 복사 금지)
                    loss_dev, dY_tmp = loss_fn.forward(cur, y, return_scalar=False)
                    if dY is not None:
                        dY[...] = dY_tmp
                        g_in = dY
                    else:
                        g_in = dY_tmp

                    # ✅ BWD 전에 캡처 grad 버퍼를 0으로 초기화 (누적 방지)
                    for i in range(len(self.layers)):
                        b = bwd_bufs[i]
                        ga = b.get("gA", None)
                        gw = b.get("gW", None)
                        gb = b.get("gB", None)
                        if ga is not None: ga.fill(0)
                        if gw is not None: gw.fill(0)
                        if gb is not None: gb.fill(0)

                    # BWD
                    for ridx, lyr in enumerate(reversed(self.layers)):
                        i = len(self.layers) - 1 - ridx
                        b = bwd_bufs[i]
                        ws = workspaces[i]["work"]
                        if b["gW"] is not None:
                            lyr.backward_into(
                                g_in,
                                gA_out=b["gA"], gW_out=b["gW"], gB_out=b["gB"],
                                work_dZ=(ws.dZ if ws is not None else None),
                                lt_workspace=(ws.lt_ws if (ws is not None and ws.lt_ws is not None) else None),
                                stream=stream.ptr
                            )
                        else:
                            lyr.backward_into(  # type: ignore
                                g_in, gA_out=b["gA"],
                                work_dZ=None, lt_workspace=None, stream=stream.ptr
                            )
                        g_in = b["gA"]
                    # OPT
                    optimizer_step_fn()

            graph = cap.graph
            gexec = graph.instantiate()
            return gexec

        # ========== 폴백: 그래프 미지원 시 GraphExec 유사 객체 반환 ==========
        def _one_step():
            # 주의: 스트림/버퍼/shape는 이미 고정되어 있다고 가정(워밍업 동일 순서)
            cur = X
            for idx, lyr in enumerate(self.layers):
                ybuf = fwd_bufs[idx]["y"]
                zbuf = fwd_bufs[idx]["z"]
                lyr.forward_into(cur, out=ybuf, z_out=zbuf, stream=stream.ptr)
                cur = ybuf
            loss_dev, dY_tmp = loss_fn.forward(cur, y, return_scalar=False)
            g_in = dY if dY is not None else dY_tmp

            # ✅ BWD 전에 캡처 grad 버퍼를 0으로 초기화 (누적 방지)
            for i in range(len(self.layers)):
                b = bwd_bufs[i]
                ga = b.get("gA", None)
                gw = b.get("gW", None)
                gb = b.get("gB", None)
                if ga is not None: ga.fill(0)
                if gw is not None: gw.fill(0)
                if gb is not None: gb.fill(0)

            for ridx, lyr in enumerate(reversed(self.layers)):
                i = len(self.layers) - 1 - ridx
                b = bwd_bufs[i]
                ws = workspaces[i]["work"]
                if b["gW"] is not None:
                    lyr.backward_into(
                        g_in,
                        gA_out=b["gA"], gW_out=b["gW"], gB_out=b["gB"],
                        work_dZ=(ws.dZ if ws is not None else None),
                        lt_workspace=(ws.lt_ws if (ws is not None and ws.lt_ws is not None) else None),
                        stream=stream.ptr
                    )
                else:
                    lyr.backward_into(  # type: ignore
                        g_in, gA_out=b["gA"],
                        work_dZ=None, lt_workspace=None, stream=stream.ptr
                    )
                g_in = b["gA"]
            optimizer_step_fn()

        class _PseudoGraphExec:
            """CUDA Graph이 없을 때를 위한 간단한 대체. launch(stream_ptr)를 흉내낸다."""
            def __init__(self, step_fn, stream_obj):
                self._step_fn = step_fn
                self._stream = stream_obj
            def launch(self, stream_ptr=None):
                # stream_ptr는 무시(이미 self._stream로 고정)
                with self._stream:
                    self._step_fn()

        return _PseudoGraphExec(_one_step, stream)

    # =========================
    # ======== Compile ========
    # =========================
    def compile(
        self,
        input_shape,
        *,
        loss,
        optimizer,
        lt_bytes: int = (8 << 20),
        stream: Optional[cp.cuda.Stream] = None,
    ) -> "TrainGraph":
        """
        얇은 래퍼: plan 생성 → optimizer grad rebind → graph 녹화 → TrainGraph 반환
        """
        if not self.built:
            self.build(tuple(map(int, input_shape)))

        assert self.supports_capture(), "All layers must implement forward_into/backward_into for capture"

        if stream is None:
            stream = cp.cuda.Stream(non_blocking=True)

        plan = make_plan_for_sequential(
            self, tuple(map(int, input_shape)),
            loss_kind="softmax_ce", lt_bytes=lt_bytes
        )
        try_rebind_grads(self, optimizer, plan)

        bs, in_dim = int(input_shape[0]), int(input_shape[1])
        X_buf = cp.zeros((bs, in_dim), dtype=cp.float32)
        y_buf = cp.zeros((bs,), dtype=cp.int32)

        gexec = record_step_graph(
            self, loss, optimizer.step_into,
            plan, stream=stream
        )
        io = {"X": X_buf, "y": y_buf, "logits": plan.per_layer[-1].y}
        return TrainGraph(gexec, io, stream)
