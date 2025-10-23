# python/graph_executor_v2/graph/graph_executor.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
import cupy as cp
import math

from graph_executor_v2.layers.conv2d import Conv2D
from graph_executor_v2.layers.dense_gemm import Dense
from graph_executor_v2.ops import conv2d as conv_ops
from graph_executor_v2.ops import gemm as gemm_ops

# ======================== IR ========================

@dataclass
class TensorDesc:
    shape: Tuple[int, ...]
    dtype: Any = cp.float32
    buffer_id: int = -1
    offset: int = 0  # element offset (arena 1D)

@dataclass
class Node:
    kind: str
    inputs: List[int]
    outputs: List[int]
    attrs: Dict[str, Any] = field(default_factory=dict)
    layer_ref: Any = None
    ws_fwd: Optional[Any] = None       # Conv2DWorkspaces(forward)
    ws_bwd: Optional[Any] = None       # Conv2DWorkspaces(backward) or optional

@dataclass
class GraphIR:
    tensors: List[TensorDesc] = field(default_factory=list)
    nodes: List[Node] = field(default_factory=list)
    input_ids: List[int] = field(default_factory=list)
    output_ids: List[int] = field(default_factory=list)

# ======================== Memory Arena ========================

class MemoryArena:
    def __init__(self):
        self.buffers: Dict[int, cp.ndarray] = {}
        self.alloc_elems: Dict[int, int] = {}

    def allocate(self, buffer_id: int, numel: int, dtype=cp.float32):
        need = int(numel)
        if buffer_id in self.buffers:
            if self.alloc_elems[buffer_id] < need:
                self.buffers[buffer_id] = cp.empty(need, dtype=dtype)
                self.alloc_elems[buffer_id] = need
        else:
            self.buffers[buffer_id] = cp.empty(need, dtype=dtype)
            self.alloc_elems[buffer_id] = need

    def view(self, desc: TensorDesc) -> cp.ndarray:
        buf = self.buffers[desc.buffer_id]
        start = int(desc.offset)
        numel = int(math.prod(int(s) for s in desc.shape))
        end = start + numel
        segment = buf[start:end]
        # 안전장치: 버퍼 dtype과 요청 dtype이 다르면 재해석 대신 필요한 구간만 캐스팅
        if segment.dtype != desc.dtype:
            segment = segment.astype(desc.dtype, copy=False)
        return segment.reshape(tuple(int(s) for s in desc.shape))

def _numel(shape: Tuple[int, ...]) -> int:
    return int(math.prod(int(s) for s in shape))

def plan_memory(ir: GraphIR) -> MemoryArena:
    arena = MemoryArena()
    per_buf_need: Dict[int, int] = {}
    per_buf_dtype: Dict[int, Any] = {}
    for t in ir.tensors:
        need = t.offset + _numel(t.shape)
        bid = t.buffer_id
        per_buf_need[bid] = max(per_buf_need.get(bid, 0), need)
        if bid not in per_buf_dtype:
            per_buf_dtype[bid] = t.dtype
        else:
            # 같은 buffer_id에 서로 다른 dtype 혼용 금지
            if per_buf_dtype[bid] != t.dtype:
                raise TypeError(f"Mixed dtypes in buffer {bid}: {per_buf_dtype[bid]} vs {t.dtype}")
    for bid, need in per_buf_need.items():
        arena.allocate(bid, need, dtype=per_buf_dtype.get(bid, cp.float32))
    return arena

# ======================== Compiler ========================

class GraphCompiler:
    def __init__(self, model) -> None:
        self._use_runtime_graph_api: bool = True
        self.model = model
        self.ir: Optional[GraphIR] = None
        self.arena: Optional[MemoryArena] = None

        # FWD capture
        self._gexec: Optional[Any] = None
        self._cap_stream: Optional[cp.cuda.Stream] = None
        self._graph_api_kind: str = "unknown"

        # BWD capture
        self._gexec_bwd: Optional[Any] = None
        self._cap_stream_bwd: Optional[cp.cuda.Stream] = None
        self._graph_api_kind_bwd: str = "none"
        self._in_bwd_capture: bool = False        # 캡처 중 플래그
        self._bwd_capture_error: Optional[str] = None
        self._last_bwd_node: Optional[str] = None # 최근 실행한 BWD 노드(디버그)

        # Grad buffers (per tensor id)
        self._grad_bufs: List[Optional[cp.ndarray]] = []

    def _grad_view(self, tid: int) -> cp.ndarray:
        buf = self._grad_bufs[tid]
        assert buf is not None
        return buf

    def compile(self, input_shape: Tuple[int, ...],
                use_cuda_graph: bool = False,
                use_cuda_graph_bwd: bool = False):
        ir = GraphIR()
        tid = 0
        ir.tensors.append(TensorDesc(shape=input_shape, buffer_id=0, offset=0))
        ir.input_ids = [tid]
        cur_tid = tid
        cur_shape = input_shape
        tid += 1
        next_buf_id = 1

        for layer in self.model.layers:
            lname = layer.__class__.__name__

            if lname == "Conv2D":
                N, Cin, H, W = cur_shape
                KH, KW = layer.kernel_size
                sH, sW  = layer.stride
                pH, pW  = layer.padding
                dH, dW  = layer.dilation
                H_out = (H + 2*pH - dH*(KH-1) - 1)//sH + 1
                W_out = (W + 2*pW - dW*(KW-1) - 1)//sW + 1
                Cout  = layer.out_channels
                y_shape = (N, Cout, H_out, W_out)

                ir.tensors.append(TensorDesc(shape=y_shape, buffer_id=next_buf_id, offset=0))
                y_tid = tid; tid += 1

                node = Node(
                    kind="Conv2D",
                    inputs=[cur_tid],
                    outputs=[y_tid],
                    attrs=dict(
                        stride=layer.stride, padding=layer.padding, dilation=layer.dilation,
                        groups=layer.groups, with_bias=layer.use_bias, act="none", save_z=True
                    ),
                    layer_ref=layer,
                )

                # === WS (fwd/bwd) 사전할당 ===
                CinW = int(layer.W.shape[1]) if hasattr(layer, "W") and layer.W is not None else int(Cin // max(1, layer.groups))
                K    = int(CinW * KH * KW)
                HWo  = int(H_out * W_out)

                # Forward WS
                wf = conv_ops.Conv2DWorkspaces()
                wf.dCol   = cp.empty((HWo, K),    dtype=cp.float32)
                wf.W_KC   = cp.empty((K,   Cout), dtype=cp.float32)
                wf.Y_tmp  = cp.empty((HWo, Cout), dtype=cp.float32)
                wf.Z_rows = cp.empty((HWo, Cout), dtype=cp.float32)  # save_z=True
                node.ws_fwd = wf

                # Backward WS (gX, gW, gB 모두 켜는 일반 케이스)
                wb = conv_ops.Conv2DWorkspaces()
                wb.dCol_b  = cp.empty((HWo, K), dtype=cp.float32)
                wb.dTmp    = cp.empty((max(Cout*K, HWo*K),), dtype=cp.float32)
                wb.gy_rows = cp.empty((Cout, HWo), dtype=cp.float32)
                wb.Z_rows_b= cp.empty((Cout, HWo), dtype=cp.float32)
                wb.W_CK    = cp.empty((Cout, K), dtype=cp.float32)      # want_gX
                wb.dY_HT   = cp.empty((HWo,  Cout), dtype=cp.float32)   # want_gX
                wb.dWpack  = cp.empty((Cout, K), dtype=cp.float32)      # want_gW
                node.ws_bwd = wb

                # 파라미터 grad 버퍼 보장
                if not hasattr(layer, "dW") or layer.dW is None:
                    layer.dW = cp.empty_like(layer.W)
                if layer.use_bias:
                    if not hasattr(layer, "db") or layer.db is None:
                        layer.db = cp.empty_like(layer.b)

                ir.nodes.append(node)
                cur_tid = y_tid
                cur_shape = y_shape
                next_buf_id += 1

            elif lname in ("ReLU",):
                in_desc = ir.tensors[cur_tid]
                ir.tensors.append(TensorDesc(shape=in_desc.shape, buffer_id=in_desc.buffer_id, offset=in_desc.offset))
                y_tid = tid; tid += 1
                node = Node(kind="ReLU", inputs=[cur_tid], outputs=[y_tid], layer_ref=layer)
                # ★ 캡처-세이프: ReLU 마스크는 bwd에서 (Y>0)로 생성/사용. fwd는 최대연산만.
                mask_shape = tuple(int(v) for v in in_desc.shape)
                node.attrs["relu_mask"] = cp.empty(mask_shape, dtype=cp.bool_)
                ir.nodes.append(node)
                cur_tid = y_tid

            elif lname in ("BatchNorm2D", "BatchNorm2d", "BatchNorm"):
                in_desc = ir.tensors[cur_tid]
                ir.tensors.append(TensorDesc(shape=in_desc.shape, buffer_id=next_buf_id, offset=0))
                y_tid = tid; tid += 1
                ir.nodes.append(Node(kind="BN2D", inputs=[cur_tid], outputs=[y_tid],
                                     attrs=dict(eps=getattr(layer, "eps", 1e-5)), layer_ref=layer))
                cur_tid = y_tid
                next_buf_id += 1

            elif lname == "Flatten":
                N, C, H, W = cur_shape
                new_shape = (N, C*H*W)
                in_desc = ir.tensors[cur_tid]
                ir.tensors.append(TensorDesc(shape=new_shape, buffer_id=in_desc.buffer_id, offset=in_desc.offset))
                y_tid = tid; tid += 1
                ir.nodes.append(Node(kind="Flatten", inputs=[cur_tid], outputs=[y_tid], layer_ref=layer))
                cur_tid = y_tid
                cur_shape = new_shape

            elif lname in ("Dense", "Linear"):
                N, K = cur_shape
                Nout = int(getattr(layer, "units", None) or getattr(layer, "out_features"))
                y_shape = (N, Nout)

                # 출력 및 pre-activation(Z) 텐서 등록
                ir.tensors.append(TensorDesc(shape=y_shape, buffer_id=next_buf_id, offset=0))
                y_tid = tid; tid += 1
                ir.tensors.append(TensorDesc(shape=y_shape, buffer_id=next_buf_id, offset=_numel(y_shape)))  # Z
                z_tid = tid; tid += 1

                node = Node(
                    kind="GEMM",
                    inputs=[cur_tid],
                    outputs=[y_tid],
                    attrs=dict(
                        act=getattr(layer, "activation", "none"),
                        with_bias=bool(getattr(layer, "use_bias", True)),
                        z_tid=z_tid
                    ),
                    layer_ref=layer
                )

                # 파라미터 grad 버퍼
                if not hasattr(layer, "dW") or layer.dW is None:
                    layer.dW = cp.empty_like(layer.W)
                if bool(getattr(layer, "use_bias", True)):
                    if not hasattr(layer, "db") or layer.db is None:
                        layer.db = cp.empty_like(layer.b)
                    # GEMM gBias는 (1,Nout) 필요 → 노드별 row 버퍼
                    node.attrs["gBias_row"] = cp.empty((1, Nout), dtype=cp.float32)

                # --- GEMM BWD 워크스페이스(캡처-세이프) ---
                node.attrs["gemm_ws_dZ"] = cp.empty(y_shape, dtype=cp.float32)          # (M,N)
                node.attrs["gemm_lt_ws"] = cp.empty(8 * 1024 * 1024, dtype=cp.uint8)     # 8MB (optional)

                ir.nodes.append(node)
                cur_tid = y_tid
                cur_shape = y_shape
                next_buf_id += 1

            else:
                raise NotImplementedError(f"Unsupported layer in graph compile: {lname}")

        ir.output_ids = [cur_tid]
        self.ir = ir

        # 메모리 플래닝
        self.arena = plan_memory(ir)

        # Grad 버퍼 사전할당 (모든 텐서)
        self._grad_bufs = []
        for t in ir.tensors:
            g = cp.empty(tuple(int(s) for s in t.shape), dtype=cp.float32)
            self._grad_bufs.append(g)

        # (옵션) Forward CUDA Graph 캡처
        if use_cuda_graph:
            warm_x = cp.empty(input_shape, dtype=cp.float32)
            _ = self._run_impl(warm_x)
            cp.cuda.get_current_stream().synchronize()

            self._cap_stream = cp.cuda.Stream(non_blocking=True)
            with self._cap_stream:
                self._cap_stream.begin_capture()
                _ = self._run_impl(None)
                graph = self._cap_stream.end_capture()

            self._use_runtime_graph_api = False
            self._graph_api_kind = "unknown"
            try:
                if hasattr(graph, "upload") and hasattr(graph, "launch"):
                    graph.upload(self._cap_stream)
                    self._gexec = graph
                    self._graph_api_kind = "graph-object"
                else:
                    raise AttributeError("no-graph-object-methods")
            except AttributeError:
                try:
                    cp.cuda.graph.upload(graph, self._cap_stream)
                    self._gexec = graph
                    self._graph_api_kind = "module-func"
                except Exception:
                    raw = getattr(graph, "graph", None) or getattr(graph, "ptr", None)
                    if raw is None:
                        raise TypeError("This CuPy version exposes neither Graph.launch/upload nor graph/ptr.")
                    gexec = cp.cuda.runtime.graphInstantiate(int(raw))
                    if isinstance(gexec, tuple):
                        gexec = gexec[0]
                    self._gexec = int(gexec)
                    cp.cuda.runtime.graphUpload(self._gexec, self._cap_stream.ptr)
                    self._use_runtime_graph_api = True
                    self._graph_api_kind = "runtime-pointer"

        # (옵션) Backward CUDA Graph 캡처
        if use_cuda_graph_bwd:
            # 1) fwd 한 번 돌려서 Z 등 채우고
            warm_x = cp.empty(input_shape, dtype=cp.float32)
            _ = self._run_impl(warm_x)

            # 2) gout 워밍업 + 비캡처 한 번 (워크스페이스들 확정)
            out_tid = self.ir.output_ids[0]
            gout_warm = self._grad_view(out_tid); gout_warm.fill(1.0)
            # 핸들/플랜을 미리 초기화(내부에서 작은 더미 GEMM 실행)
            self._warmup_gemm_handles()
            _ = self._backward_impl(None)
            cp.cuda.get_current_stream().synchronize()

            # 3) 스트림은 try 밖에서 생성!
            self._bwd_capture_error = None
            self._cap_stream_bwd = cp.cuda.Stream(non_blocking=True)
            try:
                # 3-a) **같은 캡처 스트림에서** 미리 한 번 더 워밍업(캡처 없이)
                with self._cap_stream_bwd:
                    self._in_bwd_capture = False
                    self._warmup_gemm_handles()   # cuBLAS/cuBLASLt lazy init on this stream
                    _ = self._backward_impl(None) # Conv/GEMM/ReLU bwd 전부 이 스트림에서 한 번 실행
                    cp.cuda.get_current_stream().synchronize()

                # 3-b) 이제 캡처 시작
                with self._cap_stream_bwd:
                    self._in_bwd_capture = True
                    self._cap_stream_bwd.begin_capture()
                    try:
                        _ = self._backward_impl(None)  # gout은 grad_buf에 이미 있음
                    except Exception as e:
                        where = getattr(self, "_last_bwd_node", "?")
                        self._bwd_capture_error = f"{type(e).__name__} at node {where}: {e}"
                        raise
                    graph_b = self._cap_stream_bwd.end_capture()

                # 업로드/런치 방식 분기 (FWD와 동일 패턴)
                if hasattr(graph_b, "upload") and hasattr(graph_b, "launch"):
                    graph_b.upload(self._cap_stream_bwd)
                    self._gexec_bwd = graph_b
                    self._graph_api_kind_bwd = "graph-object"
                else:
                    try:
                        cp.cuda.graph.upload(graph_b, self._cap_stream_bwd)
                        self._gexec_bwd = graph_b
                        self._graph_api_kind_bwd = "module-func"
                    except Exception:
                        raw = getattr(graph_b, "graph", None) or getattr(graph_b, "ptr", None)
                        if raw is None:
                            raise TypeError("This CuPy version exposes neither Graph.launch/upload nor graph/ptr.")
                        gexec = cp.cuda.runtime.graphInstantiate(int(raw))
                        if isinstance(gexec, tuple):
                            gexec = gexec[0]
                        self._gexec_bwd = int(gexec)
                        cp.cuda.runtime.graphUpload(self._gexec_bwd, self._cap_stream_bwd.ptr)
                        self._graph_api_kind_bwd = "runtime-pointer"
            except Exception as e:
                self._bwd_capture_error = f"{type(e).__name__}: {e}"
                # 깨끗하게 폴백 (FWD 캡처는 유지)
                self._gexec_bwd = None
                self._cap_stream_bwd = None
                self._graph_api_kind_bwd = "none"
            finally:
                self._in_bwd_capture = False

        return self

    def _warmup_gemm_handles(self) -> None:
        """캡처 전에 작은 더미 GEMM으로 cuBLAS/cuBLASLt 핸들을 초기화."""
        s = cp.cuda.Stream(non_blocking=True)
        with s:
            A = cp.empty((1, 1), dtype=cp.float32)
            B = cp.empty((1, 1), dtype=cp.float32)
            Y = cp.empty((1, 1), dtype=cp.float32)
            Z = cp.empty((1, 1), dtype=cp.float32)
            # fwd 핸들 초기화
            gemm_ops.forward_into(A, B, out=Y, with_bias=False, act="none", save_z=True, z_out=Z)
            # bwd 핸들/플랜 초기화
            gY = cp.empty((1, 1), dtype=cp.float32)
            gA = cp.empty((1, 1), dtype=cp.float32)
            gB = cp.empty((1, 1), dtype=cp.float32)
            dZ = cp.empty((1, 1), dtype=cp.float32)  # 선택적 워크스페이스
            try:
                gemm_ops.backward_into(A, B, gY, Z, with_bias=False, gA_out=gA, gB_out=gB, work_dZ=dZ)
            except TypeError:
                gemm_ops.backward_into(A, B, gY, Z, with_bias=False, gA_out=gA, gB_out=gB)
        s.synchronize()

    # ======================== 실행(FWD) ========================
    def run(self, x: cp.ndarray) -> cp.ndarray:
        if self._gexec is not None:
            inp = self.arena.view(self.ir.tensors[self.ir.input_ids[0]])
            inp[...] = x

            if self._use_runtime_graph_api:
                cp.cuda.runtime.graphLaunch(self._gexec, self._cap_stream.ptr)
            else:
                if self._graph_api_kind == "graph-object":
                    self._gexec.launch(self._cap_stream)
                else:
                    cp.cuda.graph.launch(self._gexec, self._cap_stream)

            self._cap_stream.synchronize()
            return self.arena.view(self.ir.tensors[self.ir.output_ids[0]]).copy()
        else:
            return self._run_impl(x)

    def _run_impl(self, x: Optional[cp.ndarray]) -> cp.ndarray:
        if x is not None:
            inp = self.arena.view(self.ir.tensors[self.ir.input_ids[0]])
            inp[...] = x

        for node in self.ir.nodes:
            kind = node.kind

            if kind == "Conv2D":
                X = self.arena.view(self.ir.tensors[node.inputs[0]])
                Y = self.arena.view(self.ir.tensors[node.outputs[0]])
                layer: Conv2D = node.layer_ref
                attrs = node.attrs

                conv_ops.forward_into(
                    X, layer.W,
                    out=Y,
                    B=layer.b if layer.use_bias else None,
                    stride=attrs["stride"], padding=attrs["padding"],
                    dilation=attrs["dilation"], groups=attrs["groups"],
                    with_bias=attrs["with_bias"], act="none",
                    save_z=True, Z_saved=Y,  # act='none' → Z=Y alias
                    work=node.ws_fwd,
                )

            elif kind == "ReLU":
                X = self.arena.view(self.ir.tensors[node.inputs[0]])
                Y = self.arena.view(self.ir.tensors[node.outputs[0]])
                # ReLU forward: no extra mask; backward uses (Y > 0) mask which is equivalent to (X > 0)
                cp.maximum(X, 0, out=Y)

            elif kind == "BN2D":
                X = self.arena.view(self.ir.tensors[node.inputs[0]])
                Y = self.arena.view(self.ir.tensors[node.outputs[0]])
                eps = float(node.attrs.get("eps", 1e-5))
                mu  = X.mean(axis=(0,2,3), keepdims=True)
                var = X.var(axis=(0,2,3), keepdims=True)
                inv = cp.reciprocal(cp.sqrt(var + eps))
                Y[...] = (X - mu) * inv

            elif kind == "Flatten":
                pass  # alias reshape

            elif kind == "GEMM":
                A = self.arena.view(self.ir.tensors[node.inputs[0]])
                Y = self.arena.view(self.ir.tensors[node.outputs[0]])
                Z = self.arena.view(self.ir.tensors[node.attrs["z_tid"]])
                layer = node.layer_ref
                act = node.attrs.get("act", "none")
                with_bias = node.attrs.get("with_bias", True)
                W = getattr(layer, "W")
                b = getattr(layer, "b", None) if with_bias else None

                gemm_ops.forward_into(
                    A, W,
                    out=Y,
                    bias=b,
                    act=str(act),
                    with_bias=bool(with_bias),
                    save_z=True, z_out=Z,
                )
            else:
                raise RuntimeError(f"Unknown node kind: {kind}")

        return self.arena.view(self.ir.tensors[self.ir.output_ids[0]])

    # ======================== 역전파(BWD) ========================
    def backward(self, g_out: cp.ndarray) -> cp.ndarray:
        assert self.ir is not None and self.arena is not None, "compile 먼저 호출 필요"

        if self._gexec_bwd is not None:
            # g_out을 사전할당된 grad 버퍼(out_tid)에 써넣고 런치
            out_tid = self.ir.output_ids[0]
            self._grad_view(out_tid)[...] = g_out

            if self._graph_api_kind_bwd == "runtime-pointer":
                cp.cuda.runtime.graphLaunch(self._gexec_bwd, self._cap_stream_bwd.ptr)
            else:
                if self._graph_api_kind_bwd == "graph-object":
                    self._gexec_bwd.launch(self._cap_stream_bwd)
                else:
                    cp.cuda.graph.launch(self._gexec_bwd, self._cap_stream_bwd)
            self._cap_stream_bwd.synchronize()

            # 캡처 중 미뤄둔 bias(gBias) 반영: 각 GEMM 노드의 gBias_row → layer.db
            try:
                for n in self.ir.nodes:
                    if n.kind == "GEMM" and n.attrs.get("with_bias", True):
                        gb = n.attrs.get("gBias_row", None)
                        if gb is not None:
                            n.layer_ref.db[...] = gb.reshape(-1)
            except Exception:
                pass

            return self._grad_view(self.ir.input_ids[0]).copy()

        # 비캡쳐 경로
        return self._backward_impl(g_out)

    def _backward_impl(self, g_out: Optional[cp.ndarray]) -> cp.ndarray:
        out_tid = self.ir.output_ids[0]
        if g_out is not None:
            self._grad_view(out_tid)[...] = g_out  # seed

        # 역순으로 전파 (+ 어떤 노드에서 깨졌는지 기록)
        for idx, node in reversed(list(enumerate(self.ir.nodes))):
            self._last_bwd_node = f"#{idx}:{node.kind}"
            cur_stream = cp.cuda.get_current_stream().ptr  # ★ 현재 스트림 포인터 고정 전달

            kind = node.kind

            if kind == "GEMM":
                A = self.arena.view(self.ir.tensors[node.inputs[0]])
                Z = self.arena.view(self.ir.tensors[node.attrs["z_tid"]])
                gY = self._grad_view(node.outputs[0])

                layer = node.layer_ref
                act = node.attrs.get("act", "none")
                with_bias = bool(node.attrs.get("with_bias", True))
                W = getattr(layer, "W")

                gA_out = self._grad_view(node.inputs[0])
                gB_out = layer.dW
                gBias_out = node.attrs.get("gBias_row", None) if with_bias else None

                # GEMM BWD 워크스페이스
                ws_dZ = node.attrs.get("gemm_ws_dZ")
                lt_ws = node.attrs.get("gemm_lt_ws")

                gemm_ops.backward_into(
                    A, W, gY, Z,
                    act=str(act), with_bias=with_bias,
                    gA_out=gA_out, gB_out=gB_out,
                    gBias_out=gBias_out,
                    work_dZ=ws_dZ,
                    lt_workspace=lt_ws,
                    # ★ 캡처-세이프: 반드시 현재 스트림을 명시
                    stream=cur_stream,
                )
                # 캡처 중에는 파라미터로의 쓰기를 미룸
                if with_bias and gBias_out is not None and not self._in_bwd_capture:
                    layer.db[...] = gBias_out.reshape(-1)

            elif kind == "ReLU":
                in_tid = node.inputs[0]
                out_tid = node.outputs[0]
                Y = self.arena.view(self.ir.tensors[out_tid])
                gY = self._grad_view(out_tid)
                gX = self._grad_view(in_tid)

                # ★ 임시 배열 금지: 미리 할당한 마스크 사용
                mask = node.attrs.get("relu_mask", None)
                if mask is None or mask.shape != Y.shape:
                    # 안전장치(이상 케이스용). 원래는 compile에서 이미 만들어 둠
                    mask = cp.empty_like(Y, dtype=cp.bool_)
                    node.attrs["relu_mask"] = mask

                cp.greater(Y, 0, out=mask)      # in-place: no alloc
                cp.multiply(gY, mask, out=gX)   # in-place: no alloc

            elif kind == "Flatten":
                in_tid  = node.inputs[0]
                out_tid = node.outputs[0]
                gY = self._grad_view(out_tid)
                in_shape = tuple(int(s) for s in self.ir.tensors[in_tid].shape)
                self._grad_view(in_tid)[...] = gY.reshape(in_shape)

            elif kind == "BN2D":
                # NOTE: 현 구현은 캡쳐-세이프가 아님 (mean/var 임시 생성).
                # 테스트 모델엔 BN 없음. 필요시 별도 WS 설계 필요.
                in_tid  = node.inputs[0]
                out_tid = node.outputs[0]
                X  = self.arena.view(self.ir.tensors[in_tid])
                gY = self._grad_view(out_tid)
                eps = float(node.attrs.get("eps", 1e-5))
                mu  = X.mean(axis=(0,2,3), keepdims=True)
                var = X.var(axis=(0,2,3), keepdims=True)
                inv = cp.reciprocal(cp.sqrt(var + eps))
                Xhat = (X - mu) * inv
                Naxis = X.shape[0] * X.shape[2] * X.shape[3]
                gX = (1.0/Naxis) * inv * (
                    Naxis*gY - gY.sum(axis=(0,2,3), keepdims=True)
                    - Xhat * (gY*Xhat).sum(axis=(0,2,3), keepdims=True)
                )
                self._grad_view(in_tid)[...] = gX

            elif kind == "Conv2D":
                in_tid  = node.inputs[0]
                out_tid = node.outputs[0]
                X  = self.arena.view(self.ir.tensors[in_tid])
                Y  = self.arena.view(self.ir.tensors[out_tid])  # act='none' → Z==Y
                gY = self._grad_view(out_tid)
                layer: Conv2D = node.layer_ref
                attrs = node.attrs

                gX_out = self._grad_view(in_tid)
                gW_out = layer.dW
                gB_out = layer.db if attrs["with_bias"] else None

                conv_ops.backward_into(
                    X, layer.W, gY, Y,
                    stride=attrs["stride"], padding=attrs["padding"],
                    dilation=attrs["dilation"], groups=attrs["groups"],
                    with_bias=attrs["with_bias"], act=attrs.get("act", "none"),
                    gX_out=gX_out, gW_out=gW_out, gB_out=gB_out,
                    work=node.ws_bwd,
                    # ★ 캡처-세이프: 반드시 현재 스트림을 명시(바인딩이 지원해야 함)
                    stream=cur_stream,
                )
            else:
                raise RuntimeError(f"Unknown node kind in backward: {kind}")

        return self._grad_view(self.ir.input_ids[0])
