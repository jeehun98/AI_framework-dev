from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
import cupy as cp
import math

# 레이어/ops 가져오기 (프로젝트 구조에 맞춤)
from graph_executor_v2.layers.conv2d import Conv2D
from graph_executor_v2.layers.linear import Linear
from graph_executor_v2.layers.dense_gemm import Dense
from graph_executor_v2.ops import conv2d as conv_ops
from graph_executor_v2.ops import gemm as gemm_ops
from graph_executor_v2.ops import _ops_conv2d as _gconv


# ======================== IR ========================

@dataclass
class TensorDesc:
    shape: Tuple[int, ...]
    dtype: Any = cp.float32
    buffer_id: int = -1
    offset: int = 0  # element offset (아레나 1D 버퍼 기준)

@dataclass
class Node:
    kind: str                        # "Conv2D", "ReLU", "BN2D", "Flatten", "GEMM"
    inputs: List[int]                # tensor ids
    outputs: List[int]               # tensor ids
    attrs: Dict[str, Any] = field(default_factory=dict)
    layer_ref: Any = None
    ws_fwd: Optional[Dict[str, cp.ndarray]] = None
    ws_bwd: Optional[Dict[str, cp.ndarray]] = None

@dataclass
class GraphIR:
    tensors: List[TensorDesc] = field(default_factory=list)
    nodes: List[Node] = field(default_factory=list)
    input_ids: List[int] = field(default_factory=list)
    output_ids: List[int] = field(default_factory=list)

TesnorDesc = TensorDesc


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
        return buf[start:end].view(dtype=desc.dtype).reshape(tuple(int(s) for s in desc.shape))

def _numel(shape: Tuple[int, ...]) -> int:
    return int(math.prod(int(s) for s in shape))


def plan_memory(ir: GraphIR) -> MemoryArena:
    arena = MemoryArena()
    per_buf_need: Dict[int, int] = {}
    for t in ir.tensors:
        need = t.offset + _numel(t.shape)
        per_buf_need[t.buffer_id] = max(per_buf_need.get(t.buffer_id, 0), need)
    for bid, need in per_buf_need.items():
        arena.allocate(bid, need, dtype=cp.float32)
    return arena


# ======================== Compiler ========================

class GraphCompiler:
    def __init__(self, model) -> None:
        self._use_runtime_graph_api: bool = True
        self.model = model
        self.ir: Optional[GraphIR] = None
        self.arena: Optional[MemoryArena] = None
        self._gexec: Optional[Any] = None
        self._cap_stream: Optional[cp.cuda.Stream] = None
        self._graph_api_kind: str = "unknown"

    def compile(self, input_shape: Tuple[int, ...], use_cuda_graph: bool = False):
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
                Cout  = layer.filters
                y_shape = (N, Cout, H_out, W_out)

                # Y만 등록 (act='none'이므로 Z==Y alias로 사용)
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

                # groups==1 fast path WS
                if layer.groups == 1:
                    HWo = H_out * W_out
                    K = Cin * KH * KW
                    ws_fwd = {
                        "dCol":   cp.empty((HWo, K),    dtype=cp.float32),
                        "W_KC":   cp.empty((K,   Cout), dtype=cp.float32),
                        "Y_tmp":  cp.empty((HWo, Cout), dtype=cp.float32),
                        "Z_rows": cp.empty((HWo, Cout), dtype=cp.float32),
                    }
                    node.ws_fwd = ws_fwd
                else:
                    node.ws_fwd = None

                ir.nodes.append(node)
                cur_tid = y_tid
                cur_shape = y_shape
                next_buf_id += 1

            elif lname in ("ReLU",):
                in_desc = ir.tensors[cur_tid]
                ir.tensors.append(TensorDesc(shape=in_desc.shape, buffer_id=in_desc.buffer_id, offset=in_desc.offset))
                y_tid = tid; tid += 1
                ir.nodes.append(Node(kind="ReLU", inputs=[cur_tid], outputs=[y_tid], layer_ref=layer))
                cur_tid = y_tid

            elif lname in ("BatchNorm2D", "BatchNorm"):
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
                Nout = getattr(layer, "units", None) or getattr(layer, "out_features")
                y_shape = (N, int(Nout))

                ir.tensors.append(TensorDesc(shape=y_shape, buffer_id=next_buf_id, offset=0))
                y_tid = tid; tid += 1
                # Z는 같은 버퍼에 연속 배치
                ir.tensors.append(TensorDesc(shape=y_shape, buffer_id=next_buf_id, offset=_numel(y_shape)))
                z_tid = tid; tid += 1

                ir.nodes.append(Node(
                    kind="GEMM",
                    inputs=[cur_tid],
                    outputs=[y_tid],
                    attrs=dict(
                        act=getattr(layer, "activation", "none"),
                        with_bias=bool(getattr(layer, "use_bias", True)),
                        z_tid=z_tid
                    ),
                    layer_ref=layer
                ))
                cur_tid = y_tid
                cur_shape = y_shape
                next_buf_id += 1

            else:
                raise NotImplementedError(f"Unsupported layer in graph compile: {lname}")

        ir.output_ids = [cur_tid]
        self.ir = ir

        # 메모리 플래닝
        self.arena = plan_memory(ir)

        # (옵션) CUDA Graph 캡처
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
                    raw = getattr(graph, "graph", None)
                    if raw is None:
                        raw = getattr(graph, "ptr", None)
                    if raw is None:
                        raise TypeError("This CuPy version exposes neither Graph.launch/upload nor graph/ptr.")
                    gexec = cp.cuda.runtime.graphInstantiate(int(raw))
                    if isinstance(gexec, tuple):
                        gexec = gexec[0]
                    self._gexec = int(gexec)
                    cp.cuda.runtime.graphUpload(self._gexec, self._cap_stream.ptr)
                    self._use_runtime_graph_api = True
                    self._graph_api_kind = "runtime-pointer"

        return self

    # ======================== 실행 ========================

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
                stream_ptr = cp.cuda.get_current_stream().ptr

                if layer.groups == 1:
                    a = _gconv.Conv2DAttrs()
                    a.stride_h, a.stride_w = attrs["stride"]
                    a.pad_h, a.pad_w       = attrs["padding"]
                    a.dil_h, a.dil_w       = attrs["dilation"]
                    a.groups               = 1
                    a.with_bias            = attrs["with_bias"]
                    a.act                  = getattr(_gconv.ActKind, "None")
                    a.leaky_slope          = 0.01
                    a.save_z               = True  # Z==Y alias

                    ws = node.ws_fwd
                    assert ws is not None, "Conv2D ws_fwd missing in fast path"

                    _gconv.forward(
                        int(X.data.ptr), list(map(int, X.shape)),
                        int(layer.W.data.ptr), list(map(int, layer.W.shape)),
                        int(Y.data.ptr), list(map(int, Y.shape)),
                        int(layer.b.data.ptr) if (layer.use_bias and getattr(layer, "b", None) is not None) else None,
                        int(Y.data.ptr),  # Z alias to Y
                        a,
                        int(stream_ptr),
                        int(ws["dCol"].data.ptr),
                        int(ws["W_KC"].data.ptr),
                        int(ws["Y_tmp"].data.ptr),
                        int(ws["Z_rows"].data.ptr),
                    )
                else:
                    # fallback: 고수준 구현. Z 저장 필요 → Z_saved=Y alias
                    conv_ops.forward(
                        X, layer.W, layer.b if layer.use_bias else None,
                        stride=layer.stride, padding=layer.padding,
                        dilation=layer.dilation, groups=layer.groups,
                        with_bias=layer.use_bias, act="none",
                        save_z=True, Z_saved=Y, out=Y
                    )

            elif kind == "ReLU":
                X = self.arena.view(self.ir.tensors[node.inputs[0]])
                Y = self.arena.view(self.ir.tensors[node.outputs[0]])
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
                pass

            elif kind == "GEMM":
                A = self.arena.view(self.ir.tensors[node.inputs[0]])
                Y = self.arena.view(self.ir.tensors[node.outputs[0]])
                Z = self.arena.view(self.ir.tensors[node.attrs["z_tid"]])
                layer = node.layer_ref
                act = node.attrs.get("act", "none")
                with_bias = node.attrs.get("with_bias", True)
                W = getattr(layer, "W")
                b = getattr(layer, "b", None) if with_bias else None

                res = gemm_ops.forward(
                    A, W, b,
                    act=str(act), with_bias=bool(with_bias),
                    save_z=True, z_out=Z, out=Y
                )
                if isinstance(res, tuple):
                    y_ret, _ = res
                else:
                    y_ret = res
                if y_ret is not Y:
                    Y[...] = y_ret
            else:
                raise RuntimeError(f"Unknown node kind: {kind}")

        return self.arena.view(self.ir.tensors[self.ir.output_ids[0]])

    # ======================== 역전파 ========================

    def backward(self, g_out: cp.ndarray) -> cp.ndarray:
        assert self.ir is not None and self.arena is not None, "compile 먼저 호출 필요"

        grads: Dict[int, cp.ndarray] = {}
        out_tid = self.ir.output_ids[0]
        gY = cp.empty_like(self.arena.view(self.ir.tensors[out_tid]))
        gY[...] = g_out
        grads[out_tid] = gY

        for node in reversed(self.ir.nodes):
            kind = node.kind

            if kind == "GEMM":
                A = self.arena.view(self.ir.tensors[node.inputs[0]])
                Z = self.arena.view(self.ir.tensors[node.attrs["z_tid"]])
                gY = grads[node.outputs[0]]

                layer = node.layer_ref
                act = node.attrs.get("act", "none")
                with_bias = bool(node.attrs.get("with_bias", True))
                W = getattr(layer, "W")
                b_present = (getattr(layer, "b", None) is not None) and with_bias

                bwd = gemm_ops.backward(
                    A, W, gY, Z,
                    act=str(act),
                    with_bias=with_bias,
                    want_gA=True, want_gB=True, want_gBias=b_present,
                    warn_mismatch=False
                )

                if "gB" in bwd:
                    if hasattr(layer, "dW") and layer.dW is not None:
                        layer.dW[...] = bwd["gB"]
                    else:
                        layer.dW = bwd["gB"]

                if b_present and "gBias" in bwd:
                    gBias = bwd["gBias"]
                    if hasattr(layer, "db") and layer.db is not None:
                        layer.db[...] = gBias.reshape(-1)
                    else:
                        layer.db = gBias.reshape(-1)

                gA = bwd["gA"]
                in_tid = node.inputs[0]
                grads[in_tid] = grads[in_tid] + gA if in_tid in grads else gA

            elif kind == "ReLU":
                in_tid = node.inputs[0]
                out_tid = node.outputs[0]
                Y = self.arena.view(self.ir.tensors[out_tid])
                gY = grads[out_tid]
                gX = gY * (Y > 0)
                grads[in_tid] = grads[in_tid] + gX if in_tid in grads else gX

            elif kind == "Flatten":
                in_tid = node.inputs[0]
                out_tid = node.outputs[0]
                gY = grads[out_tid]
                in_shape = tuple(int(s) for s in self.ir.tensors[in_tid].shape)
                gX = gY.reshape(in_shape)
                grads[in_tid] = grads[in_tid] + gX if in_tid in grads else gX

            elif kind == "BN2D":
                in_tid = node.inputs[0]
                out_tid = node.outputs[0]
                X = self.arena.view(self.ir.tensors[in_tid])
                gY = grads[out_tid]
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
                grads[in_tid] = grads[in_tid] + gX if in_tid in grads else gX

            elif kind == "Conv2D":
                in_tid = node.inputs[0]
                out_tid = node.outputs[0]
                X = self.arena.view(self.ir.tensors[in_tid])
                Y = self.arena.view(self.ir.tensors[out_tid])  # act='none'이므로 Z==Y
                gY = grads[out_tid]
                layer: Conv2D = node.layer_ref
                attrs = node.attrs

                # conv2d backward 호출 (Z는 pre-activation → 여기서는 Y alias)
                bwd = conv_ops.backward(
                    X, layer.W, gY, Y,
                    stride=attrs["stride"], padding=attrs["padding"],
                    dilation=attrs["dilation"], groups=attrs["groups"],
                    with_bias=attrs["with_bias"], act=attrs.get("act", "none"),
                    want_gX=True, want_gW=True, want_gB=attrs["with_bias"]
                )

                # 파라미터 grad 반영
                if "gW" in bwd:
                    if hasattr(layer, "dW") and layer.dW is not None:
                        layer.dW[...] = bwd["gW"]
                    else:
                        layer.dW = bwd["gW"]
                if attrs["with_bias"] and "gB" in bwd and bwd["gB"] is not None:
                    if hasattr(layer, "db") and layer.db is not None:
                        layer.db[...] = bwd["gB"]
                    else:
                        layer.db = bwd["gB"]

                # 입력 grad 전파
                if "gX" in bwd and bwd["gX"] is not None:
                    gX = bwd["gX"]
                    grads[in_tid] = grads[in_tid] + gX if in_tid in grads else gX
                else:
                    raise RuntimeError("conv2d.backward did not return gX")

            else:
                raise RuntimeError(f"Unknown node kind in backward: {kind}")

        return grads[self.ir.input_ids[0]]
