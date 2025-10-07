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
    # 그래프에서 바로 쓸 파라미터 핸들(레이어 참조)
    layer_ref: Any = None
    # conv fast path용 WS (forward)
    ws_fwd: Optional[Dict[str, cp.ndarray]] = None
    # conv fast path용 WS (backward; v0 forward-only라 미사용)
    ws_bwd: Optional[Dict[str, cp.ndarray]] = None

@dataclass
class GraphIR:
    tensors: List[TensorDesc] = field(default_factory=list)
    nodes: List[Node] = field(default_factory=list)
    input_ids: List[int] = field(default_factory=list)
    output_ids: List[int] = field(default_factory=list)


# small alias (빠른 붙여넣기용 하위호환)
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
        # 순수 파이썬으로 numel 계산 (캡처 중 H2D 방지)
        numel = int(math.prod(int(s) for s in desc.shape))
        end = start + numel
        # 전부 디바이스 메모리 상의 view/reshape라서 캡처에 안전
        return buf[start:end].view(dtype=desc.dtype).reshape(tuple(int(s) for s in desc.shape))

def _numel(shape: Tuple[int, ...]) -> int:
    # 순수 파이썬 계산 (CUDA 그래프 캡처 중 H2D 발생 방지)
    return int(math.prod(int(s) for s in shape))


def plan_memory(ir: GraphIR) -> MemoryArena:
    # 간단 플래너: 텐서마다 고유 buffer_id라고 가정 → buffer별 필요 최대량 예약
    arena = MemoryArena()
    per_buf_need: Dict[int, int] = {}
    for t in ir.tensors:
        need = t.offset + _numel(t.shape)
        per_buf_need[t.buffer_id] = max(per_buf_need.get(t.buffer_id, 0), need)
    for bid, need in per_buf_need.items():
        # (단순화) float32 아레나. 필요시 t.dtype별 풀로 확장 가능.
        arena.allocate(bid, need, dtype=cp.float32)
    return arena


# ======================== Compiler ========================

class GraphCompiler:
    """
    model: Sequential-like (model.layers 리스트 보유)
    """
    def __init__(self, model) -> None:
        self._use_runtime_graph_api: bool = True  # 기본값: runtime API 경로 사용
        self.model = model
        self.ir: Optional[GraphIR] = None
        self.arena: Optional[MemoryArena] = None
        # CUDA Graph 관련 핸들
        self._gexec: Optional[Any] = None          # GraphExec 또는 graphExec_t(int)
        self._cap_stream: Optional[cp.cuda.Stream] = None

    def compile(self, input_shape: Tuple[int, ...], use_cuda_graph: bool = False):
        """
        1) 레이어 시퀀스를 IR로 변환
        2) 메모리 플래닝 (arena)
        3) (선택) CUDA Graph 캡처
        """
        # ----- 1. IR 작성 -----
        ir = GraphIR()
        tid = 0
        # 입력 텐서 desc
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

                # groups==1 → fast path 워크스페이스 미리 생성
                if layer.groups == 1:
                    HWo = H_out * W_out
                    K = Cin * KH * KW
                    ws_fwd = {
                        "dCol":   cp.empty((HWo, K),    dtype=cp.float32),
                        "W_KC":   cp.empty((K,   Cout), dtype=cp.float32),
                        "Y_tmp":  cp.empty((HWo, Cout), dtype=cp.float32),
                        "Z_rows": cp.empty((HWo, Cout), dtype=cp.float32),  # act=none이어도 내부 로직용
                    }
                    node.ws_fwd = ws_fwd
                else:
                    node.ws_fwd = None  # fallback 경로

                ir.nodes.append(node)
                cur_tid = y_tid
                cur_shape = y_shape
                next_buf_id += 1

            elif lname in ("ReLU",):
                # in-place 가능하나, 명확성을 위해 동일 buffer alias desc 추가
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
                ir.nodes.append(Node(kind="GEMM", inputs=[cur_tid], outputs=[y_tid],
                                     attrs=dict(act=getattr(layer, "activation", "none"),
                                                with_bias=bool(getattr(layer, "use_bias", True))),
                                     layer_ref=layer))
                cur_tid = y_tid
                cur_shape = y_shape
                next_buf_id += 1

            else:
                raise NotImplementedError(f"Unsupported layer in graph compile: {lname}")

        ir.output_ids = [cur_tid]
        self.ir = ir

        # ----- 2. 메모리 플래닝 -----
        self.arena = plan_memory(ir)

        # ----- 3. CUDA Graph(선택) -----
        if use_cuda_graph:
            # 3-1) 워밍업 1회 (플랜/워크스페이스 고정)
            warm_x = cp.empty(input_shape, dtype=cp.float32)
            _ = self._run_impl(warm_x)
            cp.cuda.get_current_stream().synchronize()

            # 3-2) 캡처
            self._cap_stream = cp.cuda.Stream(non_blocking=True)
            with self._cap_stream:
                self._cap_stream.begin_capture()
                _ = self._run_impl(None)
                graph = self._cap_stream.end_capture()  # cp.cuda.graph.Graph (버전에 따라 다름)

            # 3-3) instantiate & upload (객체 메서드 → 모듈함수 → 런타임 포인터 폴백)
            # 기본은 high-level 경로로 가정
            self._use_runtime_graph_api = False
            self._graph_api_kind = "unknown"

            try:
                # A) 가장 쉬운 경로: Graph 객체가 launch/upload 메서드를 직접 제공
                #    (여기서는 따로 instantiate 필요 없음)
                if hasattr(graph, "upload") and hasattr(graph, "launch"):
                    graph.upload(self._cap_stream)
                    self._gexec = graph            # 실행 시 graph.launch(stream) 호출
                    self._graph_api_kind = "graph-object"
                else:
                    raise AttributeError("no-graph-object-methods")
            except AttributeError:
                # B) 모듈 함수 스타일 (일부 버전)
                try:
                    # 일부 구버전에는 cupy.cuda.graph.upload/launch 만 있고
                    # instantiate는 아예 없을 수 있음. 그래프 객체 그대로 전달.
                    cp.cuda.graph.upload(graph, self._cap_stream)
                    self._gexec = graph            # 실행 시 cp.cuda.graph.launch(graph, stream)
                    self._graph_api_kind = "module-func"
                except Exception:
                    # C) 최후: 런타임 포인터 API
                    #   - 그래프 포인터는 graph.graph (cudaGraph_t) 또는 graph.ptr 중 하나로 노출됨
                    raw = getattr(graph, "graph", None)
                    if raw is None:
                        raw = getattr(graph, "ptr", None)
                    if raw is None:
                        # 그래도 없으면 더 진행 불가 → 명시 에러
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
                # C 경로: 포인터 기반 런타임 API
                cp.cuda.runtime.graphLaunch(self._gexec, self._cap_stream.ptr)
            else:
                # A/B 경로: high-level
                if self._graph_api_kind == "graph-object":
                    # Graph 객체 자체에 launch 메서드
                    self._gexec.launch(self._cap_stream)
                else:
                    # 모듈 함수 스타일
                    cp.cuda.graph.launch(self._gexec, self._cap_stream)

            self._cap_stream.synchronize()
            return self.arena.view(self.ir.tensors[self.ir.output_ids[0]]).copy()
        else:
            return self._run_impl(x)

    def _run_impl(self, x: Optional[cp.ndarray]) -> cp.ndarray:
        # 1) 입력 주입 (CUDA Graph 캡처 중일 때는 None로 들어옴)
        if x is not None:
            inp = self.arena.view(self.ir.tensors[self.ir.input_ids[0]])
            inp[...] = x

        # 2) 노드 실행
        for node in self.ir.nodes:
            kind = node.kind

            if kind == "Conv2D":
                X = self.arena.view(self.ir.tensors[node.inputs[0]])
                Y = self.arena.view(self.ir.tensors[node.outputs[0]])
                layer: Conv2D = node.layer_ref
                attrs = node.attrs

                # 현재 스트림 포인터 (캡처/일반 모두 동일 API)
                stream_ptr = cp.cuda.get_current_stream().ptr

                # fast path: groups==1 → 저수준 런처 + WS 주입 + Z==Y alias
                if layer.groups == 1:
                    a = _gconv.Conv2DAttrs()
                    a.stride_h, a.stride_w = attrs["stride"]
                    a.pad_h, a.pad_w       = attrs["padding"]
                    a.dil_h, a.dil_w       = attrs["dilation"]
                    a.groups               = 1
                    a.with_bias            = attrs["with_bias"]
                    a.act                  = getattr(_gconv.ActKind, "None")
                    a.leaky_slope          = 0.01
                    a.save_z               = True

                    ws = node.ws_fwd
                    assert ws is not None, "Conv2D ws_fwd missing in fast path"

                    _gconv.forward(
                        int(X.data.ptr), list(map(int, X.shape)),
                        int(layer.W.data.ptr), list(map(int, layer.W.shape)),
                        int(Y.data.ptr), list(map(int, Y.shape)),
                        int(layer.b.data.ptr) if (layer.use_bias and getattr(layer, "b", None) is not None) else None,
                        int(Y.data.ptr),  # Z alias to Y
                        a,
                        int(stream_ptr),  # ★ 현재 스트림 사용 (0 금지: 캡처 스트림과 분리 위험)
                        int(ws["dCol"].data.ptr),
                        int(ws["W_KC"].data.ptr),
                        int(ws["Y_tmp"].data.ptr),
                        int(ws["Z_rows"].data.ptr),
                    )

                    print(a,"이게 뭔데")
                else:
                    # fallback: 그룹 conv은 고수준 헬퍼 사용
                    y = conv_ops.forward(
                        X, layer.W, layer.b if layer.use_bias else None,
                        stride=layer.stride, padding=layer.padding,
                        dilation=layer.dilation, groups=layer.groups,
                        with_bias=layer.use_bias, act="none",
                        save_z=False, out=Y
                    )
                    if y is not Y:
                        Y[...] = y  # 안전복사

            elif kind == "ReLU":
                X = self.arena.view(self.ir.tensors[node.inputs[0]])
                Y = self.arena.view(self.ir.tensors[node.outputs[0]])
                cp.maximum(X, 0, out=Y)

            elif kind == "BN2D":
                # 간단 train 모드 BN (gamma=1, beta=0 가정)
                X = self.arena.view(self.ir.tensors[node.inputs[0]])
                Y = self.arena.view(self.ir.tensors[node.outputs[0]])
                eps = float(node.attrs.get("eps", 1e-5))
                mu  = X.mean(axis=(0,2,3), keepdims=True)
                var = X.var(axis=(0,2,3), keepdims=True)
                inv = cp.reciprocal(cp.sqrt(var + eps))
                Y[...] = (X - mu) * inv

            elif kind == "Flatten":
                # view만 변경했으므로 아무 것도 안 함
                pass

            elif kind == "GEMM":
                A = self.arena.view(self.ir.tensors[node.inputs[0]])
                Y = self.arena.view(self.ir.tensors[node.outputs[0]])
                layer = node.layer_ref  # Dense or Linear
                act = node.attrs.get("act", "none")
                with_bias = node.attrs.get("with_bias", True)
                W = getattr(layer, "W")
                b = getattr(layer, "b", None) if with_bias else None
                y = gemm_ops.forward(
                    A, W, b,
                    act=str(act), with_bias=bool(with_bias),
                    save_z=False, out=Y
                )
                if y is not Y:
                    Y[...] = y
            else:
                raise RuntimeError(f"Unknown node kind: {kind}")

        # 3) 출력 (view 반환; 필요시 호출측에서 copy)
        return self.arena.view(self.ir.tensors[self.ir.output_ids[0]])
