from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
import cupy as cp

# 레이어/ops 가져오기 (당신 프로젝트 구조에 맞춤)
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
    tensors: List[TesnorDesc] = field(default_factory=list)  # typo guard below
    nodes: List[Node] = field(default_factory=list)
    input_ids: List[int] = field(default_factory=list)
    output_ids: List[int] = field(default_factory=list)

# small typo correction (keep compatibility if you paste quickly)
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
        start = desc.offset
        end = start + int(cp.prod(cp.array(desc.shape)))
        return buf[start:end].view(dtype=desc.dtype).reshape(desc.shape)


def _numel(shape: Tuple[int, ...]) -> int:
    return int(cp.prod(cp.array(shape)))


def plan_memory(ir: GraphIR) -> MemoryArena:
    # 간단 플래너: 텐서마다 고유 buffer_id라고 가정 → buffer별 필요 최대량 예약
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
    """
    model: Sequential-like (model.layers 리스트 보유)
    """
    def __init__(self, model) -> None:
        self.model = model
        self.ir: Optional[GraphIR] = None
        self.arena: Optional[MemoryArena] = None
        self._cuda_graph_exec = None

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
                        "Z_rows": cp.empty((HWo, Cout), dtype=cp.float32),  # act=none이어도 내부 로직이 쓰므로 확보
                    }
                    node.ws_fwd = ws_fwd
                else:
                    node.ws_fwd = None  # fallback 경로에서 내부 helper 사용

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
            g = cp.cuda.graph.Graph()
            with g.capture():
                _ = self._run_impl(None)  # warm-up inside capture
            self._cuda_graph_exec = g.instantiate()
        return self

    # ======================== 실행 ========================

    def run(self, x: cp.ndarray) -> cp.ndarray:
        if self._cuda_graph_exec is not None:
            # 입력 복사 → 캡처된 실행 → 출력 view 반환
            inp = self.arena.view(self.ir.tensors[self.ir.input_ids[0]])
            inp[...] = x
            self._cuda_graph_exec.launch()
            return self.arena.view(self.ir.tensors[self.ir.output_ids[0]])
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
                # fast path: groups==1 → low-level 런처 + WS 주입 + Z==Y alias
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
                        int(layer.b.data.ptr) if (layer.use_bias and layer.b is not None) else None,
                        int(Y.data.ptr),  # Z alias to Y
                        a,
                        0,  # stream
                        int(ws["dCol"].data.ptr),
                        int(ws["W_KC"].data.ptr),
                        int(ws["Y_tmp"].data.ptr),
                        int(ws["Z_rows"].data.ptr),
                    )
                else:
                    # fallback: 그룹 conv은 고수준 헬퍼 사용(임시 내부버퍼 쓰지만 동작 OK)
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
                y = gemm_ops.forward(A, W, b, act=str(act), with_bias=bool(with_bias),
                                     save_z=False, out=Y)
                if y is not Y:
                    Y[...] = y
            else:
                raise RuntimeError(f"Unknown node kind: {kind}")

        # 3) 출력
        return self.arena.view(self.ir.tensors[self.ir.output_ids[0]])
