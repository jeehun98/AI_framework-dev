# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, Tuple, List, Any
import importlib, os, sys
import ctypes as C

from ..ir.nodes import Graph, Tensor, Op
from ..passes.pass_manager import PassManager
from ..passes.canonicalize import canonicalize
from ..passes.fuse_elementwise import fuse_elementwise
from ..kernels.selector import pick_kernel

PREFERRED = os.environ.get("GE_NATIVE", "graph_executor_v2")
CANDIDATES = (PREFERRED, "graph_executor_v2", "graph_executor",
              "backend.graph_executor_v2", "backend.graph_executor_v2.graph_executor_v2")

_native = None; _chosen = None; _last_err: Exception | None = None
def _import_first(modnames):
    last = None
    for name in modnames:
        try:
            m = importlib.import_module(name)
            sys.modules.setdefault("graph_executor", m)
            return name, m, None
        except Exception as e:
            last = e
    return None, None, last

_chosen, _native, _last_err = _import_first(CANDIDATES)
_HAS_LAUNCH = hasattr(_native, "launch_kernel") if _native else False

# --- 텐서 포인터 어댑터 (Torch/CuPy 자동 인식) ---
def _try_ptr_from_torch(x: Any) -> int | None:
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return int(x.data_ptr())
    except Exception: pass
    return None

def _try_ptr_from_cupy(x: Any) -> int | None:
    try:
        import cupy as cp
        if isinstance(x, cp.ndarray):
            return int(x.data.ptr)
    except Exception: pass
    return None

def _device_ptr(x: Any) -> int:
    p = _try_ptr_from_torch(x)
    if p is not None: return p
    p = _try_ptr_from_cupy(x)
    if p is not None: return p
    if hasattr(x, "ptr"): return int(getattr(x, "ptr"))
    if hasattr(x, "device_ptr"): return int(getattr(x, "device_ptr"))
    raise TypeError(f"Unsupported tensor type for device pointer: {type(x)}")

# --- GEMM 파라미터-블록(ctypes, Host 메모리) ---
class GemmBiasActParams(C.Structure):
    _fields_ = [
        ("M", C.c_int),
        ("N", C.c_int),
        ("K", C.c_int),
        ("has_bias", C.c_int),
        ("act", C.c_int),  # 0:none, 1:ReLU
    ]

class ExecutorV2:
    def __init__(self, device_caps: Dict[str, bool] | None = None,
                 stream: int | None = 0, dry_run: bool = True, native_module=None):
        self.device_caps = device_caps or {"tensor_core": True}
        self.stream = stream
        self.dry_run = dry_run
        self.native = native_module or _native
        self.has_launch = hasattr(self.native, "launch_kernel") if self.native else False
        self._live_param_blocks: List[object] = []  # launch 동안 참조 유지

    def run(self, graph: Graph):
        pm = PassManager([canonicalize, fuse_elementwise])
        plan = pm.run(graph)

        for op in plan.ops:
            if op.op_type in ("MATMUL", "BIAS_ADD", "RELU", "GELU"):
                print(f"[SKIP] unfused op: {op.op_type}")
                continue

            kname = pick_kernel(op, self.device_caps)
            bufs, descs = self._prepare_buffers(op, kname)

            if self.dry_run:
                print(f"[DRY] launch {kname} with {len(bufs)} buffers attrs={op.attrs}")
                continue

            if not self.has_launch or not self.native:
                raise RuntimeError("native.launch_kernel가 없습니다.")

            stream_i = 0 if self.stream is None else int(self.stream)
            self.native.launch_kernel(kname, bufs, descs, stream_i)

        # 파라미터 블록 생명주기: run() 끝나면 정리
        self._live_param_blocks.clear()

    def _prepare_buffers(self, op: Op, kname: str) -> Tuple[List[int], Dict]:
        ins  = getattr(op, "inputs",  [])
        outs = getattr(op, "outputs", [])

        # 1) 기본 bufs: 입력 → 출력 (device pointer)
        bufs: List[int] = [ _device_ptr(getattr(t, "t", t)) for t in ins ] \
                        + [ _device_ptr(getattr(t, "t", t)) for t in outs ]

        # 2) descs (현재는 참고용)
        def _shape_of(t): s = getattr(t, "shape", None); return list(s) if s is not None else []
        def _dtype_of(t): dt = getattr(t, "dtype", None); return str(dt) if dt is not None else "unknown"
        def _to_desc(t: Tensor) -> Dict:
            return {"shape": _shape_of(t), "dtype": _dtype_of(t),
                    "layout": getattr(t, "layout", "rowmajor"),
                    "device": getattr(t, "device", "cuda")}
        descs = {"buffers": [_to_desc(t) for t in (ins + outs)]}

        # 3) 커널별 파라미터-블록 생성 (Host, 마지막에 추가)
        if op.op_type == "GEMM_BIAS_ACT":
            # MNK는 attrs나 텐서 shape에서 추출
            M, N, K = op.attrs.get("mnk", (0, 0, 0))
            if not (M and N and K):
                # 가능한 경우: A[M,K], B[K,N], C[M,N]
                if len(ins) >= 2:
                    A = ins[0]; B = ins[1]
                    ash = _shape_of(A); bsh = _shape_of(B)
                    if len(ash) == 2 and len(bsh) == 2:
                        M, K = ash
                        K2, N = bsh
                        if K2 != K: raise ValueError(f"K mismatch: {K2} vs {K}")
                if len(outs) >= 1 and not (M and N):
                    csh = _shape_of(outs[0])
                    if len(csh) == 2: M, N = csh

            has_bias = 1 if (len(ins) >= 3) else 0
            act_name = op.attrs.get("act", "relu")
            act = 0 if act_name in (None, "none") else 1  # ReLU=1 기본

            p = GemmBiasActParams(M=M, N=N, K=K, has_bias=has_bias, act=act)
            self._live_param_blocks.append(p)  # 수명 유지
            bufs.append(C.addressof(p))        # 마지막에 Host 파라미터 포인터 추가

        return bufs, descs
