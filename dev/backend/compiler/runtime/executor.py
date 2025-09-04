# -*- coding: utf-8 -*-
"""
executor.py (graph_executor_v2 파이프라인 연동)

역할
- IR -> PassManager (canonicalize, fuse_elementwise)
- 커널 선택(pick_kernel)
- 네이티브 모듈로 launch
- GEMM_BIAS_ACT 파라미터 블록(ctypes) 생성 및 bufs 마지막에 Host 포인터 추가

버퍼 규약(ABI: ge_v2_api.h)
- buffers: [inputs ...] + [outputs ...] + [옵션: Host param block address]
- 각 input/output은 'device pointer' 정수값(ge2_uintptr; uintptr_t 호환)
"""
from __future__ import annotations
from typing import Dict, Tuple, List, Any
import importlib, os, sys
import ctypes as C

from ..ir.nodes import Graph, Tensor, Op
from ..passes.pass_manager import PassManager
from ..passes.canonicalize import canonicalize
from ..passes.fuse_elementwise import fuse_elementwise
from ..kernels.selector import pick_kernel

# ------------------------------ 네이티브 모듈 로딩 ------------------------------
PREFERRED = os.environ.get("GE_NATIVE", "graph_executor_v2")
CANDIDATES = (
    PREFERRED,
    "graph_executor_v2",
    "graph_executor",
    "backend.graph_executor_v2",
    "backend.graph_executor_v2.test.graph_executor_v2",  # ★ 추가
)


_native = None
_chosen = None
_last_err: Exception | None = None


def _import_first(modnames):
    last = None
    for name in modnames:
        try:
            m = importlib.import_module(name)
            # 호출부에서 'graph_executor'로도 import 가능하도록 alias
            sys.modules.setdefault("graph_executor", m)
            return name, m, None
        except Exception as e:
            last = e
    return None, None, last


_chosen, _native, _last_err = _import_first(CANDIDATES)
_HAS_LAUNCH = hasattr(_native, "launch_kernel") if _native else False

# ------------------------------ 텐서 포인터 추출 ------------------------------
def _try_ptr_from_torch(x: Any) -> int | None:
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return int(x.data_ptr())
    except Exception:
        pass
    return None


def _try_ptr_from_cupy(x: Any) -> int | None:
    try:
        import cupy as cp
        if isinstance(x, cp.ndarray):
            return int(x.data.ptr)
    except Exception:
        pass
    return None


def _device_ptr(x: Any) -> int:
    """
    Torch/CuPy/커스텀 텐서에서 device pointer를 정수로 추출.
    실패 시 TypeError.
    """
    # 일반적으로 Tensor wrapper가 .t 같은 실제 텐서를 갖고 있을 수 있음
    obj = getattr(x, "t", x)

    # case 1: 이미 정수 주소
    if isinstance(obj, int):
        return obj

    # case 2: 프레임워크 텐서
    p = _try_ptr_from_torch(obj)
    if p is not None:
        return p
    p = _try_ptr_from_cupy(obj)
    if p is not None:
        return p

    # case 3: 커스텀 속성
    for attr in ("ptr", "device_ptr"):
        if hasattr(obj, attr):
            return int(getattr(obj, attr))

    raise TypeError(f"Unsupported tensor type for device pointer: {type(obj)}")


# ------------------------------ GEMM 파라미터 블록 ------------------------------
class GemmBiasActParams(C.Structure):
    """
    네이티브 쪽(C++/CUDA)과 ABI가 맞아야 하므로 필드 순서/타입 고정.
    """
    _fields_ = [
        ("M", C.c_int),        # A: MxK, B: KxN, D: MxN
        ("N", C.c_int),
        ("K", C.c_int),
        ("has_bias", C.c_int), # 0/1
        ("act", C.c_int),      # 0:none, 1:ReLU (확장 여지: 2:GELU 등)
    ]


# ------------------------------ ExecutorV2 ------------------------------
class ExecutorV2:
    def __init__(
        self,
        device_caps: Dict[str, bool] | None = None,
        stream: int | None = 0,
        dry_run: bool = True,
        native_module=None,
    ):
        """
        device_caps: 커널 선택 힌트. e.g., {"tensor_core": True}
        stream: 정수형 cudaStream 주소(0이면 default stream), None이면 0 취급
        dry_run: True면 launch 호출 대신 print만 수행
        native_module: 주입형 네이티브 모듈(테스트용)
        """
        self.device_caps = device_caps or {"tensor_core": True}
        self.stream = stream
        self.dry_run = dry_run
        self.native = native_module or _native
        self.has_launch = hasattr(self.native, "launch_kernel") if self.native else False
        # run() 동안 C 구조체 메모리 해제 방지(수명관리)
        self._live_param_blocks: List[object] = []

    # -------------------------- 메인 엔트리 --------------------------
    def run(self, graph: Graph):
        pm = PassManager([canonicalize, fuse_elementwise])
        plan = pm.run(graph)

        for op in plan.ops:
            # 패스에서 미합쳐진 잔여 elementwise는 여기선 실행하지 않음(디버그 로그만)
            if op.op_type in ("MATMUL", "BIAS_ADD", "RELU", "GELU"):
                print(f"[SKIP] unfused op: {op.op_type}")
                continue

            kname = pick_kernel(op, self.device_caps)  # ex) "gemm_bias_act_tc_f16"
            bufs, descs = self._prepare_buffers(op, kname)

            if self.dry_run:
                print(f"[DRY] launch {kname} with {len(bufs)} buffers attrs={op.attrs}")
                continue

            if not self.has_launch or not self.native:
                raise RuntimeError("native.launch_kernel가 없습니다.")

            stream_i = 0 if self.stream is None else int(self.stream)
            # descs는 선택 정보(디버그/로깅용). 네이티브는 buffers만 사용해도 됨.
            self.native.launch_kernel(kname, bufs, descs, stream_i)

        # 파라미터 블록 생명주기: run() 종료 시 정리
        self._live_param_blocks.clear()

    # -------------------------- 버퍼/파라미터 구성 --------------------------
    def _prepare_buffers(self, op: Op, kname: str) -> Tuple[List[int], Dict]:
        """
        - inputs → outputs → [Host param block ptr] 순으로 bufs 구성
        - descs: 디버그용 메타 정보(네이티브와 ABI 영향 X)
        """
        ins: List[Tensor] = getattr(op, "inputs", [])
        outs: List[Tensor] = getattr(op, "outputs", [])

        # 1) device pointer 수집
        bufs: List[int] = [_device_ptr(t) for t in ins] + [_device_ptr(t) for t in outs]

        # 2) descs(옵션) 구성
        def _shape_of(t): 
            s = getattr(t, "shape", None)
            return list(s) if s is not None else []

        def _dtype_of(t):
            dt = getattr(t, "dtype", None)
            return str(dt) if dt is not None else "unknown"

        def _to_desc(t: Tensor) -> Dict:
            return {
                "shape": _shape_of(t),
                "dtype": _dtype_of(t),
                "layout": getattr(t, "layout", "rowmajor"),
                "device": getattr(t, "device", "cuda"),
            }

        descs = {"buffers": [_to_desc(t) for t in (ins + outs)], "kernel": kname}

        # 3) 커널별 파라미터 블록 추가
        if op.op_type == "GEMM_BIAS_ACT":
            # 우선 attrs에서 시도
            M, N, K = op.attrs.get("mnk", (0, 0, 0))
            # shape에서 보정
            if not (M and N and K):
                if len(ins) >= 2:
                    A, B = ins[0], ins[1]
                    ash, bsh = _shape_of(A), _shape_of(B)
                    if len(ash) == 2 and len(bsh) == 2:
                        M, K = ash
                        K2, N = bsh
                        if K2 != K:
                            raise ValueError(f"K mismatch: {K2} vs {K}")
                if len(outs) >= 1 and not (M and N):
                    csh = _shape_of(outs[0])
                    if len(csh) == 2:
                        M, N = csh

            has_bias = 1 if (len(ins) >= 3) else 0
            act_name = op.attrs.get("act", "relu")
            act = 0 if act_name in (None, "none") else 1  # 1=ReLU (필요시 확장)

            p = GemmBiasActParams(M=M, N=N, K=K, has_bias=has_bias, act=act)
            self._live_param_blocks.append(p)         # 수명 유지
            bufs.append(C.addressof(p))               # Host 파라미터 포인터를 bufs 마지막에

        return bufs, descs
