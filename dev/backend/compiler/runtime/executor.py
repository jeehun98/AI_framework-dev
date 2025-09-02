# -*- coding: utf-8 -*-
"""
Python 주도 실행 파이프라인:
IR(Graph) → Pass(canonicalize, fuse_elementwise) → 커널 선택 → (옵션) 네이티브 실행

핵심 포인트:
- 네이티브 모듈을 (우선순위에 따라) import 하여 graph_executor 별칭으로 등록
- dry_run=True면 커널 선택과 플랜만 출력(실제 launch 미수행)
- _prepare_buffers: 프레임워크 텐서를 디바이스 포인터(uintptr) 리스트로 변환하는 자리
  (현재 스켈레톤: 실제 텐서 어댑터는 프로젝트 사양에 맞춰 구현)
"""

from __future__ import annotations
from typing import Dict, Tuple, List
import importlib
import os
import sys

from ..ir.nodes import Graph, Tensor, Op
from ..passes.pass_manager import PassManager
from ..passes.canonicalize import canonicalize
from ..passes.fuse_elementwise import fuse_elementwise
from ..kernels.selector import pick_kernel

# ---------------------------------------------------------------------
# 네이티브 바인딩 선택: v2 우선, 없으면 v1 폴백. 환경변수로 강제 가능.
# ---------------------------------------------------------------------
PREFERRED = os.environ.get("GE_NATIVE", "graph_executor_v2")

# 현재 .pyd 위치를 고려해 import 후보를 확장 (필요 시 경로 조정)
CANDIDATES = (
    PREFERRED,
    "graph_executor_v2",
    "graph_executor",
    "backend.graph_executor_v2",
    "backend.graph_executor_v2.graph_executor_v2",
)

_native = None
_chosen = None
_last_err: Exception | None = None


def _import_first(modnames):
    last = None
    for name in modnames:
        try:
            m = importlib.import_module(name)
            # 하위 모듈이 selector 등에서 'graph_executor' 이름으로 접근해도 되도록 별칭 등록
            sys.modules.setdefault("graph_executor", m)
            return name, m, None
        except Exception as e:
            last = e
    return None, None, last


_chosen, _native, _last_err = _import_first(CANDIDATES)

_HAS_LAUNCH = hasattr(_native, "launch_kernel") if _native else False
_HAS_QUERY = any(
    hasattr(_native, n) for n in ("query_capability", "query_kernels")
) if _native else False


def _debug_backend() -> Dict[str, object]:
    return {
        "chosen": _chosen,
        "has_launch": _HAS_LAUNCH,
        "has_query": _HAS_QUERY,
        "last_err": repr(_last_err),
        "module": repr(_native),
    }


class ExecutorV2:
    """
    - device_caps: 하드웨어 기능 힌트 (예: {"tensor_core": True})
    - stream: cudaStream_t를 정수(uintptr)로 전달 (옵션)
    - dry_run: True면 커널 선택/플랜만 출력하고 launch는 생략
    - native_module: 주입 모듈(테스트/모킹용)
    """

    def __init__(
        self,
        device_caps: Dict[str, bool] | None = None,
        stream: int | None = 0,
        dry_run: bool = True,
        native_module=None,  # 테스트/강제 주입용
    ):
        self.device_caps = device_caps or {"tensor_core": True}
        self.stream = stream
        self.dry_run = dry_run

        # 주입 모듈이 있으면 그걸 사용 (테스트 안정화용)
        self.native = native_module or _native
        self.has_launch = hasattr(self.native, "launch_kernel") if self.native else False
        self.has_query = any(
            hasattr(self.native, n) for n in ("query_capability", "query_kernels")
        ) if self.native else False

    def run(self, graph: Graph):
        # 1) 패스 파이프라인
        pm = PassManager([canonicalize, fuse_elementwise])
        plan = pm.run(graph)

        # 2) 커널 선택 + (옵션) 네이티브 실행
        for op in plan.ops:
            # 아직 fuse 안 된 단일 op들은 데모 단계라 스킵
            if op.op_type in ("MATMUL", "BIAS_ADD", "RELU", "GELU"):
                print(f"[SKIP] unfused op: {op.op_type}")
                continue

            kname = pick_kernel(op, self.device_caps)
            bufs, descs = self._prepare_buffers(op)  # TODO: 실제 텐서 → 디바이스 포인터/desc

            if self.dry_run:
                print(f"[DRY] launch {kname} with {len(bufs)} buffers attrs={op.attrs}")
                continue

            if not self.has_launch or not self.native:
                info = _debug_backend()
                raise RuntimeError(
                    "native.launch_kernel가 없습니다. 바인딩/로딩 확인 필요. "
                    f"backend_info={info}"
                )

            # 실제 런치
            stream_i = 0 if self.stream is None else int(self.stream)
            self.native.launch_kernel(kname, bufs, descs, stream_i)

    # === 프레임워크 텐서 → 디바이스 포인터/디스크립터 변환 자리 =================
    def _prepare_buffers(self, op: Op) -> Tuple[List[int], Dict]:
        """
        지금은 스켈레톤: 디바이스 포인터/디스크립터를 준비해 native에 넘긴다.

        규약(헤더와 일치):
        - bufs: [in0_ptr, in1_ptr, ..., out0_ptr, ...] (각각 uintptr_t로 변환된 device ptr)
        - descs: {'buffers': [{'shape':[...], 'dtype':'f16', 'layout':'rowmajor', 'device':'cuda'}, ...]}

        실제 환경에서는 CuPy/Torch/Numba 등의 텐서에서 device pointer를 추출해 넣으면 된다.
        아래는 "어댑터 패턴" 스케치: 프로젝트 상황에 맞춰 하나 선택하여 구현.
        """
        def _to_desc(t: Tensor) -> Dict:
            return {
                "shape": list(getattr(t, "shape", [])),
                "dtype": getattr(t, "dtype", "unknown"),
                "layout": getattr(t, "layout", "rowmajor"),
                "device": getattr(t, "device", "cuda"),
                # "stride": getattr(t, "stride", []),  # 필요 시 확장
            }

        # --- 예시 어댑터(주석): Torch ---
        # import torch
        # def _ptr_from_torch(t):
        #     assert isinstance(t, torch.Tensor) and t.is_cuda
        #     return int(t.data_ptr())

        # --- 예시 어댑터(주석): CuPy ---
        # import cupy as cp
        # def _ptr_from_cupy(t):
        #     assert isinstance(t, cp.ndarray)
        #     return int(t.data.ptr)

        # 지금은 스켈레톤: 포인터 변환이 없으므로 빈 리스트 반환.
        bufs: List[int] = []
        descs = {"buffers": [_to_desc(t) for t in (op.inputs + op.outputs)]}
        return bufs, descs
