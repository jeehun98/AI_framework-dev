# -*- coding: utf-8 -*-
"""
IR(Graph) → Pass → 커널선택 → (옵션) 네이티브 실행
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

# 네이티브 바인딩 선택: v2 우선, 없으면 v1 폴백. 환경변수로 강제 가능.
PREFERRED = os.environ.get("GE_NATIVE", "graph_executor_v2")

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
            # selector 등에서 'graph_executor'로 접근 가능하도록 별칭
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
    def __init__(
        self,
        device_caps: Dict[str, bool] | None = None,
        stream: int | None = 0,
        dry_run: bool = True,
        native_module=None,
    ):
        self.device_caps = device_caps or {"tensor_core": True}
        self.stream = stream
        self.dry_run = dry_run

        self.native = native_module or _native
        self.has_launch = hasattr(self.native, "launch_kernel") if self.native else False
        self.has_query = any(
            hasattr(self.native, n) for n in ("query_capability", "query_kernels")
        ) if self.native else False

    def run(self, graph: Graph):
        pm = PassManager([canonicalize, fuse_elementwise])
        plan = pm.run(graph)

        for op in plan.ops:
            # 아직 fuse 안 된 단일 op들은 데모 단계라 스킵
            if op.op_type in ("MATMUL", "BIAS_ADD", "RELU", "GELU"):
                print(f"[SKIP] unfused op: {op.op_type}")
                continue

            kname = pick_kernel(op, self.device_caps)
            bufs, descs = self._prepare_buffers(op)

            if self.dry_run:
                print(f"[DRY] launch {kname} with {len(bufs)} buffers attrs={op.attrs}")
                continue

            if not self.has_launch or not self.native:
                info = _debug_backend()
                raise RuntimeError(
                    "native.launch_kernel가 없습니다. 바인딩/로딩 확인 필요. "
                    f"backend_info={info}"
                )

            stream_i = 0 if self.stream is None else int(self.stream)
            self.native.launch_kernel(kname, bufs, descs, stream_i)

    # === 프레임워크 텐서 → device ptr/desc 변환 자리 ===
    def _prepare_buffers(self, op: Op) -> Tuple[List[int], Dict]:
        """
        규약:
        - bufs: [in0_ptr, in1_ptr, ..., out0_ptr, ...] (각각 uintptr_t로 변환된 device ptr)
        - descs: {'buffers': [{'shape':[...], 'dtype':'f16', 'layout':'rowmajor', 'device':'cuda'}, ...]}
        """
        def _to_desc(t: Tensor) -> Dict:
            return {
                "shape": list(getattr(t, "shape", [])),
                "dtype": getattr(t, "dtype", "unknown"
