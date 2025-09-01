# dev/backend/compiler/runtime/executor.py
from __future__ import annotations
from typing import Dict, Tuple, List
import importlib
import os

from ..ir.nodes import Graph, Tensor, Op
from ..passes.pass_manager import PassManager
from ..passes.canonicalize import canonicalize
from ..passes.fuse_elementwise import fuse_elementwise
from ..kernels.selector import pick_kernel

# ---------------------------------------------------------------------
# 네이티브 바인딩 선택: v2 우선, 없으면 v1 폴백. 환경변수로 강제 가능.
# ---------------------------------------------------------------------
PREFERRED = os.environ.get("GE_NATIVE", "graph_executor_v2")
CANDIDATES = (PREFERRED, "graph_executor_v2", "graph_executor")

_native = None
_chosen = None
_last_err: Exception | None = None
for _mod in CANDIDATES:
    try:
        _native = importlib.import_module(_mod)
        _chosen = _mod
        break
    except Exception as e:
        _last_err = e
        _native = None

_HAS_LAUNCH = hasattr(_native, "launch_kernel") if _native else False
_HAS_QUERY = hasattr(_native, "query_capability") if _native else False


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
    Python 주도: IR → Pass → 커널선택 → (옵션) 네이티브 실행
    - device_caps: 하드웨어 기능 힌트 (예: {"tensor_core": True})
    - stream: cudaStream_t를 정수(uintptr)로 전달 (옵션)
    - dry_run: True면 커널 선택/플랜만 출력하고 launch는 생략
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
        self.has_query = hasattr(self.native, "query_capability") if self.native else False

    def run(self, graph: Graph):
        # 1) 패스 파이프라인
        pm = PassManager([canonicalize, fuse_elementwise])
        plan = pm.run(graph)

        # 2) 커널 선택 + 실행
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

    # === 네 프레임워크 텐서로 교체할 자리 ===
    def _prepare_buffers(self, op: Op) -> Tuple[List[int], Dict]:
        """
        지금은 스켈레톤: 디바이스 포인터/디스크립터를 준비해 native에 넘긴다.
        - bufs: [in0_ptr, in1_ptr, ..., out0_ptr, ...] (각각 uintptr_t로 변환된 device ptr)
        - descs: {'buffers': [{'shape':[...], 'dtype':'f16', 'layout':'rowmajor', 'device':'cuda'}, ...]}
        실제 환경에서는 CuPy/torch 텐서에서 device pointer를 추출해 넣으면 된다.
        """
        def _to_desc(t: Tensor) -> Dict:
            return {
                "shape": list(t.shape),
                "dtype": t.dtype,
                "layout": t.layout,
                "device": t.device,
            }

        # NOTE: 현재는 포인터 없음(드라이런/스켈레톤). 통합 시:
        # bufs = [int(x.data.ptr), int(w.data.ptr), ..., int(y.data.ptr)]
        bufs: List[int] = []
        descs = {"buffers": [_to_desc(t) for t in (op.inputs + op.outputs)]}
        return bufs, descs
