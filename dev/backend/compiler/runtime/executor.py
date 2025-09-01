from __future__ import annotations
from typing import Dict, Tuple, List
import importlib
from ..ir.nodes import Graph, Tensor, Op
from ..passes.pass_manager import PassManager
from ..passes.canonicalize import canonicalize
from ..passes.fuse_elementwise import fuse_elementwise
from ..kernels.selector import pick_kernel

# graph_executor(.pyd) 로딩 (없어도 드라이런 가능)
try:
    native = importlib.import_module("graph_executor")
    _HAS_LAUNCH = hasattr(native, "launch_kernel")
except Exception:
    native = None
    _HAS_LAUNCH = False

class ExecutorV2:
    """
    Python 주도: IR → Pass → 커널선택 → (옵션) 네이티브 실행
    """
    def __init__(self, device_caps: Dict[str, bool] | None = None, stream: int | None = None, dry_run: bool = True):
        self.device_caps = device_caps or {"tensor_core": True}
        self.stream = stream
        self.dry_run = dry_run

    def run(self, graph: Graph):
        # 1) 패스 파이프라인
        pm = PassManager([canonicalize, fuse_elementwise])
        plan = pm.run(graph)

        # 2) 커널 선택 + 실행
        for op in plan.ops:
            if op.op_type in ("MATMUL", "BIAS_ADD", "RELU", "GELU"):
                # 아직 fuse 안 된 남은 op들은 임시로 스킵(데모)
                print(f"[SKIP] unfused op: {op.op_type}")
                continue

            kname = pick_kernel(op, self.device_caps)
            bufs, descs = self._prepare_buffers(op)  # TODO: 실제 텐서 → 디바이스 포인터/desc
            if self.dry_run:
                print(f"[DRY] launch {kname} with {len(bufs)} buffers attrs={op.attrs}")
            else:
                if not _HAS_LAUNCH:
                    raise RuntimeError("native.launch_kernel가 없습니다. 바인딩에 추가하세요.")
                native.launch_kernel(kname, bufs, descs, self.stream)

    # === 네 프레임워크 텐서로 교체할 자리 ===
    def _prepare_buffers(self, op: Op) -> Tuple[List[int], Dict]:
        """
        지금은 스켈레톤: 디바이스 포인터/디스크립터를 준비해 native에 넘긴다.
        - bufs: [in0_ptr, in1_ptr, ..., out0_ptr, ...]
        - descs: {'buffers': [{'shape':[...], 'dtype':'f16', 'layout':'rowmajor'}, ...]}
        """
        def _to_desc(t: Tensor) -> Dict:
            return {"shape": list(t.shape), "dtype": t.dtype, "layout": t.layout, "device": t.device}

        # 실제로는 CuPy/torch 텐서에서 device pointer 꺼내서 넣으면 됨.
        bufs: List[int] = []   # 예: [x.data.ptr, w.data.ptr, b.data.ptr, y.data.ptr]
        descs = {"buffers": [_to_desc(t) for t in (op.inputs + op.outputs)]}
        return bufs, descs
