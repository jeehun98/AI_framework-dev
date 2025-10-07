# python/test/graph/test_graph_exec_forward.py
import os, sys, time
THIS = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(THIS, "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import cupy as cp

from graph_executor_v2.layers.base import Layer
from graph_executor_v2.layers.conv2d import Conv2D
from graph_executor_v2.layers.linear import Linear
from graph_executor_v2.graph.graph_executor import GraphCompiler
from graph_executor_v2.ops import conv2d as conv_ops


# -----------------------------
# Mini layers
# -----------------------------
class ReLU(Layer):
    def call(self, x):
        return cp.maximum(x, 0)

    def build(self, input_shape):
        super().build(input_shape)
        self.output_shape = tuple(int(v) for v in input_shape)

    def compute_output_shape(self, input_shape):
        return tuple(int(v) for v in input_shape)


class Flatten(Layer):
    def call(self, x):
        n = int(x.shape[0])
        return x.reshape(n, -1)

    def build(self, input_shape):
        super().build(input_shape)
        if len(input_shape) != 4:
            raise ValueError(f"Flatten expects 4D input (N,C,H,W), got {input_shape}")
        n, c, h, w = map(int, input_shape)
        self.output_shape = (n, c*h*w)

    def compute_output_shape(self, input_shape):
        if len(input_shape) != 4:
            raise ValueError(f"Flatten expects 4D input (N,C,H,W), got {input_shape}")
        n, c, h, w = map(int, input_shape)
        return (n, c*h*w)


class MiniModel:
    def __init__(self):
        self.layers = [
            Conv2D(filters=8, kernel_size=(3,3), stride=(1,1), padding=(1,1), use_bias=True),
            ReLU(),
            Flatten(),
            Linear(out_features=10, use_bias=True),
        ]

    def build(self, input_shape):
        cur = tuple(int(v) for v in input_shape)
        for l in self.layers:
            l.build(cur)
            cur = l.compute_output_shape(cur)
        self.output_shape = cur


# -----------------------------
# Debug helpers
# -----------------------------
def _fmt_bytes(n):
    units = ["B","KB","MB","GB","TB"]
    i = 0
    x = float(n)
    while x >= 1024.0 and i < len(units)-1:
        x /= 1024.0; i += 1
    return f"{x:.2f} {units[i]}"

def _arena_snapshot(ge):
    try:
        arena = ge.arena
        total = 0
        lines = []
        for bid, buf in arena.buffers.items():
            alloc_elems = arena.alloc_elems.get(bid, buf.size)
            bytes_ = alloc_elems * buf.dtype.itemsize
            total += bytes_
            lines.append(f"  - buffer_id={bid:<2d} elems={alloc_elems:<10d} dtype={str(buf.dtype):<8s} bytes={_fmt_bytes(bytes_)}")
        return "Arena Buffers:\n" + "\n".join(lines) + f"\n  => total ~ {_fmt_bytes(total)}"
    except Exception as e:
        return f"(arena snapshot failed: {e})"

def _graph_info_fwd(ge):
    info = []
    try:
        info.append(f"capture_enabled: {ge._gexec is not None}")
        info.append(f"graph_api_kind: {getattr(ge, '_graph_api_kind', 'unknown')}")
        cap_stream = getattr(ge, "_cap_stream", None)
        if cap_stream is not None:
            info.append(f"cap_stream.ptr: 0x{int(cap_stream.ptr):x}")
        gexec = getattr(ge, "_gexec", None)
        if gexec is not None:
            try:
                ptr_val = int(gexec) if isinstance(gexec, int) else id(gexec)
                info.append(f"graph_exec_ptr: 0x{ptr_val:x}")
            except Exception:
                info.append(f"graph_exec_obj: {type(gexec)}")
    except Exception as e:
        info.append(f"(graph info failed: {e})")
    return " | ".join(info)

def _graph_info_bwd(ge):
    info = []
    try:
        info.append(f"capture_enabled: {ge._gexec_bwd is not None}")
        info.append(f"graph_api_kind_bwd: {getattr(ge, '_graph_api_kind_bwd', 'unknown')}")
        cap_stream = getattr(ge, "_cap_stream_bwd", None)
        if cap_stream is not None:
            info.append(f"cap_stream_bwd.ptr: 0x{int(cap_stream.ptr):x}")
        gexec = getattr(ge, "_gexec_bwd", None)
        if gexec is not None:
            try:
                ptr_val = int(gexec) if isinstance(gexec, int) else id(gexec)
                info.append(f"graph_exec_bwd_ptr: 0x{ptr_val:x}")
            except Exception:
                info.append(f"graph_exec_bwd_obj: {type(gexec)}")
    except Exception as e:
        info.append(f"(graph bwd info failed: {e})")
    return " | ".join(info)

def _dump_ws_shapes(ge):
    try:
        lines = []
        for i, n in enumerate(ge.ir.nodes):
            if n.kind == "Conv2D" and isinstance(n.ws_fwd, conv_ops.Conv2DWorkspaces):
                ws = n.ws_fwd
                def shp(x): return None if x is None else tuple(int(v) for v in x.shape)
                lines.append(
                    f"  [node#{i} Conv2D WS] "
                    f"dCol={shp(ws.dCol)}, W_KC={shp(ws.W_KC)}, Y_tmp={shp(ws.Y_tmp)}, Z_rows={shp(ws.Z_rows)}"
                )
            if n.kind == "GEMM":
                gz = n.attrs.get("gemm_ws_dZ", None)
                lt = n.attrs.get("gemm_lt_ws", None)
                if gz is not None or lt is not None:
                    gz_s = None if gz is None else tuple(map(int, gz.shape))
                    lt_b = None if lt is None else lt.size
                    lines.append(f"  [node#{i} GEMM WS] dZ={gz_s}, lt_ws_bytes={lt_b}")
        if not lines:
            return "(no WS found)"
        return "Workspaces:\n" + "\n".join(lines)
    except Exception as e:
        return f"(ws dump failed: {e})"

def _env_info():
    try:
        dev = cp.cuda.Device()
        props = cp.cuda.runtime.getDeviceProperties(dev.id)
        name_v = props.get("name")
        name = name_v.decode() if isinstance(name_v, (bytes, bytearray)) else name_v
        drv = cp.cuda.runtime.driverGetVersion()
        rt  = cp.cuda.runtime.runtimeGetVersion()
        return (f"Device: {name}, CC {props.get('major')}.{props.get('minor')} | "
                f"Driver {drv}, Runtime {rt} | CuPy {cp.__version__}")
    except Exception as e:
        return f"(env info failed: {e})"


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    VERBOSE = True
    cp.random.seed(1234)

    print("[ENV]", _env_info())

    N, C, H, W = 4, 3, 32, 32
    x = cp.random.randn(N, C, H, W).astype(cp.float32)

    model = MiniModel()
    model.build(x.shape)

    # ----- Compile with CUDA Graph capture (FWD+BWD) -----
    ge_cap = GraphCompiler(model).compile(
        input_shape=x.shape,
        use_cuda_graph=True,
        use_cuda_graph_bwd=True
    )

    # FWD run
    y = ge_cap.run(x)
    print(f"[RUN] y.shape={tuple(map(int, y.shape))}  max={float(y.max()):.6f}  norm={float(cp.linalg.norm(y)):.6f}")

    if VERBOSE:
        print("[DEBUG] Forward Capture Info:", _graph_info_fwd(ge_cap))
        print("[DEBUG] Backward Capture Info:", _graph_info_bwd(ge_cap))
        print("[DEBUG]", _arena_snapshot(ge_cap))
        print("[DEBUG]", _dump_ws_shapes(ge_cap))

    # Non-captured (reference)
    ge_nocap = GraphCompiler(model).compile(
        input_shape=x.shape,
        use_cuda_graph=False,
        use_cuda_graph_bwd=False
    )
    y_nc = ge_nocap.run(x)
    max_abs = float(cp.max(cp.abs(y - y_nc)))
    print(f"[CHECK] captured vs non-captured (FWD) max_abs_diff = {max_abs:.6e}")

    # ----- Backward check -----
    gout = cp.random.randn(*y.shape).astype(cp.float32)
    gx_cap = ge_cap.backward(gout)
    gx_nc  = ge_nocap.backward(gout)
    gdiff = float(cp.max(cp.abs(gx_cap - gx_nc)))
    print(f"[CHECK] captured vs non-captured (BWD) max_abs_diff = {gdiff:.6e}")

    # ----- Timing -----
    def time_gpu(fn, iters=30):
        s = cp.cuda.Event(); e = cp.cuda.Event()
        for _ in range(5):
            fn()
        cp.cuda.get_current_stream().synchronize()
        s.record()
        for _ in range(iters):
            fn()
        e.record(); e.synchronize()
        return cp.cuda.get_elapsed_time(s, e) / iters  # ms

    t_f_cap = time_gpu(lambda: ge_cap.run(x), iters=30)
    t_f_nc  = time_gpu(lambda: ge_nocap.run(x), iters=30)
    t_b_cap = time_gpu(lambda: ge_cap.backward(gout), iters=30)
    t_b_nc  = time_gpu(lambda: ge_nocap.backward(gout), iters=30)

    print(f"[TIME] FWD  captured: {t_f_cap:.3f} ms | non-cap: {t_f_nc:.3f} ms")
    print(f"[TIME] BWD  captured: {t_b_cap:.3f} ms | non-cap: {t_b_nc:.3f} ms")

    # sanity
    if not cp.all(cp.isfinite(y)):
        print("[WARN] Non-finite values detected in output!")

    print("[OK] forward+backward-capture test completed.")
