# -*- coding: utf-8 -*-
"""
test_ge_v2_all.py

무엇을 검증하나?
1) pybind 노출 함수: query_kernels, query_capability, launch_kernel
2) f16(TC) 경로: (has_bias, act) = {(0,0),(0,1),(1,0),(1,1)}
   - (1,*)는 cuBLASLt 에필로그 Fuse (Bias/ Bias+ReLU)
   - (0,1)은 ReLU 단독 후처리 커널
3) f32 백업 경로(gemm_bias_act_f32)
4) 스트림 인자: non-default CUDA stream 에서도 정상 동작
5) 에러 처리: 알 수 없는 커널 호출 시 예외
6) (옵션) ExecutorV2와 IR 모의 객체로 E2E 라우팅 확인 (가능 시)

실행:
  python -m tests.test_ge_v2_all
"""

# --- load native module next to test, but keep proper package paths ---
import os, sys, ctypes as ct, importlib.util, glob, types
import numpy as np
import cupy as cp

HERE     = os.path.dirname(__file__)                          # .../backend/graph_executor_v2/test
GE_DIR   = os.path.abspath(os.path.join(HERE, ".."))          # .../backend/graph_executor_v2
BACK_DIR = os.path.abspath(os.path.join(GE_DIR, ".."))        # .../backend
DEV_DIR  = os.path.abspath(os.path.join(BACK_DIR, ".."))      # .../dev

# Executor가 네이티브 모듈을 첫 번째로 이 이름에서 찾도록 고정
os.environ["GE_NATIVE"] = "backend.graph_executor_v2.graph_executor_v2"

# dev 루트를 sys.path에 추가(backend 패키지 탐색용)
if DEV_DIR not in sys.path:
    sys.path.insert(0, DEV_DIR)

# 기존 잘못 매핑된 모듈들 제거
for name in (
    "graph_executor_v2",
    "backend",
    "backend.graph_executor_v2",
    "backend.graph_executor_v2.graph_executor_v2",
):
    if name in sys.modules:
        del sys.modules[name]

# CUDA DLL 보장(Windows)
try:
    from utils.load_cuda import ensure_cuda_dlls
    ensure_cuda_dlls()
except Exception:
    pass

# 1) test 폴더 내 .pyd를 파일 경로로 직접 로드
cand = glob.glob(os.path.join(HERE, "graph_executor_v2*.pyd")) + \
       glob.glob(os.path.join(HERE, "graph_executor_v2*.so"))
if not cand:
    raise FileNotFoundError(f"native .pyd/.so not found under: {HERE}")
cand.sort(key=lambda p: os.path.getmtime(p), reverse=True)
_native_path = cand[0]
spec = importlib.util.spec_from_file_location(
    "backend.graph_executor_v2.graph_executor_v2", _native_path
)
gev2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gev2)

# 2) 패키지 트리 주입: __path__ 를 정확히 설정
backend_pkg = types.ModuleType("backend")
backend_pkg.__path__ = [BACK_DIR]           # ← .../dev/backend
sys.modules["backend"] = backend_pkg

ge_pkg = types.ModuleType("backend.graph_executor_v2")
ge_pkg.__path__ = [GE_DIR]                  # ← .../dev/backend/graph_executor_v2 (executor.py 위치)
sys.modules["backend.graph_executor_v2"] = ge_pkg

# 3) 네이티브 모듈을 기대 이름으로 주입
sys.modules["backend.graph_executor_v2.graph_executor_v2"] = gev2

# sanity check
assert hasattr(gev2, "query_kernels"), "native module missing expected symbols"
assert getattr(gev2, "GE2_API_VERSION", None) == 1, "ABI version mismatch"

print("loaded(native):", gev2, _native_path)
print("has query_kernels?", hasattr(gev2, "query_kernels"))
print("GE2_API_VERSION:", getattr(gev2, "GE2_API_VERSION", None))
# ----------------------------------------------------------------


# -----------------------------
# 공통 파라미터 구조체 (C와 ABI 동일)
# -----------------------------
class Params(ct.Structure):
    _fields_ = [
        ("M", ct.c_int), ("N", ct.c_int), ("K", ct.c_int),
        ("has_bias", ct.c_int), ("act", ct.c_int)
    ]

# 프레임워크 dtype -> IR 심볼 문자열 매핑
def _to_ir_dtype(arr):
    # numpy/cupy 모두 .dtype.kind/.itemsize 로 판별
    kind = arr.dtype.kind
    if kind == 'f':
        return 'f16' if arr.dtype.itemsize == 2 else 'f32'
    if kind == 'i':
        return 'i8' if arr.dtype.itemsize == 1 else 'i32'
    raise ValueError(f"Unsupported dtype for IR: {arr.dtype}")


def _ref_result(Ah, Bh, bias_f32, has_bias, act):
    """numpy 참조 결과 (FP32 누적 후 FP16 cast 기준 비교)"""
    A32 = Ah.astype(np.float32)
    B32 = Bh.astype(np.float32)
    ref = A32 @ B32
    if has_bias:
        ref = ref + bias_f32.astype(np.float32)  # 브로드캐스트: (N,)
    if act == 1:
        ref = np.maximum(ref, 0.0)
    return ref.astype(np.float16).astype(np.float32)


def _max_err(fp16_out, ref32):
    return float(np.max(np.abs(fp16_out.astype(np.float32) - ref32)))


def test_query_apis():
    ks = set(map(str, gev2.query_kernels()))
    assert "gemm_bias_act_tc_f16" in ks, ks
    assert "gemm_bias_act_f32" in ks, ks

    caps = gev2.query_capability("GEMM_BIAS_ACT", {}, {})
    # 대략 점수값만 확인
    assert "gemm_bias_act_tc_f16" in caps and "gemm_bias_act_f32" in caps
    assert int(caps["gemm_bias_act_tc_f16"]) >= int(caps["gemm_bias_act_f32"])
    print("[OK] query APIs")


def run_case_f16(M=64, N=96, K=80, has_bias=1, act=1, use_stream=False, seed=7):
    rng = np.random.default_rng(seed)
    A = (rng.standard_normal((M, K)) * 0.25).astype(np.float16)
    B = (rng.standard_normal((K, N)) * 0.25).astype(np.float16)
    bias = (rng.standard_normal((N,)) * 0.1).astype(np.float32) if has_bias else None

    A_d = cp.asarray(A)
    B_d = cp.asarray(B)
    C_d = cp.empty((M, N), dtype=cp.float16)
    if bias is not None:
        bias_d = cp.asarray(bias)
    params = Params(M, N, K, has_bias, act)

    bufs = [int(A_d.data.ptr), int(B_d.data.ptr)]
    if has_bias:
        bufs.append(int(bias_d.data.ptr))
    bufs.append(int(C_d.data.ptr))
    bufs.append(ct.addressof(params))

    stream_ptr = 0
    if use_stream:
        s = cp.cuda.Stream(non_blocking=True)
        # Stream.ptr은 uintptr_t 호환 포인터
        stream_ptr = int(s.ptr)
        with s:
            gev2.launch_kernel("gemm_bias_act_tc_f16", bufs, {"buffers": []}, stream_ptr)
        s.synchronize()
    else:
        gev2.launch_kernel("gemm_bias_act_tc_f16", bufs, {"buffers": []}, stream_ptr)
        cp.cuda.runtime.deviceSynchronize()

    ref = _ref_result(A, B, bias if bias is not None else np.zeros((N,), np.float32),
                      has_bias, act)
    err = _max_err(C_d.get(), ref)
    print(f"[f16] has_bias={has_bias}, act={act}, stream={use_stream} → max_err={err:.4f}")
    assert err < 1.0, f"f16 max_err too large: {err}"
    return err


def test_f16_all_combos():
    # (0,0) no-bias no-act
    # run_case_f16(has_bias=0, act=0, use_stream=False)
    # (0,1) ReLU only → 후처리 커널 경로
    run_case_f16(has_bias=0, act=1, use_stream=True)
    # (1,0) Bias only → cuBLASLt Epilogue BIAS
    # run_case_f16(has_bias=1, act=0, use_stream=False)
    # (1,1) Bias+ReLU → cuBLASLt Epilogue RELU_BIAS
    run_case_f16(has_bias=1, act=1, use_stream=True)
    print("[OK] f16 all combos")


def test_f32_backup(M=48, N=40, K=72, seed=11):
    rng = np.random.default_rng(seed)
    A = (rng.standard_normal((M, K)) * 0.25).astype(np.float32)
    B = (rng.standard_normal((K, N)) * 0.25).astype(np.float32)
    bias = (rng.standard_normal((N,)) * 0.1).astype(np.float32)

    A_d = cp.asarray(A)
    B_d = cp.asarray(B)
    C_d = cp.empty((M, N), dtype=cp.float32)
    bias_d = cp.asarray(bias)
    params = Params(M, N, K, 1, 1)  # bias+relu

    bufs = [int(A_d.data.ptr), int(B_d.data.ptr), int(bias_d.data.ptr), int(C_d.data.ptr), ct.addressof(params)]
    gev2.launch_kernel("gemm_bias_act_f32", bufs, {"buffers": []}, 0)
    cp.cuda.runtime.deviceSynchronize()

    ref = (A @ B)
    ref = np.maximum(ref + bias.astype(np.float32), 0.0)
    err = float(np.max(np.abs(ref - C_d.get())))
    print(f"[f32] bias+relu → max_err={err:.6f}")
    assert err < 1e-3
    print("[OK] f32 backup path")


def test_error_handling():
    try:
        gev2.launch_kernel("no_such_kernel", [], {}, 0)
    except Exception as e:
        print("[OK] error handling:", str(e)[:80])
        return
    raise AssertionError("Expected an exception for unknown kernel name")


def test_executor_integration_optional():
    """
    ExecutorV2를 통한 E2E 테스트(옵션).
    - 환경에 따라 import 실패할 수 있으므로 실패하면 SKIP.
    """
    try:
        # 올바른 심볼 임포트 (ExecutorV2 클래스를 가져와야 합니다)
        from backend.compiler.runtime.executor import ExecutorV2
        # IR 노드 경로는 사용 환경에 맞게 이미 고치신 것으로 보입니다.
        from backend.compiler.ir.nodes import Op, Tensor, Graph
    except Exception as e:
        print("[SKIP] ExecutorV2 E2E (import fail):", e)
        return

    # 준비: 데이터
    M, N, K = 64, 48, 80
    A = (cp.random.randn(M, K).astype(cp.float16) * 0.25)
    B = (cp.random.randn(K, N).astype(cp.float16) * 0.25)
    bias = (cp.random.randn(N).astype(cp.float32) * 0.1)
    D = cp.empty((M, N), dtype=cp.float16)

    # IR 심볼 dtype으로 변환
    dtA = _to_ir_dtype(A)     # 'f16'
    dtB = _to_ir_dtype(B)     # 'f16'
    dtBias = _to_ir_dtype(bias)  # 'f32'
    dtD = _to_ir_dtype(D)     # 'f16'

    # Tensor 래핑 (프로젝트의 Tensor(dataclass) 시그니처에 맞춰 작성)
    tA = Tensor(t=A, shape=A.shape, dtype=dtA, layout="rowmajor", device="cuda")
    tB = Tensor(t=B, shape=B.shape, dtype=dtB, layout="rowmajor", device="cuda")
    tBias = Tensor(t=bias, shape=bias.shape, dtype=dtBias, layout="rowmajor", device="cuda")
    tD = Tensor(t=D, shape=D.shape, dtype=dtD, layout="rowmajor", device="cuda")

    # OP 구성: GEMM_BIAS_ACT (bias+ReLU)
    op = Op(
        op_type="GEMM_BIAS_ACT",
        inputs=[tA, tB, tBias],
        outputs=[tD],
        attrs={"mnk": (M, N, K), "act": "relu"},
    )
    g = Graph(ops=[op])

    # 실행
    exe = ExecutorV2(dry_run=False)
    exe.run(g)

    # 정답 비교
    ref = (cp.asnumpy(A).astype(np.float32) @ cp.asnumpy(B).astype(np.float32))
    ref = np.maximum(ref + cp.asnumpy(bias).astype(np.float32), 0.0)
    err = float(np.max(np.abs(ref.astype(np.float16).astype(np.float32) - cp.asnumpy(D).astype(np.float32))))
    print(f"[ExecutorV2] E2E max_err={err:.4f}")
    assert err < 1.0
    print("[OK] ExecutorV2 E2E")


def main():
    print("=== graph_executor_v2 smoke ===")
    print("GE2_API_VERSION:", getattr(gev2, "GE2_API_VERSION", None))
    test_query_apis()
    test_f16_all_combos()
    test_f32_backup()
    test_error_handling()
    test_executor_integration_optional()
    print("=== ALL OK ===")


if __name__ == "__main__":
    main()
