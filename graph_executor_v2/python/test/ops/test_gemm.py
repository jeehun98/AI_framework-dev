# python/test/ops/test_gemm_lowlevel.py
# ============================================================
# Low-level GEMM bindings tests (ai::Tensor 기반)
# - Backend: CuPy -> Torch fallback (CUDA 필수)
# - Targets:
#   * forward/backward NN path (row-major, f32)
#   * save_z 경로 동작 확인
#   * gBias PerN 강제 가드 (PerM 형태는 에러)
#   * 다양한 activation(epilogue) 점검
#   * (선택) numpy helper와 수치 비교
# ============================================================

import os, sys
import numpy as np
import pytest

# --- import path / CUDA DLLs (Windows) ---
THIS = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(THIS, "..", "..", ".."))
PKG  = os.path.join(ROOT, "python")
if PKG not in sys.path:
    sys.path.insert(0, PKG)

cuda_bins = [
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin",
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5\bin",
]
if hasattr(os, "add_dll_directory"):
    for d in cuda_bins:
        if os.path.isdir(d):
            os.add_dll_directory(d)

# --- pick GPU array backend: CuPy -> Torch fallback ---
xp = None
cupy = None
torch = None
use_cupy = False
use_torch = False
_backend_name = None
try:
    import cupy as cp
    _ = cp.arange(1, dtype=cp.float32)  # sanity
    cupy = cp
    xp = cp
    use_cupy = True
    _backend_name = "cupy"
except Exception:
    try:
        import torch as _torch
        assert _torch.cuda.is_available(), "CUDA not available for torch"
        torch = _torch
        xp = _torch
        use_torch = True
        _backend_name = "torch"
    except Exception as e:
        raise RuntimeError("Need CuPy or PyTorch CUDA to run low-level tests.") from e


"""
graph_executor_v2.ops.gemm 단일 실행 테스트 스크립트
-----------------------------------------------------
▶ 실행: python test_gemm_run.py
▶ 요구사항: CuPy, CUDA, graph_executor_v2.ops._ops_gemm 빌드 완료
"""
import math
import cupy as cp
from graph_executor_v2.ops import gemm as gemm_ops

# --- 호환 유틸 ---
# --- 호환 유틸 (업데이트 버전) ---
def _instantiate_graph(graph):
    # 신버전: Graph.instantiate() -> GraphExec
    if hasattr(graph, "instantiate"):
        try:
            return graph.instantiate()
        except TypeError:
            pass  # 일부 버전 시그니처 차이 대비
    # 구버전: end_capture()가 이미 실행 가능 객체를 반환
    return graph

def _launch_graph(exec_graph, stream):
    # 1) 객체 메서드 우선 (버전에 따라 Stream 또는 ptr 기대)
    if hasattr(exec_graph, "launch"):
        try:
            # 대부분의 CuPy 버전이 Stream 객체를 기대
            exec_graph.launch(stream)
            return
        except TypeError:
            # 일부는 정수 ptr을 기대
            exec_graph.launch(stream.ptr)
            return

    # 2) 모듈 함수 형태(cp.cuda.graph.launch)가 있는 버전
    if hasattr(cp.cuda.graph, "launch"):
        try:
            cp.cuda.graph.launch(exec_graph, stream)
            return
        except TypeError:
            cp.cuda.graph.launch(exec_graph, stream.ptr)
            return

    raise RuntimeError("CUDA Graph launch API not found for this CuPy version.")


def ref_act(x, act: str, leaky=0.01):
    s = (act or "none").lower().replace("_","").replace("-","")
    if s == "none": return x
    if s == "relu": return cp.maximum(x,0)
    if s in ("leakyrelu","lrelu"): return cp.where(x>=0,x,leaky*x)
    if s == "sigmoid": return 1/(1+cp.exp(-x))
    if s == "tanh": return cp.tanh(x)
    if s == "gelu":
        c = math.sqrt(2/math.pi)
        return 0.5*x*(1+cp.tanh(c*(x+0.044715*x**3)))
    raise ValueError(act)

def ref_dact(z, act, leaky=0.01):
    s = (act or "none").lower().replace("_","").replace("-","")
    if s == "none": return cp.ones_like(z)
    if s == "relu": return (z>0).astype(cp.float32)
    if s in ("leakyrelu","lrelu"): return cp.where(z>=0,1,leaky)
    if s == "sigmoid":
        y = 1/(1+cp.exp(-z)); return y*(1-y)
    if s == "tanh":
        t = cp.tanh(z); return 1-t**2
    if s == "gelu":
        srt = math.sqrt(2/math.pi); a=0.044715
        t = cp.tanh(srt*(z+a*z**3))
        dt = (1-t**2)*srt*(1+3*a*z**2)
        return 0.5*(1+t)+0.5*z*dt
    raise ValueError(act)

def ref_forward(A,B,bias,act,leaky=0.01):
    Z = A@B
    if bias is not None:
        if bias.ndim==1: Z+=bias[None,:]
        elif bias.shape==(1,B.shape[1]): Z+=bias
        elif bias.shape==(A.shape[0],1): Z+=bias
        elif bias.shape==Z.shape: Z+=bias
        else: raise ValueError("bias shape")
    Y = ref_act(Z,act,leaky)
    return Y,Z

def ref_backward(A,B,gY,Z,act,leaky=0.01):
    dZ = gY*ref_dact(Z,act,leaky)
    gA = dZ@B.T
    gB = A.T@dZ
    gBias = cp.sum(dZ,axis=0,keepdims=True)
    return gA,gB,gBias

def allclose(a,b):
    return cp.allclose(a,b,atol=2e-4,rtol=2e-4)

def run_basic_test(M=16,K=32,N=8,act="relu"):
    print(f"\n[TEST] forward/backward ({M},{K},{N}), act={act}")
    rng = cp.random.default_rng(0)
    A = rng.standard_normal((M,K),dtype=cp.float32)
    B = rng.standard_normal((K,N),dtype=cp.float32)
    bias = cp.zeros((1,N),dtype=cp.float32)

    Y_ref,Z_ref = ref_forward(A,B,bias,act)
    Y,Z = gemm_ops.forward(A,B,bias=bias,with_bias=True,act=act,return_z=True)

    print("  forward:", allclose(Y,Y_ref), allclose(Z,Z_ref))

    gY = rng.standard_normal((M,N),dtype=cp.float32)
    gA_ref,gB_ref,gBias_ref = ref_backward(A,B,gY,Z_ref,act)
    out = gemm_ops.backward(A,B,gY,Z,act=act,with_bias=True,want_gA=True,want_gB=True,want_gBias=True)
    print("  backward:", allclose(out["gA"],gA_ref), allclose(out["gB"],gB_ref), allclose(out["gBias"],gBias_ref))

def run_capture_safe_test():
    print("\n[TEST] capture-safe forward_into/backward_into (CUDA Graph)")
    M,K,N = 64,128,32
    rng = cp.random.default_rng(1)
    A = rng.standard_normal((M,K),dtype=cp.float32); A=cp.ascontiguousarray(A)
    B = rng.standard_normal((K,N),dtype=cp.float32); B=cp.ascontiguousarray(B)
    bias = cp.zeros((1,N),dtype=cp.float32); bias=cp.ascontiguousarray(bias)
    Y = cp.empty((M,N),dtype=cp.float32)
    Z = cp.empty((M,N),dtype=cp.float32)
    gY = rng.standard_normal((M,N),dtype=cp.float32); gY=cp.ascontiguousarray(gY)
    gA = cp.empty_like(A)
    gB = cp.empty_like(B)
    gBias = cp.empty((1,N),dtype=cp.float32)
    ws = gemm_ops.ensure_workspaces(M,N,lt_bytes=4<<20)

    # 수정 (스트림 캡처 API 사용)
    stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        stream.begin_capture()

        gemm_ops.forward_into(A, B, out=Y, bias=bias, with_bias=True, act="relu", save_z=True, z_out=Z)
        gemm_ops.backward_into(A, B, gY, Z, with_bias=True, act="relu",
                            gA_out=gA, gB_out=gB, gBias_out=gBias,
                            work_dZ=ws.dZ, lt_workspace=ws.lt_ws)

        graph = stream.end_capture()          # 버전에 따라 Graph 또는 실행가능 객체 반환
        exec_graph = _instantiate_graph(graph)

        _launch_graph(exec_graph, stream)     # warmup
        _launch_graph(exec_graph, stream)     # check
    stream.synchronize()



    Y_ref,Z_ref = ref_forward(A,B,bias,"relu")
    gA_ref,gB_ref,gBias_ref = ref_backward(A,B,gY,Z_ref,"relu")
    print("  forward_into:", allclose(Y,Y_ref), allclose(Z,Z_ref))
    print("  backward_into:", allclose(gA,gA_ref), allclose(gB,gB_ref), allclose(gBias,gBias_ref))

if __name__=="__main__":
    cp.cuda.runtime.setDevice(0)
    
    run_basic_test(16,32,8,"relu")
    run_basic_test(16,32,8,"gelu")
    run_basic_test(8,16,4,"none")
    run_capture_safe_test()
    print("\n✅ All tests executed.")
