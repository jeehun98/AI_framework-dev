import cupy as cp
import matrix_ops
import time

def benchmark_matrix_mul():
    print("🚀 Testing speed on large matrix (1024 × 1024 × 1024)")

    M, K, N = 1024, 1024, 1024
    A = cp.random.rand(M, K).astype(cp.float32)
    B = cp.random.rand(K, N).astype(cp.float32)
    C1 = cp.zeros((M, N), dtype=cp.float32)
    C2 = cp.zeros((M, N), dtype=cp.float32)

    # Global Memory 방식
    start = cp.cuda.Event(); end = cp.cuda.Event()
    start.record()
    matrix_ops.matrix_mul(A, B, C1, M, N, K)
    end.record(); end.synchronize()
    time_global = cp.cuda.get_elapsed_time(start, end)  # ms

    # Shared Memory 방식
    start = cp.cuda.Event(); end = cp.cuda.Event()
    start.record()
    matrix_ops.matrix_mul_shared(A, B, C2, M, N, K)
    end.record(); end.synchronize()
    time_shared = cp.cuda.get_elapsed_time(start, end)  # ms

    # 정확도 확인
    diff = cp.linalg.norm(C1 - C2)
    max_diff = cp.max(cp.abs(C1 - C2))
    print(f"✅ Global time:  {time_global:.3f} ms")
    print(f"✅ Shared time:  {time_shared:.3f} ms")
    print(f"🧪 Max diff:     {max_diff:.5f}, L2 diff: {diff:.5f}")

    if time_shared < time_global:
        print("✅ Shared memory tiling is faster! 🎉")
    else:
        print("⚠️  Shared memory was not faster — maybe memory-bound or small tile size")

if __name__ == "__main__":
    benchmark_matrix_mul()
