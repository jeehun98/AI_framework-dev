import cupy as cp
import matrix_ops

def benchmark(name, func, *args):
    start = cp.cuda.Event(); end = cp.cuda.Event()
    start.record()
    func(*args)
    end.record(); end.synchronize()
    elapsed = cp.cuda.get_elapsed_time(start, end)  # ms
    print(f"⏱️ {name}: {elapsed:.3f} ms")
    return elapsed

def run_speed_test(M=4096, K=2048, N=2048):
    print(f"🚀 테스트 시작: {M}x{K} × {K}x{N}")

    # float32
    A32 = cp.random.rand(M, K).astype(cp.float32)
    B32 = cp.random.rand(K, N).astype(cp.float32)
    C32_add = cp.zeros((M, K), dtype=cp.float32)
    C32_mul = cp.zeros((M, N), dtype=cp.float32)

    # float16
    A16 = A32.astype(cp.float16)
    B16 = B32.astype(cp.float16)
    C16_add = cp.zeros((M, K), dtype=cp.float16)
    C16_mul = cp.zeros((M, N), dtype=cp.float16)

    print("\n🧪 덧셈 연산 시간")
    benchmark("float32 덧셈", matrix_ops.matrix_add, A32, B32, C32_add, M, K)
    benchmark("float16 덧셈", matrix_ops.matrix_add_half, A16, B16, C16_add, M, K)

    print("\n🧪 곱셈 연산 시간")
    benchmark("float32 곱셈", matrix_ops.matrix_mul, A32, B32, C32_mul, M, N, K)
    benchmark("float16 곱셈", matrix_ops.matrix_mul_half, A16, B16, C16_mul, M, N, K)

if __name__ == "__main__":
    run_speed_test()
