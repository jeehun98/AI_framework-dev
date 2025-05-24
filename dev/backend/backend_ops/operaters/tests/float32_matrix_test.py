import cupy as cp
import matrix_ops

def test_float16_accuracy(M=1024, K=1024, N=1024):
    print(f"🚀 테스트 시작: {M}x{K} × {K}x{N} (float16)")

    # ✅ 입력 생성
    A = cp.random.rand(M, K).astype(cp.float16)
    B = cp.random.rand(K, N).astype(cp.float16)
    C = cp.zeros((M, N), dtype=cp.float16)

    # ✅ CUDA float16 행렬 곱
    matrix_ops.matrix_mul_half(A, B, C, M, N, K)

    # ✅ CuPy float16 행렬 곱 (참조값)
    expected = A @ B  # CuPy 내부 연산은 float32로 올려서 더 정확할 수 있음

    # ✅ 결과 비교 (float32로 올려서 정확도 판단)
    C_f32 = C.astype(cp.float32)
    expected_f32 = expected.astype(cp.float32)

    max_diff = cp.max(cp.abs(C_f32 - expected_f32)).item()
    mean_diff = cp.mean(cp.abs(C_f32 - expected_f32)).item()
    is_close = cp.allclose(C_f32, expected_f32, atol=0.1)

    print(f"🧪 최대 오차: {max_diff:.6f}")
    print(f"📊 평균 오차: {mean_diff:.6f}")
    print(f"✅ allclose (atol=0.1): {is_close}")

    if is_close:
        print("🎉 float16 행렬 곱 결과가 신뢰할 수 있습니다.")
    else:
        print("⚠️ float16 결과가 차이가 큽니다. 정밀도가 필요한 경우 float32 사용을 권장합니다.")

if __name__ == "__main__":
    test_float16_accuracy()
