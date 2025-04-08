# tests/backend_test_code/activation_test.py 상단에 추가
import importlib.util

print("📁 activations_cuda 검색 중...")
spec = importlib.util.find_spec("activations_cuda")
print("📦 activations_cuda 모듈 위치:", spec.origin if spec else "❌ 찾을 수 없음")
