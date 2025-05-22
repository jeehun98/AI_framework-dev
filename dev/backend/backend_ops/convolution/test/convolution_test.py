import torch
import convolution_cuda

# ✅ 입력 크기: (batch=2, in_channels=3, height=32, width=32)
x = torch.randn(2, 3, 32, 32, device='cuda')

# ✅ 필터 크기: (out_channels=8, in_channels=3, kernel_size=5x5)
w = torch.randn(8, 3, 5, 5, device='cuda')

# ✅ bias 크기: (out_channels,)
b = torch.randn(8, device='cuda')

# ✅ stride, padding 설정
stride = 1
padding = 2  # padding=2 → output height/width 유지됨

# ✅ CUDA 커널 호출
out = convolution_cuda.forward(x, w, b, stride, padding)

# ✅ 출력 shape 확인
print(f"✅ 출력 shape: {out.shape}")  # 예상: (2, 8, 32, 32)
