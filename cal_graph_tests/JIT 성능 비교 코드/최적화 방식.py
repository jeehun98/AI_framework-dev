import torch
import torch.nn.functional as F
import time

torch.set_float32_matmul_precision('high')


@torch.compile  # PyTorch 2.0+ JIT
def fused_dense_relu(x, W, b):
    z = F.linear(x, W, b)
    return F.relu(z)

x = torch.randn(1024, 1024).cuda()
W = torch.randn(1024, 1024).cuda()
b = torch.randn(1024).cuda()

start = time.time()
for _ in range(1000):
    out = fused_dense_relu(x, W, b)
torch.cuda.synchronize()
print("최적화(fused) 실행 시간:", time.time() - start)
