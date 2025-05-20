import torch
import torch.nn.functional as F
import time

x = torch.randn(1024, 1024).cuda()
W = torch.randn(1024, 1024).cuda()
b = torch.randn(1024).cuda()

start = time.time()
for _ in range(1000):
    z = F.linear(x, W, b)  # Dense
    out = F.relu(z)        # Activation
torch.cuda.synchronize()
print("비최적화 실행 시간:", time.time() - start)
