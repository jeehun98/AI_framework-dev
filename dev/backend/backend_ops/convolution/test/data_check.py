import torch

def print_tensor_info(name, tensor):
    print(f"{name}: shape={tensor.shape}, device={tensor.device}, data_ptr={tensor.data_ptr()}")

# 기본 CPU 텐서
a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
print_tensor_info("CPU Tensor a", a)

# CPU 텐서를 GPU로 복사
b = a.to('cuda')
print_tensor_info("GPU Tensor b (from a)", b)

# 직접 GPU에서 생성한 텐서
c = torch.zeros((2, 2), device='cuda')
print_tensor_info("GPU Tensor c (direct)", c)

# GPU 텐서를 다시 CPU로 복사
d = c.to('cpu')
print_tensor_info("CPU Tensor d (from c)", d)
