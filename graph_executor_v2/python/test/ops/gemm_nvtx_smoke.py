import os, sys, torch
os.add_dll_directory(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin")
ROOT = r"C:\Users\owner\Desktop\AI_framework-dev\graph_executor_v2\python"
if ROOT not in sys.path: sys.path.insert(0, ROOT)

from graph_executor_v2.ops import _ops_common  # 선행
from graph_executor_v2.ops import _ops_gemm as gemm

M,K,N = 128, 256, 192
A = torch.randn(M,K, device="cuda", dtype=torch.float32)
B = torch.randn(K,N, device="cuda", dtype=torch.float32)
Y = torch.empty(M,N, device="cuda", dtype=torch.float32)

to_ai = lambda t: gemm.make_tensor_2d(int(t.data_ptr()), list(t.shape),
                                      gemm.DType.F32, gemm.Device.CUDA,
                                      0 if t.device.index is None else t.device.index)
attrs = gemm.GemmAttrs()
attrs.act = gemm.ActKind.ReLU
gemm.forward(to_ai(A), to_ai(B), None, to_ai(Y), attrs, None, None)
torch.cuda.synchronize()
