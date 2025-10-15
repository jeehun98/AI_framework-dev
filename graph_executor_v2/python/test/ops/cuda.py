import torch, sys
print("python:", sys.version)
print("torch :", torch.__version__)
print("torch.version.cuda:", torch.version.cuda)        # 빌드된 CUDA 버전 표기
print("is_available:", torch.cuda.is_available())
#print("nvml available:", torch.cuda.nccl.is_available() if hasattr(torch.cuda,"nccl") else "n/a")
try:
    print("GPU name:", torch.cuda.get_device_name(0))
except Exception as e:
    print("get_device_name error:", e)
try:
    import cupy as cp
    print("cupy:", cp.__version__)
    print("cupy runtime cuda ver:", cp.cuda.runtime.runtimeGetVersion())
    cp.cuda.runtime.getDeviceCount()  # 오류 시 예외
    print("cupy device ok")
except Exception as e:
    print("cupy error:", e)