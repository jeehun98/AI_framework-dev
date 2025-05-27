def load_activations_cuda():
    import os
    import sys
    cuda_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin"
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(cuda_path)
    pyd_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend", "backend_ops", "activations", "build", "lib.win-amd64-cpython-312"))
    if pyd_path not in sys.path:
        sys.path.insert(0, pyd_path)
    import activations_cuda
    return activations_cuda
