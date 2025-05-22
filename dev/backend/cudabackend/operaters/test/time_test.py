import numpy as np
import torch
import time
import cupy as cp

# Test size
rows, cols = 2048, 2048

# Generate data
np_a = np.random.rand(rows, cols).astype(np.float32)
np_b = np.random.rand(rows, cols).astype(np.float32)

# NumPy to CUDA (simulate cudaMemcpyHostToDevice)
start_np_to_cuda = time.time()
d_a = cp.asarray(np_a)  # implicitly allocates and copies
d_b = cp.asarray(np_b)
cp.cuda.Device().synchronize()
end_np_to_cuda = time.time()

# CUDA computation (using cupy)
start_cuda_compute = time.time()
d_c = d_a + d_b  # simulate kernel
cp.cuda.Device().synchronize()
end_cuda_compute = time.time()

# CUDA to NumPy (simulate cudaMemcpyDeviceToHost)
start_cuda_to_np = time.time()
np_c = cp.asnumpy(d_c)
end_cuda_to_np = time.time()

import pandas as pd

benchmark_df = pd.DataFrame({
    "Stage": [
        "NumPy to CUDA (H2D)",
        "CUDA compute (kernel)",
        "CUDA to NumPy (D2H)"
    ],
    "Time (seconds)": [
        end_np_to_cuda - start_np_to_cuda,
        end_cuda_compute - start_cuda_compute,
        end_cuda_to_np - start_cuda_to_np
    ]
})

print(benchmark_df)