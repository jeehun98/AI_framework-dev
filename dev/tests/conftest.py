# dev/tests/conftest.py
import os, sys
sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))
from utils.load_cuda import ensure_cuda_dlls
ensure_cuda_dlls()
