# dev/smoke_compiler.py
import os, sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from utils.load_cuda import ensure_cuda_dlls
ensure_cuda_dlls()

from backend.compiler.ir.nodes import Tensor, Op, Graph
from backend.compiler import ExecutorV2


X = Tensor("X",(256,512),"f16"); W = Tensor("W",(512,512),"f16")
B = Tensor("B",(512,),  "f16"); Z = Tensor("Z",(256,512),"f16")
Z2= Tensor("Z2",(256,512),"f16"); Z3= Tensor("Z3",(256,512),"f16")

g = Graph([
  Op("MATMUL",[X,W],[Z], attrs={"mnk":(256,512,512)}),
  Op("BIAS_ADD",[Z,B],[Z2]),
  Op("RELU",[Z2],[Z3]),
])

ex = ExecutorV2(dry_run=False)
ex.run(g)
print("OK")
