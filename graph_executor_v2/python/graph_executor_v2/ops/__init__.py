from importlib import import_module

# 코어 기초 타입/엔진을 노출
from ._core import Tensor, Graph, Executor, get_version

# 필요한 순간에만 모듈 로드
def require(op_name: str):
    mod_name = f"graph_executor_v2.ops._ops_{op_name}"
    return import_module(mod_name)

# 사용 예:
# ge2.require("gemm");  # import graph_executor_v2.ops._ops_gemm
