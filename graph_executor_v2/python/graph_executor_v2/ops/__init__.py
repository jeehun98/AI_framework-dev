# python/graph_executor_v2/ops/__init__.py
def require(op: str):
    import importlib
    return importlib.import_module(f"graph_executor_v2.ops._ops_{op}")
