# graph_executor_v2/optim/base.py
from __future__ import annotations
from typing import Iterable, Dict, Any, List, Tuple

Param = Any  # ndarray-like or tensor-like

class Optimizer:
    def __init__(self, params: Iterable[Param] | Iterable[Dict[str, Any]], **defaults):
        self.param_groups: List[Dict[str, Any]] = []
        self.state: Dict[int, Dict[str, Any]] = {}
        if params is None:
            raise ValueError("params is required")
        if isinstance(params, dict) or (len(params) > 0 and isinstance(list(params)[0], dict)):  # type: ignore
            # already param groups
            for g in params:  # type: ignore
                group = {**defaults, **g}
                group["params"] = list(group["params"])
                self._validate(group)
                self.param_groups.append(group)
        else:
            group = {**defaults, "params": list(params)}  # type: ignore
            self._validate(group)
            self.param_groups.append(group)

    def _validate(self, group: Dict[str, Any]):  # hook
        if "params" not in group or len(group["params"]) == 0:
            raise ValueError("param group must have non-empty 'params'")

    def zero_grad(self):
        for group in self.param_groups:
            for p in group["params"]:
                g = getattr(p, "grad", None)
                if g is not None:
                    try:
                        g[...] = 0
                    except Exception:
                        if hasattr(g, "zero_"): g.zero_()
                        else: setattr(p, "grad", None)

    def state_dict(self) -> Dict[str, Any]:
        return {"state": self.state, "param_groups": self.param_groups}

    def load_state_dict(self, sd: Dict[str, Any]):
        self.state = sd["state"]
        # param_groups는 객체 참조가 있으므로 하이퍼파라미터만 동기화
        for dst_g, src_g in zip(self.param_groups, sd["param_groups"]):
            for k, v in src_g.items():
                if k == "params": continue
                dst_g[k] = v

    def step(self):
        raise NotImplementedError
