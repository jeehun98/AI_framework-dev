# graph_executor_v2/optim/sgd.py
from __future__ import annotations
from typing import Any, Dict
from .base import Optimizer

class SGD(Optimizer):
    """
    옵션:
      - lr, momentum=0.0, nesterov=False
      - weight_decay: L2 (coupled) 또는 decoupled=True 시 AdamW식
      - decoupled: True면 p -= lr*wd*p 를 'step별'로 분리 적용(권장)
    """
    def __init__(self, params, lr: float=1e-2, momentum: float=0.0,
                 nesterov: bool=False, weight_decay: float=0.0, decoupled: bool=True):
        super().__init__(params, lr=lr, momentum=momentum,
                         nesterov=nesterov, weight_decay=weight_decay, decoupled=decoupled)

    def step(self):
        for group in self.param_groups:
            lr = group["lr"]; mom = group["momentum"]
            nesterov = group["nesterov"]; wd = group["weight_decay"]; dec = group["decoupled"]
            for p in group["params"]:
                g = getattr(p, "grad", None)
                if g is None: continue

                # decoupled weight decay (AdamW-style) 권장
                if wd and dec:
                    p[...] = p - lr * wd * p

                sid = id(p)
                st: Dict[str, Any] = self.state.setdefault(sid, {})
                if mom > 0:
                    v = st.get("v")
                    if v is None:
                        # v = zeros_like(p)
                        try:
                            v = p * 0  # ndarray-like
                        except Exception:
                            import numpy as np
                            v = np.zeros_like(p)
                    v[...] = mom * v + g
                    if nesterov:
                        update = g + mom * v
                    else:
                        update = v
                    st["v"] = v
                else:
                    update = g

                # coupled L2
                if wd and not dec:
                    update = update + wd * p

                p[...] = p - lr * update
