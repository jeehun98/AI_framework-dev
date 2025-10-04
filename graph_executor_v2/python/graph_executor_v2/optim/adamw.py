# graph_executor_v2/optim/adamw.py
from __future__ import annotations
from typing import Any, Dict
from .base import Optimizer
import math

class AdamW(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999),
                 eps=1e-8, weight_decay=0.01):
        super().__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

    def step(self):
        for g in self.param_groups:
            lr = g["lr"]; (b1, b2) = g["betas"]; eps = g["eps"]; wd = g["weight_decay"]
            for p in g["params"]:
                grad = getattr(p, "grad", None)
                if grad is None: continue

                # decoupled decay
                if wd:
                    p[...] = p - lr * wd * p

                sid = id(p)
                st: Dict[str, Any] = self.state.setdefault(sid, {})
                m = st.get("m"); v = st.get("v"); t = st.get("t", 0) + 1
                if m is None:
                    try:
                        m = p * 0; v = p * 0
                    except Exception:
                        import numpy as np
                        m = np.zeros_like(p); v = np.zeros_like(p)
                m[...] = b1 * m + (1 - b1) * grad
                v[...] = b2 * v + (1 - b2) * (grad * grad)

                mhat = m / (1 - b1 ** t)
                vhat = v / (1 - b2 ** t)
                p[...] = p - lr * (mhat / (vhat**0.5 + eps))

                st["m"] = m; st["v"] = v; st["t"] = t
