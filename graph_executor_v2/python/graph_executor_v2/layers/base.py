# python/graph_executor_v2/layers/base.py
from __future__ import annotations
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple, Union
import cupy as cp


class Layer:
    """
    Lightweight base class for CuPy-backed layers with optional CUDA Graph capture hooks.

    Design goals:
      - Subclasses may override: build, call, backward, forward_into, backward_into,
        compute_output_shape, parameters/param_triples, grads, zero_grad, state_dict/load_state_dict.
      - Lazy build on first __call__ (configurable).
      - Training/Inference switch via train()/eval().
      - Two parameter access styles are supported:
          * tuple style:  param_triples() -> (param, grad, name)
          * list  style:  parameters()    -> [param, ...]
        By default, parameters() is derived from param_triples() for convenience.
    """

    def __init__(self, *, name: Optional[str] = None, build_on_call: bool = True) -> None:
        self.name: Optional[str] = name
        self.built: bool = False
        self.training: bool = True
        self.build_on_call: bool = bool(build_on_call)

        # Shapes (None-friendly for dynamic batch)
        self.input_shape: Optional[Tuple[int, ...]] = None
        self.output_shape: Optional[Tuple[Optional[int], ...]] = None

    # ---------- lifecycle ----------
    def build(self, input_shape: Tuple[int, ...]) -> None:
        """
        Subclasses should allocate weights/buffers here and set output_shape.
        Base impl sets flags and stores input_shape; output_shape is inferred
        via compute_output_shape if the subclass provides it.
        """
        self.input_shape = tuple(int(x) if x is not None else None for x in input_shape)  # type: ignore
        # Try infer output_shape (safe default = input_shape)
        try:
            self.output_shape = self.compute_output_shape(input_shape)
        except NotImplementedError:
            self.output_shape = input_shape
        self.built = True

    def compute_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[Optional[int], ...]:
        """
        Optional: return the output shape given an input shape.
        Default: identity (same shape).
        """
        if input_shape is None:
            raise ValueError("input_shape must not be None")
        return tuple(int(x) if x is not None else None for x in input_shape)  # type: ignore

    # ---------- forward/backward ----------
    def call(self, x: cp.ndarray) -> cp.ndarray:
        """
        Subclasses must implement the actual forward logic.
        """
        raise NotImplementedError(f"{self.__class__.__name__}.call is not implemented")

    def backward(self, grad_output: cp.ndarray) -> cp.ndarray:
        """
        Optional eager-mode backward. Subclasses implement if needed.
        """
        raise NotImplementedError(f"{self.__class__.__name__}.backward is not implemented")

    def __call__(self, x: cp.ndarray) -> cp.ndarray:
        """
        Lazy-build on first call when enabled.
        """
        if (not self.built) and self.build_on_call:
            if not hasattr(x, "shape"):
                raise ValueError("__call__ received an input without 'shape'")
            # Best-effort: accept any length tuple
            self.build(tuple(int(s) for s in x.shape))
        return self.call(x)

    # ---------- capture-safe (NO-alloc) hooks ----------
    def forward_into(self, x: cp.ndarray, out: cp.ndarray, **_kw: Any) -> None:
        """
        CUDA Graph capture path (no allocations). Subclasses may override.
        """
        raise NotImplementedError(f"{self.__class__.__name__}.forward_into is not implemented")

    def backward_into(self, grad_output: cp.ndarray, **_kw: Any) -> None:
        """
        CUDA Graph capture path (no allocations). Subclasses may override.
        """
        raise NotImplementedError(f"{self.__class__.__name__}.backward_into is not implemented")

    # ---------- params / grads ----------
    def param_triples(self) -> Iterable[Tuple[cp.ndarray, Optional[cp.ndarray], str]]:
        """
        Default heuristic: look for common (param, grad) name pairs on the instance.

        Subclasses are encouraged to override for explicit control (see Conv2D).
        """
        # Common name patterns (ordered by typical usage)
        candidates: List[Tuple[str, str]] = [
            ("W", "dW"), ("b", "db"),
            ("weight", "grad_weight"), ("bias", "grad_bias"),
            ("gamma", "d_gamma"), ("beta", "d_beta"),
        ]
        yielded: set[str] = set()
        for p_name, g_name in candidates:
            if hasattr(self, p_name):
                p = getattr(self, p_name)
                if isinstance(p, cp.ndarray):
                    g = getattr(self, g_name, None)
                    if g is not None and not isinstance(g, cp.ndarray):
                        g = None
                    full = f"{self.name or self.__class__.__name__}.{p_name}"
                    yielded.add(p_name)
                    yield (p, g, full)

        # Fallback: emit any remaining cp.ndarray attributes not yet yielded, grad=None
        for attr, val in self.__dict__.items():
            if attr in yielded:
                continue
            if isinstance(val, cp.ndarray):
                full = f"{self.name or self.__class__.__name__}.{attr}"
                yield (val, None, full)

    def parameters(self) -> List[cp.ndarray]:
        """
        Flat parameter list (derived from param_triples by default).
        Dense overrides this to return [W, b].
        """
        return [p for (p, _g, _n) in self.param_triples()]

    def grads(self) -> List[Optional[cp.ndarray]]:
        """
        Flat grad list aligned to parameters().
        """
        return [g for (_p, g, _n) in self.param_triples()]

    def zero_grad(self) -> None:
        """
        Zero out known grads in-place if they exist and are CuPy arrays.
        """
        for (_p, g, _n) in self.param_triples():
            if isinstance(g, cp.ndarray):
                g[...] = 0

    # ---------- training mode ----------
    def train(self, mode: bool = True) -> "Layer":
        self.training = bool(mode)
        return self

    def eval(self) -> "Layer":
        self.training = False
        return self

    # ---------- serialization ----------
    def state_dict(self) -> Dict[str, Any]:
        """
        Default: pack all param_triples into a dict by their short names.
        Subclasses may override to include hyper-parameters (see Conv2D).
        """
        sd: Dict[str, Any] = {}
        for (_p, _g, full_name) in self.param_triples():
            # keep short tail after last dot as key (e.g., "W", "b")
            key = full_name.split(".")[-1]
            sd[key] = getattr(self, key, None)
        # Store minimal meta
        sd["_meta"] = {
            "name": self.name,
            "training": self.training,
            "input_shape": self.input_shape,
            "output_shape": self.output_shape,
            "class": self.__class__.__name__,
        }
        return sd

    def load_state_dict(self, sd: Dict[str, Any]) -> "Layer":
        """
        Default: assign tensors by key if shapes match (copy in-place when possible).
        Subclasses may override to restore hyper-params.
        """
        for k, v in sd.items():
            if k.startswith("_"):
                continue
            if v is None:
                continue
            if hasattr(self, k):
                cur = getattr(self, k)
                if isinstance(cur, cp.ndarray) and isinstance(v, cp.ndarray):
                    if cur.shape == v.shape and cur.dtype == v.dtype:
                        cur[...] = v
                    else:
                        setattr(self, k, v.copy())
                elif cur is None and isinstance(v, cp.ndarray):
                    setattr(self, k, v.copy())
                else:
                    setattr(self, k, v)
        # optional meta
        meta = sd.get("_meta", {})
        self.name = meta.get("name", self.name)
        # grads must be (re)created by subclass if needed
        return self

    # ---------- utils ----------
    def __repr__(self) -> str:
        cls = self.__class__.__name__
        nm = f" name={self.name!r}" if self.name is not None else ""
        st = "train" if self.training else "eval"
        shp = f" in={self.input_shape}, out={self.output_shape}" if (self.input_shape or self.output_shape) else ""
        return f"<{cls}{nm} built={self.built} mode={st}{shp}>"
