# python/graph_executor_v2/layers/activations.py
from __future__ import annotations
import cupy as cp

def apply_activation(z, act: str | None):
    if act is None or act == "none":
        return z
    a = act.lower()
    if a == "relu":
        return cp.maximum(z, 0)
    if a == "sigmoid":
        return 1 / (1 + cp.exp(-z))
    if a == "tanh":
        return cp.tanh(z)
    if a == "gelu":
        c = cp.sqrt(2.0 / cp.pi)
        return 0.5 * z * (1 + cp.tanh(c * (z + 0.044715 * z**3)))
    if a in ("leakyrelu", "leaky_relu", "lrelu"):
        return cp.where(z > 0, z, 0.01 * z)  # slope 기본값 0.01
    raise ValueError(f"Unknown activation: {act}")

def apply_activation_grad(grad_output, z, act: str | None, leaky_slope: float = 0.01):
    if act is None or act == "none":
        return grad_output
    a = act.lower()
    if a == "relu":
        return grad_output * (z > 0)
    if a == "sigmoid":
        sig = 1 / (1 + cp.exp(-z))
        return grad_output * sig * (1 - sig)
    if a == "tanh":
        t = cp.tanh(z)
        return grad_output * (1 - t * t)
    if a == "gelu":
        c = cp.sqrt(2.0 / cp.pi)
        t = cp.tanh(c * (z + 0.044715 * z**3))
        dt = (1 - t**2) * c * (1 + 3 * 0.044715 * z**2)
        gelu_grad = 0.5 * (1 + t) + 0.5 * z * dt
        return grad_output * gelu_grad
    if a in ("leakyrelu", "leaky_relu", "lrelu"):
        return grad_output * cp.where(z > 0, 1.0, leaky_slope)
    raise ValueError(f"Unknown activation grad: {act}")
