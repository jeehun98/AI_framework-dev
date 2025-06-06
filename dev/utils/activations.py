import cupy as cp

def sigmoid(x):
    return 1 / (1 + cp.exp(-x))

def relu(x):
    return cp.maximum(0, x)

def tanh(x):
    return cp.tanh(x)
