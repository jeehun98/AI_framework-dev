# backend 변환
import os
os.add_dll_directory("C:\\msys64\\mingw64\\bin")

from dev.backend.activations import activations

def relu(x):
    return activations.relu(x)

def leaky_relu(x, alpha=0.01):
    return activations.leaky_relu(x, alpha)

def sigmoid(x):
    return activations.sigmoid(x)

def tanh(x):
    return activations.tanh_activation(x)

def softmax(x):
    return activations.softmax(x)