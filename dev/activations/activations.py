# backend 변환
import os
os.add_dll_directory("C:\\msys64\\mingw64\\bin")

from dev.backend.activations import activations

def relu(x, node_list = []):
    return activations.relu(x, node_list)

def leaky_relu(x, alpha=0.01, node_list = []):
    return activations.leaky_relu(x, alpha, node_list)

def sigmoid(x, node_list = []):
    return activations.sigmoid(x, node_list)

def tanh(x, node_list = []):
    return activations.tanh_activation(x, node_list)

def softmax(x, node_list = []):
    return activations.softmax(x, node_list)