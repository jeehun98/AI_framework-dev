from dev.graph_engine.activations_graph import build_relu_node, build_sigmoid_node, build_tanh_node

def relu_graph():
    return build_relu_node()

def sigmoid_graph():
    return build_sigmoid_node()

def tanh_graph():
    return build_tanh_node()
