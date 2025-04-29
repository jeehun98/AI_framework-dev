from dev.graph_engine.activations_graph import build_relu_node, build_sigmoid_node, build_tanh_node

def relu_graph(result):
    return build_relu_node(result)

def sigmoid_graph(result):
    return build_sigmoid_node(result)

def tanh_graph(result):
    return build_tanh_node(result)
