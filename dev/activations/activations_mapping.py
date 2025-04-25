from dev.graph_engine.activations_graph import build_relu_node, build_sigmoid_node, build_tanh_node

def relu_graph(input_node):
    return build_relu_node(input_node)

def sigmoid_graph(input_node):
    return build_sigmoid_node(input_node)

def tanh_graph(input_node):
    return build_tanh_node(input_node)
