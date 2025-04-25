from dev.graph_engine.losses_graph import build_mse_node, build_binary_crossentropy_node, build_categorical_crossentropy_node

def mse_graph():
    return build_mse_node()

def binary_crossentropy_graph():
    return build_binary_crossentropy_node()

def categorical_crossentropy_graph():
    return build_categorical_crossentropy_node()
