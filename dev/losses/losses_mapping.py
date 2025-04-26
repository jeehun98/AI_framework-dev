from dev.graph_engine.losses_graph import build_mse_node, build_binary_crossentropy_node, build_categorical_crossentropy_node

# ✅ MSE 그래프 빌더 (반드시 인자 받도록 수정)
def mse_graph(num_outputs):
    return build_mse_node(num_outputs)


# ✅ Binary Crossentropy 그래프 빌더
def binary_crossentropy_graph(num_outputs):
    return build_binary_crossentropy_node(num_outputs)


# ✅ Categorical Crossentropy 그래프 빌더
def categorical_crossentropy_graph(num_classes):
    return build_categorical_crossentropy_node(num_classes)
