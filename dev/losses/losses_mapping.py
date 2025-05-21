from dev.graph_engine.losses_graph import (
    build_mse_node,
    build_binary_crossentropy_node,
    build_categorical_crossentropy_node,
)

# ✅ MSE 손실 함수의 계산 그래프 빌더
# - num_outputs: 출력 유닛 수 (예: 3)
# - result: CUDA 연산 결과값 (손실 스칼라 값)
def mse_graph(num_outputs, result):
    return build_mse_node(num_outputs, result)


# ✅ Binary Crossentropy 손실 함수의 계산 그래프 빌더
def binary_crossentropy_graph(num_outputs, result):
    return build_binary_crossentropy_node(num_outputs, result)


# ✅ Categorical Crossentropy 손실 함수의 계산 그래프 빌더
# - num_classes: 분류 클래스 수
def categorical_crossentropy_graph(num_classes, result):
    return build_categorical_crossentropy_node(num_classes, result)
