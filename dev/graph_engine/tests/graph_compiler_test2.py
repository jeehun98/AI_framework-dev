import sys
import os

# ✅ 프로젝트 루트 경로 추가

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.insert(0, PROJECT_ROOT)

import cupy as cp
from dev.models.sequential import Sequential
from dev.layers.dense import Dense
from dev.layers.activation_layer import Activation
from dev.graph_engine.graph_compiler import GraphCompiler

def test_graph_compiler():
    x = cp.random.rand(1, 5).astype(cp.float32)

    model = Sequential()
    model.add(Dense(units=8, input_shape=(1, 5)))
    model.add(Activation("relu"))
    model.add(Dense(units=4))
    model.add(Activation("sigmoid"))
    
    model.build((1, 5))

    compiler = GraphCompiler()
    for layer in model._layers:
        compiler.add_layer(layer)

    compiled = compiler.get_compiled()

    print("📊 E 행렬:")
    for row in compiled["E"]:
        print(row)

    print("📐 W_shapes:", [w.shape for w in compiled["W"]])
    print("📐 b_shapes:", [b.shape for b in compiled["b"]])
    print("🔢 input_node:", compiled["input_node"])
    print("🔢 output_node:", compiled["output_node"])

if __name__ == "__main__":
    test_graph_compiler()
