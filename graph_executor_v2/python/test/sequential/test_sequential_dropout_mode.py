import cupy as cp
from graph_executor_v2.layers.sequential import Sequential
from graph_executor_v2.layers.dense_gemm import Dense
from graph_executor_v2.layers.dropout import Dropout

def test_dropout_train_eval_behavior():
    cp.random.seed(2)
    X = cp.random.standard_normal((256, 16), dtype=cp.float32)

    m = Sequential(Dense(16, 16, activation="none", use_native_bwd=True),
                   Dropout(0.5))
    m.build(input_shape=(256, 16))

    m.train(True)
    Y_train = m(X.copy())
    m.eval()
    Y_eval  = m(X.copy())

    # 일반적으로 train 모드에서 분산이 커짐
    assert float(Y_train.std()) > float(Y_eval.std()) * 1.1
