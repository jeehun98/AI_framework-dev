import typing
import json
import numpy as np

from dev.layers.layer import Layer
from dev import optimizers
from dev import losses
from dev import metrics
from dev.node.node import Node
from dev.graph_engine.core_graph import Cal_graph


class Sequential(Node):
    def __new__(cls, *args, **kwargs):
        return typing.cast(cls, super().__new__(cls))

    def __init__(self, layers=None, trainable=True, name=None):
        self.built = False
        self._layers = []
        self.cal_graph = Cal_graph()           # ✅ 전체 계산 그래프 인스턴스
        self.loss_node_list = []

    def get_config(self):
        sequential_config = {'name': 'sequential'}
        layer_configs = [layer.get_config() for layer in self._layers]
        return {**sequential_config, "layers": layer_configs}

    def add(self, layer):
        if not isinstance(layer, Layer):
            raise ValueError("Only instances of Layer can be added.")

        if self._layers:
            previous_layer = self._layers[-1]
            input_shape = previous_layer.output_shape

            # ✅ 이전 레이어 output_shape가 없을 수도 있으므로 fallback 필요
            if input_shape is not None:
                layer.build(input_shape)
            elif hasattr(layer, "input_shape") and layer.input_shape is not None:
                layer.build(layer.input_shape)
            else:
                raise RuntimeError(f"레이어 {layer.layer_name}의 입력 shape을 추론할 수 없습니다.")
        else:
            # ✅ 첫 번째 레이어는 반드시 input_shape가 있어야 함
            if hasattr(layer, "input_shape") and layer.input_shape is not None:
                layer.build(layer.input_shape)
            else:
                raise RuntimeError("첫 번째 레이어는 input_shape를 지정해야 합니다.")

        self._layers.append(layer)


    def build(self):
        self.input_shape = self._layers[0].input_shape

    def get_build_config(self):
        return {"input_shape": self.input_shape}

    def compile(self, optimizer=None, loss=None, p_metrics=None, learning_rate=0.001):
        self.optimizer = optimizers.get(optimizer, learning_rate=learning_rate)
        self.loss = losses.get(loss)
        self.metric = metrics.get(p_metrics)
        self.build()
        self.get_weight()

    def get_compile_config(self):
        return {
            "optimizer": self.optimizer.get_config(),
            "loss": self.loss.get_config(),
            "metrics": self.metric.get_config(),
        }

    def get_weight(self):
        self.weights = [layer.weights for layer in self._layers if hasattr(layer, 'weights')]

    def serialize_model(self):
        compile_config = self.get_compile_config()
        model_config = self.get_config()
        build_config = self.get_build_config()

        weights = []
        for layer in self._layers:
            layer_weights = layer.get_weights()
            serialized_weights = [w.tolist() for w in layer_weights]
            weights.append(serialized_weights)

        model_data = {
            "compile_config": compile_config,
            "model_config": model_config,
            "build_config": build_config,
            "weights": weights,
        }

        return json.dumps(model_data)


    def predict(self, data):
        output = data
        for layer in self._layers:
            output = layer.call(output)
        return output

    def compute_loss_and_metrics(self, y_pred, y_true):
        self.loss_value = self.loss(y_true, y_pred)  # ✅ 호출 순서 주의
        self.loss_node_list = []                     # ✅ CUDA 손실은 node_list 없음
        self.metric_value = self.metric(y_pred, y_true)
        return self.loss_value
        

    def fit(self, x=None, y=None, epochs=1, batch_size=-1):
        if batch_size == -1 or batch_size < x.shape[0]:
            batch_size = x.shape[0]
        batch_counts = int(np.ceil(x.shape[0] / batch_size))

        for epoch in range(epochs):
            print(f"\n=== [Epoch {epoch + 1}] 시작 ===")

            indices = np.random.permutation(x.shape[0])
            x = x[indices]
            y = y[indices]

            for batch_idx in range(batch_counts):
                print(f"\n[Batch {batch_idx + 1}] 처리 시작")

                start = batch_idx * batch_size
                end = min(start + batch_size, x.shape[0])
                batch_x = x[start:end]
                batch_y = y[start:end]
                batch_datas = batch_x.shape[0]
                batch_loss_sum = None

                for batch_data_idx in range(batch_datas):
                    print(f" [Sample {batch_data_idx + 1}] 전방 계산 시작")

                    input_data = batch_x[batch_data_idx]
                    target = batch_y[batch_data_idx]
                    output = input_data

                    for idx, layer in enumerate(self._layers):
                        print(f"   - Layer {idx}: {layer.__class__.__name__} 호출 전")
                        output = layer.call(output)
                        print(f"   - Layer {idx}: call 완료")

                        # 그래프 연결
                        if idx == 0:
                            self.cal_graph.node_list = layer.node_list[:]
                        else:
                            print(f"   - [DEBUG] 이전 노드 수: {len(self.cal_graph.node_list)}, 현재 레이어 노드 수: {len(layer.node_list)}")

                            if self.cal_graph.node_list and layer.node_list:
                                print(f"   - 그래프 연결 시작: {self.cal_graph.node_list[-1]} → {layer.node_list[-1]}")
                                self.cal_graph.node_list = self.cal_graph.connect_graphs(
                                    self.cal_graph.node_list, layer.node_list
                                )
                                print(f"   - 그래프 연결 완료")
                            else:
                                print(f"⚠️ 그래프 연결 생략 - 연결할 노드가 없습니다.")

                    output = np.array(output).reshape(1, -1)
                    target = np.array(target).reshape(1, -1)

                    print(f" [Sample {batch_data_idx + 1}] 손실 계산 시작")
                    loss_value = self.compute_loss_and_metrics(output, target)
                    print(f" [Sample {batch_data_idx + 1}] 손실 계산 완료")

                    if self.loss_node_list:
                        self.cal_graph.node_list = self.cal_graph.connect_graphs(
                            self.cal_graph.node_list, self.loss_node_list
                        )
                        print(f" [Sample {batch_data_idx + 1}] 손실 그래프 연결 완료")
                    else:
                        print(f"⚠️ 손실 노드가 없습니다. 연결 생략")

                    if isinstance(loss_value, list):
                        if batch_loss_sum is None:
                            batch_loss_sum = [0] * len(loss_value)
                        for i in range(len(loss_value)):
                            batch_loss_sum[i] += loss_value[i]
                    else:
                        if batch_loss_sum is None:
                            batch_loss_sum = 0
                        batch_loss_sum += loss_value

                print(f"[Batch {batch_idx + 1}] 역전파 및 가중치 업데이트 시작")

                if isinstance(batch_loss_sum, list):
                    batch_loss = [loss / batch_datas for loss in batch_loss_sum]
                    for i, node in enumerate(self.cal_graph.node_list):
                        node.update(node.input_value, node.weight_value, batch_loss[i], node.bias)
                else:
                    batch_loss = batch_loss_sum / batch_datas
                    self.cal_graph.node_list[0].update(
                        self.cal_graph.node_list[0].input_value,
                        self.cal_graph.node_list[0].weight_value,
                        batch_loss,
                        self.cal_graph.node_list[0].bias,
                    )

                print(f"[Batch {batch_idx + 1}] 손실: {batch_loss}")

                for root_node in self.cal_graph.node_list:
                    print("[Backpropagation] 시작")
                    self.backpropagate(root_node)
                    print("[Backpropagation] 완료")

                for root_node in self.cal_graph.node_list:
                    self.weight_update(root_node, batch_datas, self.optimizer)

            # Epoch 마무리 평가
            loss_sum = 0
            for data_idx in range(x.shape[0]):
                input_data = x[data_idx]
                target = y[data_idx]
                predict = self.predict(input_data)
                predict = np.array(predict).reshape(1, -1)
                data_loss = self.compute_loss_and_metrics(predict, target.reshape(1, -1))
                loss_sum += data_loss

            print(f"\n📊 [Epoch {epoch + 1}] 평균 손실: {loss_sum / x.shape[0]}")
