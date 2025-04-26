import typing
import json
import numpy as np

from dev.layers.layer import Layer
from dev.losses import losses_mapping
from dev import optimizers
from dev.backend.backend_ops.losses import losses as cuda_losses
from dev import metrics
from dev.node.node import Node
from dev.graph_engine.core_graph import Cal_graph


class Sequential(Node):
    def __new__(cls, *args, **kwargs):
        return typing.cast(cls, super().__new__(cls))

    def __init__(self, layers=None, trainable=True, name=None):
        self.built = False
        self._layers = []
        self.cal_graph = Cal_graph()
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
            if input_shape is not None:
                layer.build(input_shape)
            elif hasattr(layer, "input_shape") and layer.input_shape is not None:
                layer.build(layer.input_shape)
            else:
                raise RuntimeError(f"레이어 {layer.layer_name}의 입력 shape을 추론할 수 없습니다.")
        else:
            if hasattr(layer, "input_shape") and layer.input_shape is not None:
                layer.build(layer.input_shape)
            else:
                raise RuntimeError("첫 번째 레이어는 input_shape를 지정해야 합니다.")

        print(f"✅ 레이어 추가됨: {layer.__class__.__name__} (input_shape={layer.input_shape}, output_shape={layer.output_shape})")
        self._layers.append(layer)

    def build(self):
        self.input_shape = self._layers[0].input_shape

    def get_build_config(self):
        return {"input_shape": self.input_shape}

    def compile(self, optimizer=None, loss=None, p_metrics=None, learning_rate=0.001):
        self.optimizer = optimizers.get(optimizer, learning_rate=learning_rate)
        self.loss = cuda_losses.get(loss)
        self.loss_name = loss  # 손실 함수 이름 저장
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
        for i, layer in enumerate(self._layers):
            output = layer.call(output)
        return output

    def forward_pass(self, input_data):
        output = input_data
        print(f"[SHAPE TRACE] Input: {output.shape}")
        for i, layer in enumerate(self._layers):
            output = layer.call(output)
            print(f"[SHAPE TRACE] Layer {i}: {layer.__class__.__name__} → output: {output.shape}")
        return output

    def connect_loss_graph(self):
        if self.loss_node_list:
            print(f"[DEBUG] Loss Graph 연결: 출력 유닛 {len(self.loss_leaf_nodes)}개")

            # 출력층과 손실 함수 그래프 연결
            self.cal_graph.root_node_list = self.cal_graph.connect_graphs(
                self.cal_graph.root_node_list, self.loss_leaf_nodes
            )

            # 최종 루트는 손실 함수 루트로 덮어쓰기
            self.cal_graph.root_node_list = self.loss_node_list[:]

    def compute_loss_and_metrics(self, y_pred_array, y_true_array):
        # 1️⃣ CUDA 연산
        self.loss_value = self.loss(y_true_array, y_pred_array)

        print(y_pred_array, "뭐야 이게")

        # 2️⃣ 계산 그래프 생성 (출력 유닛 수 기반)
        num_outputs = y_pred_array.shape[1]  # (1, N) 형태 기준
        try:
            builder = getattr(losses_mapping, f"{self.loss_name}_graph")
        except AttributeError:
            raise NotImplementedError(f"{self.loss_name} 계산 그래프 미지원")

        if self.loss_name == "categorical_crossentropy":
            loss_root, leaf_nodes = builder(num_classes=num_outputs)
        else:
            loss_root, leaf_nodes = builder(num_outputs=num_outputs)

        self.loss_node_list = [loss_root]
        self.loss_leaf_nodes = leaf_nodes

        # 3️⃣ 메트릭 계산
        self.metric_value = self.metric(y_pred_array, y_true_array)

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
                batch_loss_sum = 0

                for data_idx in range(batch_datas):
                    input_data = batch_x[data_idx]
                    target = batch_y[data_idx]

                    print(f"\n[SHAPE TRACE] === Sample {data_idx + 1} ===")
                    print(f"[SHAPE TRACE] Input: {input_data.shape}")

                    output = input_data
                    prev_root_nodes = None

                    for i, layer in enumerate(self._layers):
                        print(f"[DEBUG] Layer {i}: {layer.__class__.__name__} call() 실행")
                        output = layer.call(output)
                        print(f"[SHAPE TRACE] Layer {i}: {layer.__class__.__name__} → output: {output.shape}")

                        if hasattr(layer, "root_node_list") and layer.root_node_list:
                            if prev_root_nodes is None:
                                # 첫 레이어: 단순히 root 설정
                                self.cal_graph.root_node_list = layer.root_node_list[:]
                            else:
                                # ✅ 연결만 수행
                                self.cal_graph.connect_graphs(prev_root_nodes, layer.leaf_node_list)
    
                                # ✅ 루트는 항상 "현재 레이어의 root"로 갱신
                                self.cal_graph.root_node_list = layer.root_node_list[:]

                                """
                                for i in range(len(self.cal_graph.root_node_list)):
                                    self.cal_graph.root_node_list[i].print_tree()
                                    print("각 계산 그래프 확인용")
                                """
                                
                                

                            prev_root_nodes = layer.root_node_list[:]

                    
                    output = np.array(output).reshape(1, -1)
                    target = np.array(target).reshape(1, -1)

                    print("[DEBUG] 손실 및 메트릭 계산 시작")
                    loss_value = self.compute_loss_and_metrics(output, target)
                    print("[DEBUG] 손실 및 메트릭 계산 완료")

                    self.connect_loss_graph()

                    self.cal_graph.print_graph()

                    batch_loss_sum += loss_value

                batch_loss = batch_loss_sum / batch_datas
                print(f"[Batch {batch_idx + 1}] 평균 손실: {batch_loss}")

                print("[DEBUG] 역전파 시작")
                for root_node in self.cal_graph.root_node_list:
                    self.backpropagate(root_node)
                print("[DEBUG] 역전파 완료")

                print("[DEBUG] 가중치 업데이트 시작")
                for root_node in self.cal_graph.root_node_list:
                    self.weight_update(root_node, batch_datas, self.optimizer)
                print("[DEBUG] 가중치 업데이트 완료")

                print("[DEBUG] 계산 그래프 출력:")
                self.cal_graph.print_graph()

            total_loss = 0
            for i in range(x.shape[0]):
                pred = self.predict(x[i])
                pred = np.array(pred).reshape(1, -1)
                loss = self.compute_loss_and_metrics(pred, y[i].reshape(1, -1))
                total_loss += loss

            print(f"\n📊 [Epoch {epoch + 1}] 전체 평균 손실: {total_loss / x.shape[0]}")
