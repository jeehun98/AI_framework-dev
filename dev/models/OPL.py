import numpy as np

class OPL():
    def __new__(cls, *args, **kwargs):
        pass

    def __init__(self):
        # 각 operation 에 대한 가중치가 생성되었는지 확인하기 위한 built
        self.built = False

        # operation 들의 리스트
        self.operations = []

        # 루트 노드들의 리스트
        self.node_list = []

        # 비용 함수의 계산 그래프
        self.loss_node_list = []

    