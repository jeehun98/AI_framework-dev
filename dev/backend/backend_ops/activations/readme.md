### activation 연산 수행 부분 담당

backend, cuda 에서 activation 의 연산에 해당하는, result 를 얻을 수 있음

해당 결과와 결합하여 각 activation function 의 계산 그래프 구조를 정의해야 함

이는 cal_graph 에서의 activation 에서 구현해야 할 듯? 

실제 backend 연산과 계산 그래프 구성의 차이이