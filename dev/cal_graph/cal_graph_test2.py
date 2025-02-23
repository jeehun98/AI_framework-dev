import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from dev.cal_graph.cal_graph import Cal_graph

# 예제 1: 올바른 계산 그래프 출력
cal_graph = Cal_graph()
nodes1 = cal_graph.matrix_add([[1, 2], [3, 4]], [[5, 6], [7, 8]])
cal_graph.print_graph()

print("------------------------------------")
cal_graph2 = Cal_graph()
nodes2 = cal_graph2.matrix_multiply([[1, 2], [3, 4]], [[5, 6], [7, 8]])
cal_graph2.print_graph()