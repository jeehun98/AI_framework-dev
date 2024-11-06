import sys
# 경로를 절대 경로로 변환하여 추가
sys.path.insert(0, 'C:/Users/owner/Desktop/AI_framework-dev')

from dev.models.OPL import OPL
from dev.layers.core.input_layer import InputLayer

inputs = InputLayer