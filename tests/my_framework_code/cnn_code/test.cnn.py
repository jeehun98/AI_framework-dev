import sys
# 경로를 절대 경로로 변환하여 추가
sys.path.insert(0, 'C:/Users/owner/Desktop/AI_framework-dev')

import numpy as np
from dev.models.sequential import Sequential
from dev.layers.core.dense import Dense
from dev.layers.core.Conv2D import Conv2D

model = Sequential()

model.add(Conv2D(32, (3,3)))

print("완료")