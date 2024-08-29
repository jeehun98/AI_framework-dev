import sys
import os

# 현재 스크립트 파일의 디렉토리를 가져옵니다.
current_dir = os.path.dirname(os.path.abspath(__file__))

# dev 디렉토리의 부모 디렉토리를 가져옵니다.
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))

# sys.path에 dev 디렉토리의 부모 디렉토리를 추가합니다.
sys.path.append(parent_dir)

# 이제 dev.models 모듈을 가져올 수 있습니다.
from dev.models.sequential import Sequential

model = Sequential()

print(model.__class__)



