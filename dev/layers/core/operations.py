import os
os.add_dll_directory("C:\\msys64\\mingw64\\bin")

from dev.models.OPL import OPL

# 각 연산 별 동일한 구조가 있을 것이기에 상위 클래스, OPERATIONS 을 구현

# 하위 클래스에서 실행되는 동일한 함수의 구현
class OPERATIONS(OPL):
    
    def __init__(self):
        pass