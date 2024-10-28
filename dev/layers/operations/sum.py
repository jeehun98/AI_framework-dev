import os
os.add_dll_directory("C:\\msys64\\mingw64\\bin")

from dev.layers.core.operations import OPERATIONS

class SUM(OPERATIONS):
    def __init__(self):
        pass

    def call(self):
        # 여기서 cuda 코드를 실행해야 함
        
        pass