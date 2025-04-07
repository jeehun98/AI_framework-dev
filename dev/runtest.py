# dev/runtests.py
import os
import subprocess

# PYTHONPATH 설정
os.environ["PYTHONPATH"] = "."

# 명시적으로 제외 옵션 추가
subprocess.run([
    "pytest",
    "-v",
    "--ignore-glob=other_frameworks/**/*"
])
