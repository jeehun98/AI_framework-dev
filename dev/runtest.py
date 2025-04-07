import os
import sys
import subprocess

# 프로젝트 루트 설정 및 PYTHONPATH 적용
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.environ["PYTHONPATH"] = project_root
sys.path.insert(0, project_root)

# pytest 실행
subprocess.run([
    sys.executable, "-m", "pytest",
    "tests", "dev",                             # 주요 테스트 디렉토리
    "--rootdir", project_root,
    "--ignore-glob=other_frameworks/**/*",
    "--ignore-glob=dev/other_frameworks/**/*",
    "-v"
])
