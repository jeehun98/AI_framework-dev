# ✅ conftest.py 내 임포트를 절대 경로가 아닌 상대경로로 수정
# dev/tests/conftest.py

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from dev.tests.test_setup import setup_paths

import pytest

@pytest.fixture(scope="session", autouse=True)
def global_test_setup():
    setup_paths()

def pytest_ignore_collect(path, config):
    return "other_frameworks" in str(path)

def ensure_project_root_in_sys_path(target_dir="AI_framework-dev"):
    current = os.path.abspath(__file__)
    while True:
        current = os.path.dirname(current)
        if os.path.basename(current) == target_dir:
            if current not in sys.path:
                sys.path.insert(0, current)
            return current
        if current == os.path.dirname(current):
            raise RuntimeError(f"프로젝트 루트 '{target_dir}'를 찾을 수 없습니다.")

@pytest.fixture(scope="session", autouse=True)
def global_test_setup():
    ensure_project_root_in_sys_path()
    from dev.tests.test_setup import setup_paths
    setup_paths()

def pytest_ignore_collect(path, config):
    return "other_frameworks" in str(path)
