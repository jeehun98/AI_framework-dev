# dev/tests/conftest.py
import pytest
from tests.test_setup import setup_paths

@pytest.fixture(scope="session", autouse=True)
def global_test_setup():
    setup_paths()

def pytest_ignore_collect(path, config):
    # 경로 문자열에 포함된 경우 무시
    return "other_frameworks" in str(path)
