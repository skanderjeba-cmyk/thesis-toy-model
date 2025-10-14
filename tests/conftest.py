# tests/conftest.py
import os, sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Optional: keep Matplotlib headless if any plotting code gets imported
os.environ.setdefault("MPLBACKEND", "Agg")

# Optional: ensure tests run with repo root as CWD (helps with 'params.yaml' etc.)
@pytest.fixture(autouse=True, scope="session")
def _cd_repo_root():
    prev = os.getcwd()
    os.chdir(str(ROOT))
    try:
        yield
    finally:
        os.chdir(prev)
