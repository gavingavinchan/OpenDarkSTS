from pathlib import Path
from tempfile import TemporaryDirectory

import pytest


@pytest.fixture
def tmp_path():
    # Managed runs must stay under workspace/runs, including subprocess CLI fixtures.
    root = Path(__file__).resolve().parent.parent / "workspace" / "runs"
    root.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="pytest_", dir=root) as directory:
        yield Path(directory)
