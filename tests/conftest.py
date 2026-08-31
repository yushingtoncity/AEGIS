import json
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixture():
    """Load a canned JSON fixture from tests/fixtures by filename."""

    def load(name: str):
        return json.loads((FIXTURES / name).read_text(encoding="utf-8"))

    return load
