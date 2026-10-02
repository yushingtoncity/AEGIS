import json
import sys
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"
EXECUTION = "aegis.execution"


@pytest.fixture(autouse=True)
def plain_cli_output(monkeypatch):
    """CLI tests compare text. Python 3.14's argparse colours ``--help`` when
    the terminal asks for it (``FORCE_COLOR``, ``PYTHON_COLORS``, a tty), and
    no assertion expects ANSI codes — so every test runs with colour off,
    whatever the shell that launched pytest exports."""
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.delenv("FORCE_COLOR", raising=False)
    monkeypatch.delenv("PYTHON_COLORS", raising=False)


@pytest.fixture
def fixture():
    """Load a canned JSON fixture from tests/fixtures by filename."""

    def load(name: str):
        return json.loads((FIXTURES / name).read_text(encoding="utf-8"))

    return load


class _RefuseExecution:
    """A ``sys.meta_path`` finder that refuses to import ``aegis.execution``."""

    def find_spec(self, name, path=None, target=None):
        if name == EXECUTION or name.startswith(EXECUTION + "."):
            raise ImportError(f"the brain tried to import {name}")
        return None


@pytest.fixture(autouse=True)
def execution_is_unimportable(request, monkeypatch):
    """In every ``test_brain_*`` module the brain has no execution authority:
    nothing a test drives — a cycle, a stage, the live snapshot builder, the
    CLI — may load ``aegis.execution``, lazily or by a computed name (the
    static rules and the fresh-interpreter backstop live in
    ``tests/test_brain_architecture.py``). Any copy already in
    ``sys.modules`` is hidden for the test so a cached import cannot slip by.
    Other test modules are left alone."""
    if not request.path.name.startswith("test_brain_"):
        return
    for name in [m for m in sys.modules if m == EXECUTION or m.startswith(EXECUTION + ".")]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setattr(sys, "meta_path", [_RefuseExecution(), *sys.meta_path])
