import json
import sys
from pathlib import Path

import pytest

import aegis.config as config_module

FIXTURES = Path(__file__).parent / "fixtures"
EXECUTION = "aegis.execution"
ALPACA_KEYS = ("ALPACA_API_KEY", "ALPACA_SECRET_KEY")


@pytest.fixture(autouse=True)
def plain_cli_output(monkeypatch):
    """CLI tests compare text. Python 3.14's argparse colours ``--help`` when
    the terminal asks for it (``FORCE_COLOR``, ``PYTHON_COLORS``, a tty), and
    no assertion expects ANSI codes — so every test runs with colour off,
    whatever the shell that launched pytest exports."""
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.delenv("FORCE_COLOR", raising=False)
    monkeypatch.delenv("PYTHON_COLORS", raising=False)


@pytest.fixture(autouse=True)
def no_live_alpaca(monkeypatch):
    """Tests never call a live API. An empty key in the environment wins over
    .env (``load_dotenv`` does not override), so any client that slips past a
    fake fails closed on ``require_env`` instead of reaching Alpaca. Child
    processes a test starts inherit the same empty keys. Tests that need a key
    set their own."""
    for name in ALPACA_KEYS:
        monkeypatch.setenv(name, "")


@pytest.fixture(autouse=True)
def signals_as_shipped(monkeypatch):
    """The repo config is read with its ``signals`` block as shipped (off), so
    a machine that turned signals on in its local config.yaml runs the same
    cycles and event lists as a clean checkout. ``load_config`` with an
    explicit path, and a copy imported by name before this runs (the
    shipped-defaults test), still read the file as it is."""
    original = config_module.load_config

    def load_config(path=None):
        config = original(path)
        if path is None:
            config = config.model_copy(update={"signals": config_module.SignalsConfig()})
        return config

    config_module.get_config.cache_clear()
    monkeypatch.setattr(config_module, "load_config", load_config)
    yield
    config_module.get_config.cache_clear()


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
