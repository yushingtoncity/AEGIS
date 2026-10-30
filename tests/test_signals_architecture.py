"""Signals are context for the scan stage, never an input to a decision.

Every module on the order path (the policy engine, the executors, the orders
and policy CLIs) is scanned statically for an import of the signals code or
SQL that touches the signal tables, and a fresh interpreter checks that
loading the order path's entry points brings no signals module along. In the
other direction, the signals package may not import the policy engine or an
executor: it reads the outside world and writes its own tables, nothing
else. The checkers are tested on planted sources so a clean scan means
something.
"""

import ast
import re
import subprocess
import sys
from pathlib import Path

import pytest

from aegis.config import REPO_ROOT

AEGIS = REPO_ROOT / "aegis"

ORDER_PATH = sorted(
    [
        *(AEGIS / "policy").rglob("*.py"),
        *(AEGIS / "execution").rglob("*.py"),
        AEGIS / "cli" / "orders.py",
        AEGIS / "cli" / "policy.py",
    ]
)
SIGNALS_CODE = sorted(
    [
        *(AEGIS / "signals").rglob("*.py"),
        AEGIS / "store" / "signals.py",
        AEGIS / "data" / "signals.py",
        AEGIS / "data" / "apewisdom.py",
    ]
)

SIGNAL_MODULES = ("aegis.signals", "aegis.store.signals", "aegis.data.signals", "aegis.data.apewisdom")
SIGNAL_SQL = re.compile(r"\bsignal_fetches\b|\b(from|join|into|update|table)\s+signals\b", re.IGNORECASE)
ORDER_MODULES = ("aegis.policy", "aegis.execution", "aegis.cli.orders", "aegis.cli.policy")
ENTRY_POINTS = ("aegis.cli.orders", "aegis.cli.policy", "aegis.policy.engine", "aegis.policy.dispatch")


def _module_name(path: Path) -> str:
    parts = path.relative_to(REPO_ROOT).with_suffix("").parts
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def _resolve(node: ast.ImportFrom, module: str) -> str:
    if not node.level:
        return node.module or ""
    base = module.split(".")[: -node.level]
    return ".".join([*base, *([node.module] if node.module else [])])


def imports_of(source: str, module: str, targets: tuple[str, ...]) -> list[str]:
    """Every module among ``targets`` that ``source`` imports, by any spelling."""
    found: list[str] = []
    for node in ast.walk(ast.parse(source)):
        names: list[str] = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            base = _resolve(node, module)
            names = [base, *(f"{base}.{alias.name}" for alias in node.names)]
        found.extend(name for name in names if any(name == t or name.startswith(t + ".") for t in targets))
    return found


def strings(source: str) -> list[str]:
    return [
        node.value
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    ]


def _id(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT))


class TestTheCheckers:
    def test_every_spelling_of_the_import_is_caught(self):
        planted = (
            "import aegis.signals.collect\n"
            "from aegis.store import signals\n"
            "from aegis.data.apewisdom import apewisdom_signals\n"
            "from ..signals import rank\n"
        )
        found = imports_of(planted, "aegis.policy.engine", SIGNAL_MODULES)
        assert {"aegis.signals.collect", "aegis.store.signals", "aegis.data.apewisdom"} <= set(found)
        assert any(name.startswith("aegis.signals") for name in found[3:])

    def test_the_reverse_direction_is_caught(self):
        planted = "from aegis.policy.engine import decide\nimport aegis.execution.paper\n"
        assert len(imports_of(planted, "aegis.signals.collect", ORDER_MODULES)) >= 2

    def test_signal_sql_is_caught_and_ordinary_words_are_not(self):
        assert SIGNAL_SQL.search("SELECT * FROM signals WHERE fetch_id = ?")
        assert SIGNAL_SQL.search("select count(*) from signal_fetches")
        assert not SIGNAL_SQL.search("the rule signals a reject")

    def test_the_scan_has_real_targets(self):
        assert AEGIS / "policy" / "engine.py" in ORDER_PATH
        assert AEGIS / "execution" / "paper.py" in ORDER_PATH
        assert AEGIS / "signals" / "collect.py" in SIGNALS_CODE
        assert all(path.exists() for path in ORDER_PATH + SIGNALS_CODE)


@pytest.mark.parametrize("path", ORDER_PATH, ids=_id)
def test_the_order_path_imports_no_signals_module(path):
    assert imports_of(path.read_text(encoding="utf-8"), _module_name(path), SIGNAL_MODULES) == []


@pytest.mark.parametrize("path", ORDER_PATH, ids=_id)
def test_the_order_path_has_no_sql_on_the_signal_tables(path):
    assert [s for s in strings(path.read_text(encoding="utf-8")) if SIGNAL_SQL.search(s)] == []


@pytest.mark.parametrize("path", SIGNALS_CODE, ids=_id)
def test_signals_never_import_the_policy_engine_or_an_executor(path):
    assert imports_of(path.read_text(encoding="utf-8"), _module_name(path), ORDER_MODULES) == []


def test_the_order_path_entry_points_load_no_signals_module():
    code = (
        "import sys\n"
        + "".join(f"import {name}\n" for name in ENTRY_POINTS)
        + f"print('\\n'.join(sorted(m for m in sys.modules if m.startswith({SIGNAL_MODULES!r}))))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True, timeout=120, check=True
    )
    assert result.stdout.strip() == ""


def test_loading_the_signals_package_loads_no_order_path_module():
    code = (
        "import sys\nimport aegis.signals.collect\n"
        f"print('\\n'.join(sorted(m for m in sys.modules if m.startswith({ORDER_MODULES!r}))))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True, timeout=120, check=True
    )
    assert result.stdout.strip() == ""
