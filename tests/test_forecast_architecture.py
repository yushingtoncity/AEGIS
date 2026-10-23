"""Forecasts are a measuring instrument: nothing that can place an order may
read them.

Every module on the order path (the policy engine, the executors, the
orders and policy CLIs) and every module of the trading brain (the scan,
thesis and proposal stages, the cycle, the snapshot, the brain CLI) is
scanned statically for an import of the forecast code or SQL that touches
the forecast tables. The order-path modules may not even mention the word.
A fresh interpreter then imports those entry points and checks that no
forecast module came along. The checkers are tested on planted sources so a
clean scan means something.
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
TRADING_BRAIN = sorted(
    [
        *(
            path
            for path in (AEGIS / "brain").glob("*.py")
            # forecast.py is the forecast stage itself; models.py holds its shapes beside the others
            if path.name not in {"forecast.py", "models.py"}
        ),
        AEGIS / "cli" / "brain.py",
    ]
)
SCANNED = ORDER_PATH + TRADING_BRAIN

FORECAST_MODULES = ("aegis.forecasts", "aegis.store.forecasts", "aegis.brain.forecast", "aegis.cli.forecasts")
FORECAST_SQL = re.compile(r"\bforecast_runs\b|\b(from|join|into|update|table)\s+forecasts\b", re.IGNORECASE)
ANY_MENTION = re.compile(r"forecast", re.IGNORECASE)

ENTRY_POINTS = ("aegis.cli.orders", "aegis.cli.policy", "aegis.cli.brain", "aegis.policy.engine", "aegis.brain.cycle")


def _module_name(path: Path) -> str:
    parts = path.relative_to(REPO_ROOT).with_suffix("").parts
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def _resolve(node: ast.ImportFrom, module: str) -> str:
    if not node.level:
        return node.module or ""
    base = module.split(".")[: -node.level] if node.level else []
    return ".".join([*base, *([node.module] if node.module else [])])


def forecast_imports(source: str, module: str) -> list[str]:
    """Every forecast module ``source`` imports, by any spelling of the import."""
    found: list[str] = []
    for node in ast.walk(ast.parse(source)):
        names: list[str] = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            base = _resolve(node, module)
            names = [base, *(f"{base}.{alias.name}" for alias in node.names)]
        found.extend(
            name for name in names if any(name == m or name.startswith(m + ".") for m in FORECAST_MODULES)
        )
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
            "import aegis.forecasts.score\n"
            "from aegis.store import forecasts\n"
            "from aegis.brain.forecast import run_forecast_arm\n"
            "from ..forecasts import inputs\n"
        )
        found = forecast_imports(planted, "aegis.policy.engine")
        assert "aegis.forecasts.score" in found
        assert "aegis.store.forecasts" in found
        assert "aegis.brain.forecast" in found
        assert "aegis.forecasts" in found or "aegis.forecasts.inputs" in found

    def test_forecast_sql_is_caught_and_ordinary_words_are_not(self):
        assert FORECAST_SQL.search("SELECT * FROM forecasts WHERE run_id = ?")
        assert FORECAST_SQL.search("select probability from forecast_runs")
        assert not FORECAST_SQL.search("the scan stage forecasts nothing")

    def test_the_scan_has_real_targets(self):
        assert AEGIS / "execution" / "base.py" in ORDER_PATH
        assert AEGIS / "policy" / "engine.py" in ORDER_PATH
        assert AEGIS / "brain" / "cycle.py" in TRADING_BRAIN
        assert AEGIS / "brain" / "proposal.py" in TRADING_BRAIN
        assert AEGIS / "brain" / "forecast.py" not in SCANNED
        assert all(path.exists() for path in SCANNED)


@pytest.mark.parametrize("path", SCANNED, ids=_id)
def test_no_forecast_import(path):
    source = path.read_text(encoding="utf-8")
    assert forecast_imports(source, _module_name(path)) == []


@pytest.mark.parametrize("path", SCANNED, ids=_id)
def test_no_sql_on_the_forecast_tables(path):
    hits = [s for s in strings(path.read_text(encoding="utf-8")) if FORECAST_SQL.search(s)]
    assert hits == []


@pytest.mark.parametrize("path", ORDER_PATH, ids=_id)
def test_the_order_path_never_mentions_forecasts(path):
    assert not ANY_MENTION.search(path.read_text(encoding="utf-8"))


def test_the_entry_points_load_no_forecast_module():
    code = (
        "import sys\n"
        + "".join(f"import {name}\n" for name in ENTRY_POINTS)
        + f"print('\\n'.join(sorted(m for m in sys.modules if m.startswith({FORECAST_MODULES!r}))))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True, timeout=120, check=True
    )
    assert result.stdout.strip() == ""
