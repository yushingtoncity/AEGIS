"""The approval page records an answer and nothing more, and the model has no
way to it. Enforced statically over the page's source with ``ast``:

1. No module of the page (``aegis/approve_web`` and its CLI) imports the
   brain, the forecasts, the model SDK, the broker SDK, the broker client
   or ``aegis.execution``.
2. From the dispatcher it uses only ``answer``, ``awaiting_approval`` and
   the two names that describe an answer; never ``place``, ``tick``,
   ``sync``, ``cancel``, ``stand_down`` or any order method.
3. Nothing under ``aegis/brain`` (nor ``aegis/forecasts``) imports the page.
4. The server binds a literal loopback address.

The dispatcher itself imports ``aegis.execution`` (for the ``Executor``
seam), so loading ``aegis.policy.dispatch`` loads it too; rule 2 is what
keeps the page's own code off every path that reaches a broker.
"""

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PAGE_FILES = sorted((ROOT / "aegis" / "approve_web").glob("*.py")) + [
    ROOT / "aegis" / "cli" / "approve_web.py"
]
MODEL_SIDE = sorted((ROOT / "aegis" / "brain").rglob("*.py")) + sorted(
    (ROOT / "aegis" / "forecasts").rglob("*.py")
) + [ROOT / "aegis" / "cli" / "brain.py"]

FORBIDDEN_IMPORTS = (
    "aegis.brain", "aegis.forecasts", "aegis.execution", "aegis.data.clients",
    "alpaca", "anthropic", "aegis.cli.brain", "aegis.cli.orders",
)
DISPATCH_ALLOWED = {"answer", "awaiting_approval", "APPROVAL_ANSWERED", "PHONE_CHANNEL"}
ORDER_NAMES = {
    "place", "tick", "sync", "cancel", "stand_down", "mint", "paper_broker", "submit_order",
    "cancel_order", "cancel_orders", "trading_client", "close_all_positions", "ready_decisions",
}


def _imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, f"{path.name}: relative import"
            found.append(node.module or "")
            found += [f"{node.module}.{alias.name}" for alias in node.names]
    return found


def _hits(name: str, prefixes: tuple[str, ...]) -> bool:
    return any(name == p or name.startswith(p + ".") for p in prefixes)


@pytest.mark.parametrize("path", PAGE_FILES, ids=lambda p: p.name)
def test_the_page_imports_no_model_and_no_broker(path):
    bad = [name for name in _imports(path) if _hits(name, FORBIDDEN_IMPORTS)]
    assert bad == [], f"{path.relative_to(ROOT)} imports {bad}"


@pytest.mark.parametrize("path", PAGE_FILES, ids=lambda p: p.name)
def test_the_page_uses_only_the_answer_side_of_the_dispatcher(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    used = {
        node.attr for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
        and node.value.id == "dispatch"
    }
    assert used <= DISPATCH_ALLOWED, f"{path.name} uses dispatch.{sorted(used - DISPATCH_ALLOWED)}"
    imported = {
        alias.name for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
        and node.module == "aegis.policy.dispatch" for alias in node.names
    }
    assert imported <= DISPATCH_ALLOWED


@pytest.mark.parametrize("path", PAGE_FILES, ids=lambda p: p.name)
def test_the_page_names_no_order_method(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    names |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    assert not names & ORDER_NAMES, f"{path.name} names {sorted(names & ORDER_NAMES)}"


@pytest.mark.parametrize("path", MODEL_SIDE, ids=lambda p: str(p.relative_to(ROOT)))
def test_the_model_side_never_reaches_the_page(path):
    text = path.read_text(encoding="utf-8")
    bad = [name for name in _imports(path) if _hits(name, ("aegis.approve_web", "aegis.cli.approve_web"))]
    assert bad == [] and "approve_web" not in text, f"{path.relative_to(ROOT)} mentions the page"


def test_the_checker_sees_a_forbidden_import(tmp_path):
    sample = tmp_path / "sample.py"
    sample.write_text("from aegis.execution.base import Executor\nimport alpaca.trading\n")
    assert [n for n in _imports(sample) if _hits(n, FORBIDDEN_IMPORTS)] == [
        "aegis.execution.base", "aegis.execution.base.Executor", "alpaca.trading"]


def test_the_server_binds_the_loopback_literal():
    tree = ast.parse((ROOT / "aegis" / "approve_web" / "server.py").read_text(encoding="utf-8"))
    [value] = [
        node.value.value for node in ast.walk(tree) if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "LOOPBACK" for t in node.targets)
    ]
    assert value == "127.0.0.1"
