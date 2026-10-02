"""Governance, enforced statically and at runtime: who may reference an
Executor, and what the policy engine may be made of.

PART 1 — only ``aegis/policy`` may reference an Executor. Every ``.py``
under ``aegis/`` except ``aegis/policy`` and ``aegis/execution`` itself —
the data layer, the pricing engine, the store, the brain, every CLI
(``aegis/cli/policy.py`` included), ``config.py``, the notify and dashboard
stubs and ``aegis/__init__.py`` — is parsed with ``ast`` and held to six
rules:

a. No import of ``aegis.execution`` in any static form — ``import a.b``,
   ``from a.b import c``, ``from aegis import execution``, relative imports
   resolved against the module's package, or a literal argument of
   ``import_module`` / ``__import__`` (string concatenation folded, a
   relative name resolved against a literal ``package``, a literal
   ``fromlist`` included). An argument that is not a literal — or the
   function passed around instead of called — is recorded as a *computed
   import*, which counts as importing anything at all.
b. No identifier that names an Executor: no ``Name``, ``Attribute``,
   imported name or alias, function, class, argument, keyword, exception,
   pattern-capture or type-parameter name whose lower-cased form contains
   ``executor`` — which covers the class, every ``…Executor`` class, and the
   variable, parameter or attribute an instance is conventionally kept in
   (``executor``, ``self._executor``, ``EXECUTOR``). Exactly four names are
   let through: the stdlib pool executors (``ThreadPoolExecutor``,
   ``ProcessPoolExecutor``, ``InterpreterPoolExecutor``) and asyncio's
   ``run_in_executor`` — they run callables, not orders. The bare identifier
   ``execution`` is a finding too, as a ``Name``, an ``Attribute`` or an
   imported name: ``engine.execution`` and ``from aegis.policy.engine import
   execution`` name the package off a policy module that holds it.
c. No string or bytes literal that contains ``Executor`` or pairs
   ``execution`` with ``aegis`` or a path separator — ``"aegis.execution"``,
   ``"aegis/execution/base.py"``, ``getattr(module, "Executor")`` — with
   concatenations and f-strings read as the text they build. Docstrings
   (and any other bare string statement, which no code can use) may mention
   both freely.
d. No attribute path that reaches ``aegis.execution`` without importing it.
   Once anything has loaded the package — the policy engine will, in Phase 6
   — it is an attribute of ``aegis``, there for whoever holds that package:
   ``aegis.execution`` after ``import aegis`` (or ``import aegis.store``,
   which binds ``aegis`` too, or ``import aegis as a``) names it with no
   import and no ``Executor`` in sight. So does ``getattr(aegis,
   "execution")``; so does either on an alias made by assignment, on
   ``sys.modules["aegis"]`` or on what a literal ``import_module`` /
   ``__import__`` call returns; and ``from aegis import *`` binds it
   unnamed. An attribute that is not spelled out, taken from a package above
   the execution package (``getattr(aegis, name)``, ``vars(aegis)``,
   ``aegis.__dict__``), is recorded as a *computed attribute*, which counts
   as reaching it.
e. No handle on the ``aegis`` package itself. Rule (d) follows a name through
   imports and plain assignments only: handed on any other way — a parameter,
   a return value, a container, an attribute — the package gives up
   ``.execution`` unread. So the plain ways an outside module binds or
   obtains the package are each a finding: ``import aegis``, ``import
   aegis.store`` (it binds ``aegis``) or ``import aegis as a``;
   ``sys.modules["aegis"]`` or ``.get("aegis")``; a literal
   ``import_module`` / ``__import__`` call that returns it; ``from aegis
   import __dict__``, or any other dunder. A ``sys.modules`` lookup whose key
   is not a literal — or the table handed on instead of looked up — is
   recorded as a *computed module*, which counts as the package. An
   attribute called ``.modules`` on anything the source does not bind to
   ``sys`` is read as that table too (``os.sys.modules`` is), and reported
   as ``<attribute .modules>``, so the finding says what was read: honest
   code that trips it renames its attribute. In ``aegis/__init__.py``, whose
   globals ARE the package's namespace, the bare name ``execution`` and a
   no-argument ``globals()`` / ``vars()`` / ``locals()`` are that same
   handle. ``from aegis import store``, ``from aegis.store import repo`` and
   ``import aegis.store as store`` bind something below the package and stay
   fine. One thing more: no ``from aegis.policy… import *``, which copies
   whatever the policy module holds, an Executor included. That is what the
   rule reads, and no more: the package reached by a road that is not on
   this list (``sys`` handed to a library call, a library call that returns
   a module), or the namespace of an imported policy module read with
   ``vars(engine)``, is beyond a static rule — and is the business of the
   runtime check of who HOLDS an Executor, below.
f. No name of the Executor interface's order methods — ``submit_order``,
   ``cancel_order``, ``close_position`` — as an identifier of any kind (a
   call on whatever was handed over, a definition, an argument, an imported
   name) or inside a string or bytes literal (docstrings aside;
   concatenations and f-strings read as in rule (c)). ``get_open_orders``
   belongs to the interface too, but it is also a function of the store,
   which outside modules call: it is not on the list.

PART 2 — ``aegis/policy`` is pure, deterministic Python; no model calls.
Every ``.py`` under ``aegis/policy``:

- never imports ``anthropic``, ``aegis.brain``, ``alpaca``,
  ``aegis.data.clients``, ``aegis.data.news``, ``requests``, ``httpx``,
  ``httpx2``, ``urllib``, ``socket``, ``random``, ``secrets`` or
  ``subprocess``, and makes no computed import;
- only ``context.py`` imports the live fetchers (``aegis.data.market`` /
  ``aegis.data.account``);
- ``rules.py`` and ``measures.py`` import nothing outside a short allowlist
  — so not the store's repository or database, ``sqlite3``, ``time``, ``os``
  or ``uuid`` — and from ``aegis.config`` only types;
- reads no clock outside ``context.py``: no reference to ``utcnow``; no call
  of ``.now()`` / ``.today()`` on anything; no reference at all to ``.now``
  / ``.today`` on ``datetime`` or ``date`` — a clock handed on is a clock
  (``default_factory=datetime.now``, ``_now = datetime.now``) — nor on
  anything that is not a bare name (``context.now.now``, ``type(x).now``:
  the same clock, taken off an instance); and no
  ``time`` module, brought in by a statement (``import time``, ``from time
  import …``) or by a call (``__import__("time")``, or an import call whose
  target cannot be read off the source). Attribute ACCESS on a name that
  is not one of those classes is fine — ``context.now`` is how a rule learns
  the time. What the rule does not follow, pinned as unseen: the class
  reaching a bare name by a tuple unpacking or as a parameter's default
  (``a, b = datetime, 1`` / ``def f(clock=datetime)``) and the clock then
  handed on off that name — a read whose value is USED is caught by the
  behaviour tests (``decided_at`` is the context's ``now``).

Each checker is tested against synthetic snippets, so a form it missed
shows up here rather than in production. The static rules are a tripwire,
not a sandbox, so each part has a runtime backstop in a fresh interpreter.

For PART 1 it walks the tree, imports every module outside the two packages,
and makes two checks:

- Who ASKS for it. Every import of ``aegis.execution`` (and its submodules)
  is charged to the module that asked for it — the nearest calling frame,
  outside the import machinery, that belongs to ``aegis`` — and every such
  importer must be under ``aegis.policy`` or ``aegis.execution``. Not that
  nothing loaded the package: once the policy engine imports the Executor
  (Phase 6), importing ``aegis.cli.policy`` loads it too, through the one
  package that may. For an import — a statement or a call — the charge does
  not depend on who loaded the package first. A load from a spec or from a
  file path does: it is seen (by a meta-path finder, or as a load nobody
  asked for) only when it is the first load, and a reference taken off an
  attribute asks the import system for nothing at all.
- Who HOLDS it — whatever the road, whoever loaded the package first. After
  the imports, no module outside the two packages may have among its
  module-level values — or one level inside a list, tuple, set or dict value
  — the execution package or a module under it, or a class, function or
  instance whose ``__module__`` is under it (reported as ``holders``). A
  package's own submodule is not a holding: ``aegis.execution`` hangs off
  ``aegis`` once anything has loaded it. An outside module that imports a
  policy module holds nothing by that, whatever the policy module holds.

The limit of PART 1, stated here and not chased: a deliberately obfuscated,
DORMANT reference — a name assembled from string pieces and resolved through
``__builtins__`` or ``eval`` inside a function nobody calls during import —
is seen by neither the static rules nor the runtime checks. And the holding
check reads a module's own module-level values and one level into a list,
tuple, set, frozenset or dict value — no further. A reference kept anywhere
else is its blind spot too: in a function's local variable, a default
argument or a closure; as an attribute of a class or of an object the module
defines (``class Box: held = …``, ``self.held = …``); in any other container
(a ``deque``, a ``SimpleNamespace``, a ``functools.partial``); as a dict's
key, or nested more than one container deep; or as an instance of a subclass
written in a policy module (the check goes by where a class says it was
defined). Each of those is pinned as unseen, so nobody takes the silence for
coverage. The brain is held to a stricter no-dynamic-machinery rule by
its own test (``tests/test_brain_architecture.py``); the rest of the tree
has legitimate uses for ``os`` and for ``importlib.resources``-style
machinery, so that rule cannot be applied tree-wide.

For PART 2 the backstop imports the four pure modules (``rules``,
``measures``, ``engine``, ``limits``) and asserts neither ``anthropic`` nor
any ``aegis.brain*`` or ``alpaca*`` module was loaded. The harness is proven
on trees and imports it must reject — and on ones it must allow.
"""

import ast
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import aegis.config
from aegis.config import REPO_ROOT

AEGIS_DIR = REPO_ROOT / "aegis"
POLICY_DIR = AEGIS_DIR / "policy"
EXECUTION_DIR = AEGIS_DIR / "execution"
CLI_POLICY = AEGIS_DIR / "cli" / "policy.py"

ROOT = "aegis"
EXECUTION = "aegis.execution"
EXECUTOR = "Executor"
MAY_IMPORT_EXECUTION = ("aegis.policy", EXECUTION)
"""The two packages whose modules may ask for ``aegis.execution`` — and so
may hold an Executor."""
POOL_EXECUTORS = frozenset(
    {"ThreadPoolExecutor", "ProcessPoolExecutor", "InterpreterPoolExecutor"}
)
"""The stdlib pool executors: the only ``…Executor`` classes rule (b) lets through."""
NOT_AN_EXECUTOR = POOL_EXECUTORS | {"run_in_executor"}
"""Exactly the identifiers containing ``executor`` — in any case — that rule
(b) lets through: the pool executors and asyncio's way of handing a callable
to one."""
EXECUTION_NAME = EXECUTION.rpartition(".")[2]
"""``execution``: what the package is called on whatever holds it."""
ORDER_METHODS = frozenset({"submit_order", "cancel_order", "close_position"})
"""The Executor interface's order methods — what rule (f) reads for.
``get_open_orders`` is the interface's fourth method and deliberately absent:
``aegis.store.repo`` has a function of that name, which outside modules call."""

COMPUTED_IMPORT = "<computed import>"
"""Recorded for an ``import_module`` / ``__import__`` whose target cannot be
read off the source: it could resolve to anything, so it counts as importing
everything (``imports``)."""
IMPORT_CALLS = frozenset({"import_module", "__import__"})

COMPUTED_ATTRIBUTE = "<computed attribute>"
"""Recorded for an attribute whose name cannot be read off the source
(``getattr(aegis, name)``, ``vars(aegis)``): taken from a package above the
execution package it could be that package, so it counts as reaching it
(``_reaches``)."""
ATTRIBUTE_CALLS = frozenset({"getattr", "setattr", "delattr", "hasattr"})
MODULE_TABLE = "sys.modules"

COMPUTED_MODULE = "<computed module>"
"""Recorded for a ``sys.modules`` lookup whose key cannot be read off the
source, or for the table handed on instead of looked up: it could yield any
loaded module, so it counts as obtaining the ``aegis`` package
(``root_handles``)."""
ATTRIBUTE_MODULES = "<attribute .modules>"
"""Recorded instead of ``COMPUTED_MODULE`` when the table is an attribute
called ``.modules`` on something the source does not bind to ``sys``: it is
read as ``sys.modules`` all the same — ``os.sys.modules`` is that table — and
the label says so, for the honest ``self.modules`` that trips the rule and
has to be renamed."""
NAMESPACE_CALLS = frozenset({"globals", "vars", "locals"})

# What sits outside the two packages today — the coverage test pins the scan to it.
EXPECTED_OUTSIDE = {
    "__init__.py", "config.py", "brain", "cli", "dashboard", "data", "notify", "pricing", "store",
}
EXPECTED_POLICY_MODULES = {
    "__init__.py", "context.py", "engine.py", "errors.py", "limits.py", "measures.py",
    "models.py", "rules.py",
}

# PART 2
FORBIDDEN_IN_POLICY = (
    "anthropic", "aegis.brain", "alpaca", "aegis.data.clients", "aegis.data.news",
    "requests", "httpx", "httpx2", "urllib", "socket", "random", "secrets", "subprocess",
)
LIVE_FETCHERS = ("aegis.data.market", "aegis.data.account")
CONFIG = "aegis.config"
MEASURES_ALLOWED = frozenset(
    {
        "__future__", "math", "datetime", "collections.abc", "typing",
        "aegis.policy.models", CONFIG, "aegis.data.models", "aegis.pricing.models",
        "aegis.store.models",
    }
)
RULES_ALLOWED = MEASURES_ALLOWED | {"aegis.policy.measures"}
PURE_MODULES = (
    "aegis.policy.rules", "aegis.policy.measures", "aegis.policy.engine", "aegis.policy.limits",
)
NEVER_LOADED_BY_THE_PURE_CORE = ("anthropic", "aegis.brain", "alpaca")
CLOCK_METHODS = frozenset({"now", "today"})
CLOCK_OWNERS = frozenset({"datetime", "date"})
"""The names whose ``.now`` / ``.today`` / ``.utcnow`` is the wall clock —
referenced, not only called: handed on (``default_factory=datetime.now``) it
is read by whoever calls it later."""
CLOCK_ATTRIBUTES = CLOCK_METHODS | {"utcnow"}
UNNAMED_OWNER = "<expression>"
"""How a finding names the owner of a clock attribute that is taken off an
expression rather than a name (``context.now.now``, ``type(x).now``)."""
TIME_FUNCTIONS = frozenset(
    {"monotonic", "monotonic_ns", "time_ns", "perf_counter", "perf_counter_ns"}
)


# --- reading a source file ----------------------------------------------------


def _source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _id(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT))


def _package_of(path: Path) -> str:
    """The dotted package a module's relative imports resolve against."""
    parts = list(path.relative_to(REPO_ROOT).with_suffix("").parts)
    parts.pop()  # ``__init__`` or the module's own name: either way its directory's package
    return ".".join(parts)


def _module_name(path: Path) -> str:
    """The dotted name a file is imported under."""
    parts = list(path.relative_to(REPO_ROOT).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _resolve(module: str | None, level: int, package: str) -> str:
    """``from <module> import`` with ``level`` leading dots, as an absolute name."""
    if level == 0:
        return module or ""
    base = package.split(".")
    if level > 1:
        base = base[: len(base) - (level - 1)]
    return ".".join(part for part in [*base, module or ""] if part)


def _within(name: str, modules: tuple[str, ...]) -> bool:
    return any(name == module or name.startswith(module + ".") for module in modules)


def _static_imports(tree: ast.AST, package: str) -> set[str]:
    """Every module an ``import`` statement names (``from a import b`` yields ``a`` and ``a.b``)."""
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = _resolve(node.module, node.level, package)
            found.add(base)
            found.update(f"{base}.{alias.name}" if base else alias.name for alias in node.names)
    return found


def _bare_strings(tree: ast.AST) -> set[int]:
    """ids of the string constants that are whole statements — docstrings and
    their like — which no code can use."""
    return {
        id(node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
    }


def _folded(node: ast.AST | None) -> str | None:
    """The text an expression made only of string literals builds — a
    literal, ``"a" + "b"``, an f-string of literals — else None."""
    if isinstance(node, ast.Constant):
        return node.value if isinstance(node.value, str) else None
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left, right = _folded(node.left), _folded(node.right)
        return None if left is None or right is None else left + right
    if isinstance(node, ast.JoinedStr):
        parts = [_folded(value) for value in node.values]
        return None if None in parts else "".join(parts)
    if isinstance(node, ast.FormattedValue) and node.conversion == -1 and node.format_spec is None:
        return _folded(node.value)
    return None


def _argument(call: ast.Call, position: int, keyword: str) -> ast.expr | None:
    """A call's argument by position or by keyword; None when it is not passed."""
    if len(call.args) > position:
        return call.args[position]
    for item in call.keywords:
        if item.arg == keyword:
            return item.value
    return None


def _terminal_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Attribute):
        return node.attr
    return node.id if isinstance(node, ast.Name) else None


def _import_call_names(tree: ast.AST) -> set[str]:
    """The names ``import_module`` / ``__import__`` go by in a tree: their
    own, and every ``from importlib import import_module as load`` alias."""
    names = set(IMPORT_CALLS)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            names.update(a.asname for a in node.names if a.name in IMPORT_CALLS and a.asname)
    return names


def _dynamic_imports(tree: ast.AST) -> set[str]:
    """What the ``import_module`` / ``__import__`` calls in a tree import: the
    literal targets by name, ``COMPUTED_IMPORT`` for any that cannot be read
    off the source (a computed name, a relative name without a literal
    package, the function aliased or handed on instead of called)."""
    names = _import_call_names(tree)
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and _terminal_name(node.func) in names
    ]
    called = {id(call.func) for call in calls}
    found: set[str] = set()
    for node in ast.walk(tree):  # the function as a value: whatever calls it, we cannot see
        if isinstance(node, (ast.Name, ast.Attribute)) and _terminal_name(node) in names:
            if id(node) not in called:
                found.add(COMPUTED_IMPORT)
    for call in calls:
        target = _folded(_argument(call, 0, "name"))
        if target is None:
            found.add(COMPUTED_IMPORT)
            continue
        level = len(target) - len(target.lstrip("."))
        if _terminal_name(call.func) == "__import__":
            depth = _argument(call, 4, "level")
            if level or not (depth is None or isinstance(depth, ast.Constant) and depth.value == 0):
                found.add(COMPUTED_IMPORT)  # a relative __import__: resolved at runtime
                continue
            found.add(target)
            fromlist = _argument(call, 3, "fromlist")
            if fromlist is not None:  # ``__import__("aegis", fromlist=["execution"])``
                items = fromlist.elts if isinstance(fromlist, (ast.List, ast.Tuple)) else [None]
                for text in (_folded(item) for item in items):
                    found.add(COMPUTED_IMPORT if text is None else f"{target}.{text}")
            continue
        if level:  # ``import_module("..execution", "aegis.cli")``
            anchor = _folded(_argument(call, 1, "package"))
            if anchor is None:
                found.add(COMPUTED_IMPORT)
                continue
            target = _resolve(target[level:] or None, level, anchor)
        found.add(target)
    return found


def imported_modules(source: str, *, package: str = "") -> set[str]:
    """Every module name a source file imports, in any static form — plus
    ``COMPUTED_IMPORT`` when it makes an import whose target cannot be read
    off the source."""
    tree = ast.parse(source)
    return _static_imports(tree, package) | _dynamic_imports(tree)


def imports(found: set[str], module: str) -> bool:
    """Whether ``module`` or anything under it is in ``found`` — or ``found``
    holds a computed import, whose target cannot be known statically."""
    return COMPUTED_IMPORT in found or any(_within(name, (module,)) for name in found)


# --- PART 1: the checkers -----------------------------------------------------


def _identifiers(tree: ast.AST):
    """Every identifier a tree uses or binds, wherever the grammar puts it."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            yield node.id
        elif isinstance(node, ast.Attribute):
            yield node.attr
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            yield node.name
        elif isinstance(node, ast.arg):
            yield node.arg
        elif isinstance(node, ast.keyword):
            if node.arg:
                yield node.arg
        elif isinstance(node, ast.alias):
            yield from node.name.split(".")
            if node.asname:
                yield node.asname
        elif isinstance(node, ast.ImportFrom):
            yield from (node.module or "").split(".")
        elif isinstance(node, (ast.ExceptHandler, ast.MatchAs, ast.MatchStar)):
            if node.name:
                yield node.name
        elif isinstance(node, ast.MatchMapping):
            if node.rest:
                yield node.rest
        elif isinstance(node, ast.MatchClass):
            yield from node.kwd_attrs
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            yield from node.names
        elif isinstance(node, (ast.TypeVar, ast.ParamSpec, ast.TypeVarTuple)):
            yield node.name


def _references(tree: ast.AST):
    """The identifiers a tree reads, takes as an attribute or imports — every
    part of a dotted module path, and an alias — as opposed to the ones it
    only defines (a function, a class, an argument)."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            yield node.id
        elif isinstance(node, ast.Attribute):
            yield node.attr
        elif isinstance(node, ast.alias):
            yield from node.name.split(".")
            if node.asname:
                yield node.asname
        elif isinstance(node, ast.ImportFrom):
            yield from (node.module or "").split(".")


def executor_names(source: str) -> set[str]:
    """Rule (b): every identifier in ``source`` that names an Executor — its
    lower-cased form contains ``executor`` and it is not one of the four
    names in ``NOT_AN_EXECUTOR`` — and the bare identifier ``execution``
    wherever it is a ``Name``, an ``Attribute`` or an imported name (empty
    when clean)."""
    tree = ast.parse(source)
    marker = EXECUTOR.lower()
    named = {
        name
        for name in _identifiers(tree)
        if marker in name.lower() and name not in NOT_AN_EXECUTOR
    }
    return named | {name for name in _references(tree) if name == EXECUTION_NAME}


def _names_execution(text: str) -> bool:
    lowered = text.lower()
    paired = "aegis" in lowered or "/" in lowered or "\\" in lowered
    return EXECUTOR in text or ("execution" in lowered and paired)


def _literal_texts(tree: ast.AST):
    """The text of every string or bytes literal in a tree, docstrings
    aside. Bytes are read as latin-1; a concatenation or an f-string is read
    as the text its literal parts build (the holes of an f-string closed up),
    next to the parts themselves."""
    bare = _bare_strings(tree)
    for node in ast.walk(tree):
        text: str | None = None
        if isinstance(node, ast.Constant) and id(node) not in bare:
            if isinstance(node.value, bytes):
                text = node.value.decode("latin-1")
            elif isinstance(node.value, str):
                text = node.value
        elif isinstance(node, ast.BinOp):
            text = _folded(node)
        elif isinstance(node, ast.JoinedStr):  # the literal parts, the holes closed up
            text = "".join(_folded(value) or "" for value in node.values)
        if text is not None:
            yield text


def executor_literals(source: str) -> set[str]:
    """Rule (c): every string or bytes literal (docstrings aside) that
    contains ``Executor`` or pairs ``execution`` with ``aegis`` or a path
    separator. Bytes are read as latin-1; a concatenation or an f-string is
    read as the text its literal parts build, so ``"aegis." + "execution"``
    and ``f"{root}/execution"`` are seen too."""
    return {text for text in _literal_texts(ast.parse(source)) if _names_execution(text)}


def order_methods(source: str) -> set[str]:
    """Rule (f): every order method of the Executor interface
    (``ORDER_METHODS``) that ``source`` names — as an identifier, wherever
    the grammar puts one (an attribute, a call, a definition, an argument, an
    imported name), or inside a string or bytes literal (docstrings aside),
    read as rule (c) reads them. Empty when clean."""
    tree = ast.parse(source)
    found = {name for name in _identifiers(tree) if name in ORDER_METHODS}
    for text in _literal_texts(tree):
        found.update(method for method in ORDER_METHODS if method in text)
    return found


def _returned_module(call: ast.Call) -> str | None:
    """The module a literal ``import_module`` / ``__import__`` call returns —
    for ``__import__("a.b")`` without a ``fromlist`` that is the top-level
    package ``a`` — or None when it cannot be read off the source."""
    target = _folded(_argument(call, 0, "name"))
    if target is None:
        return None
    level = len(target) - len(target.lstrip("."))
    if _terminal_name(call.func) == "__import__":
        depth = _argument(call, 4, "level")
        if level or not (depth is None or isinstance(depth, ast.Constant) and depth.value == 0):
            return None  # a relative __import__: resolved at runtime
        fromlist = _argument(call, 3, "fromlist")
        empty = fromlist is None or (
            isinstance(fromlist, (ast.List, ast.Tuple)) and not fromlist.elts
        )
        return target.partition(".")[0] if empty else target
    if level:
        anchor = _folded(_argument(call, 1, "package"))
        return None if anchor is None else _resolve(target[level:] or None, level, anchor)
    return target


def _paths(node: ast.AST | None, bound: dict[str, set[str]], loaders: set[str]) -> set[str]:
    """The dotted paths an expression may stand for, as far as the source
    says (empty when it does not): a bound name; an attribute of one — written
    with a dot, or with ``getattr`` and its like, where a name that is not a
    literal is ``COMPUTED_ATTRIBUTE``, as is the whole namespace
    (``__dict__`` / ``vars()``); a literal ``sys.modules`` entry; what a
    literal ``import_module`` / ``__import__`` call returns."""
    if isinstance(node, ast.Name):
        return set(bound.get(node.id, ()))
    if isinstance(node, ast.NamedExpr):
        return _paths(node.value, bound, loaders)
    if isinstance(node, ast.Attribute):
        name = COMPUTED_ATTRIBUTE if node.attr == "__dict__" else node.attr
        return {f"{base}.{name}" for base in _paths(node.value, bound, loaders)}
    key: ast.expr | None = None
    if isinstance(node, ast.Subscript):  # ``sys.modules["aegis"]``
        table, key = node.value, node.slice
    elif isinstance(node, ast.Call):
        name = _terminal_name(node.func)
        if name in ATTRIBUTE_CALLS and len(node.args) >= 2:
            attribute = _folded(node.args[1]) or COMPUTED_ATTRIBUTE
            return {f"{base}.{attribute}" for base in _paths(node.args[0], bound, loaders)}
        if name == "vars" and len(node.args) == 1:
            return {f"{base}.{COMPUTED_ATTRIBUTE}" for base in _paths(node.args[0], bound, loaders)}
        if name in loaders:
            module = _returned_module(node)
            return set() if module is None else {module}
        if not (name == "get" and isinstance(node.func, ast.Attribute) and node.args):
            return set()
        table, key = node.func.value, node.args[0]  # ``sys.modules.get("aegis")``
    else:
        return set()
    if _terminal_name(table) == "modules" or MODULE_TABLE in _paths(table, bound, loaders):
        module = _folded(key)
        return set() if module is None else {module}
    return set()


def _module_bindings(tree: ast.AST, package: str) -> dict[str, set[str]]:
    """The dotted paths each name in a tree may stand for, as far as its
    imports and plain assignments say: ``import a.b`` binds ``a``; ``import
    a.b as c`` binds ``c`` to ``a.b``; ``from a import b`` binds ``b`` to
    ``a.b``; ``x = <a path>`` (annotated or ``:=`` alike) binds ``x`` to that
    path. Scopes and order are ignored — a name stands for everything it is
    ever bound to — so an alias made in one function and used in another is
    still followed."""
    bound: dict[str, set[str]] = {}
    assignments: list[tuple[list[ast.expr], ast.expr]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.partition(".")[0]
                path = alias.name if alias.asname else top
                bound.setdefault(alias.asname or top, set()).add(path)
        elif isinstance(node, ast.ImportFrom):
            base = _resolve(node.module, node.level, package)
            for alias in node.names:
                if alias.name != "*":
                    path = f"{base}.{alias.name}" if base else alias.name
                    bound.setdefault(alias.asname or alias.name, set()).add(path)
        elif isinstance(node, ast.Assign):
            assignments.append((node.targets, node.value))
        elif isinstance(node, (ast.AnnAssign, ast.NamedExpr)) and node.value is not None:
            assignments.append(([node.target], node.value))
    loaders = _import_call_names(tree)
    # an alias of an alias takes a pass each; one pass per assignment is enough
    # for any chain, and bounds a name that feeds itself (``node = node.parent``)
    for _ in assignments:
        grown = False
        for targets, value in assignments:
            paths = _paths(value, bound, loaders)
            for target in targets:
                if isinstance(target, ast.Name) and not paths <= bound.get(target.id, set()):
                    bound.setdefault(target.id, set()).update(paths)
                    grown = True
        if not grown:
            break
    return bound


def _reaches(path: str, target: str) -> bool:
    """Whether a dotted path is ``target`` or below it — or leaves a package
    above ``target`` through a computed attribute, which could be the next
    step towards it."""
    spelled, computed, _ = path.partition("." + COMPUTED_ATTRIBUTE)
    return _within(spelled, (target,)) or (bool(computed) and _within(target, (spelled,)))


def execution_attributes(source: str, *, package: str = "", target: str = EXECUTION) -> set[str]:
    """Rule (d): every dotted path in ``source`` that reaches ``target`` —
    the execution package — off an attribute rather than through an import
    of it: ``aegis.execution`` on anything bound to the ``aegis`` package (by
    an import, an alias, ``sys.modules`` or an import call), ``getattr`` and
    its like with a literal name, a computed attribute of a package above the
    target, and ``from aegis import *``, which binds the submodule unnamed
    (reported as ``aegis.*``). Empty when clean."""
    tree = ast.parse(source)
    bound, loaders = _module_bindings(tree, package), _import_call_names(tree)
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            base = _resolve(node.module, node.level, package)
            if _within(target, (base,)) and any(alias.name == "*" for alias in node.names):
                found.add(f"{base}.*")
        elif isinstance(node, (ast.Attribute, ast.Subscript, ast.Call)):
            found.update(path for path in _paths(node, bound, loaders) if _reaches(path, target))
    return found


def _is_dunder(name: str) -> bool:
    return len(name) > 4 and name.startswith("__") and name.endswith("__")


def root_handles(source: str, *, package: str = "", module: str = "") -> set[str]:
    """Rule (e): every way ``source`` binds or obtains the ``aegis`` package
    object itself, described (empty when clean): an ``import`` that binds it;
    its ``sys.modules`` entry — ``COMPUTED_MODULE`` when the key is not a
    literal, the table is handed on, or the table is itself a computed
    attribute of ``sys``; a literal import call that returns it; a dunder
    imported from it. Also a star import from a package that may hold an
    Executor, which binds one unnamed.

    Any attribute called ``.modules`` is read as the table, whatever it hangs
    off (``os.sys.modules`` is the table too). Where the source does not bind
    it to ``sys.modules``, an unreadable lookup or a handing-on is reported as
    ``ATTRIBUTE_MODULES`` rather than ``COMPUTED_MODULE``: fail-closed either
    way, but the finding says what was read.

    ``module`` is the name ``source`` is imported under: in the package's own
    ``__init__`` the globals are the package's namespace, so the bare name
    ``execution`` and a no-argument ``globals()`` / ``vars()`` / ``locals()``
    are reported too."""
    tree = ast.parse(source)
    bound, loaders = _module_bindings(tree, package), _import_call_names(tree)

    def is_table(node: ast.AST) -> bool:
        # any ``….modules`` — ``os.sys.modules`` is the table too — or a name bound to it
        named = isinstance(node, ast.Attribute) and node.attr == "modules"
        return named or MODULE_TABLE in _paths(node, bound, loaders)

    def unread(table: ast.AST) -> str:
        # what to call a table nobody can read: by what the source shows it to be
        known = MODULE_TABLE in _paths(table, bound, loaders)
        return COMPUTED_MODULE if known else ATTRIBUTE_MODULES

    found: set[str] = set()
    keys: list[tuple[ast.AST, ast.expr | None]] = []  # each table, and what it is looked up by
    read: set[int] = set()  # the table where a lookup or a membership test reads it
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:  # ``import aegis.store as store`` binds the submodule
                top = alias.name.partition(".")[0]
                if top == ROOT and (alias.name == ROOT or not alias.asname):
                    rename = f" as {alias.asname}" if alias.asname else ""
                    found.add(f"import {alias.name}{rename}")
        elif isinstance(node, ast.ImportFrom):
            base = _resolve(node.module, node.level, package)
            for alias in node.names:
                star = alias.name == "*" and _within(base, MAY_IMPORT_EXECUTION)
                if star or (base == ROOT and _is_dunder(alias.name)):
                    found.add(f"from {base} import {alias.name}")
        elif isinstance(node, ast.Subscript) and is_table(node.value):
            keys.append((node.value, node.slice))
            read.add(id(node.value))
        elif isinstance(node, ast.Compare):  # ``name in sys.modules`` yields no module
            read.update(
                id(side)
                for op, side in zip(node.ops, node.comparators)
                if isinstance(op, (ast.In, ast.NotIn)) and is_table(side)
            )
        elif isinstance(node, ast.Call):
            func, name = node.func, _terminal_name(node.func)
            if name == "get" and isinstance(func, ast.Attribute) and is_table(func.value):
                keys.append((func.value, node.args[0] if node.args else None))
                read.add(id(func.value))
            elif name in loaders and _returned_module(node) == ROOT:
                found.add(f"{name}() returns {ROOT}")
            elif module == ROOT and name in NAMESPACE_CALLS and not (node.args or node.keywords):
                found.add(f"{name}()")
        elif isinstance(node, ast.Name) and module == ROOT:
            if node.id == EXECUTION.removeprefix(f"{ROOT}."):
                found.add(node.id)
    for table, key in keys:
        text = _folded(key)
        if text is None:
            found.add(unread(table))
        elif text == ROOT:
            found.add(f"{MODULE_TABLE}[{ROOT!r}]")
    for node in ast.walk(tree):
        paths = _paths(node, bound, loaders)
        if is_table(node) and id(node) not in read:  # handed on: whoever looks it up, unseen
            found.add(unread(node))
        elif any(COMPUTED_ATTRIBUTE in path and _reaches(path, MODULE_TABLE) for path in paths):
            found.add(COMPUTED_MODULE)  # ``vars(sys)``, ``getattr(sys, name)``: the table, maybe
    return found


def _handles_in(path: Path) -> set[str]:
    """Rule (e) on a file of the tree, read as the module it is imported as —
    which is what makes ``aegis/__init__.py`` the package's own namespace."""
    return root_handles(_source(path), package=_package_of(path), module=_module_name(path))


def _sources(directory: Path) -> list[Path]:
    return sorted(directory.rglob("*.py"))


def _outside_sources() -> list[Path]:
    """Every ``.py`` under ``aegis/`` that is neither the policy engine nor
    the execution package: what PART 1 scans."""
    return [
        path
        for path in _sources(AEGIS_DIR)
        if POLICY_DIR not in path.parents and EXECUTION_DIR not in path.parents
    ]


OUTSIDE_SOURCES = _outside_sources()
POLICY_SOURCES = _sources(POLICY_DIR)


class TestImportChecker:
    @pytest.mark.parametrize(
        "snippet",
        [
            "import aegis.execution",
            "import aegis.execution as ex",
            "import os, aegis.execution",
            "import aegis.execution.base",
            "from aegis.execution import Executor",
            "from aegis.execution.base import Executor as E",
            "from aegis.execution import *",
            "from aegis import execution",
            "from aegis import store, execution",
            "from aegis import execution as ex",
            "import importlib\nimportlib.import_module('aegis.execution')",
            "import importlib\nimportlib.import_module('aegis.execution.base')",
            "import importlib\nimportlib.import_module(name='aegis.execution')",
            "from importlib import import_module\nimport_module('aegis.execution')",
            "from importlib import import_module as load\nload('aegis.execution')",
            "__import__('aegis.execution')",
            "import importlib\nimportlib.__import__('aegis.' + 'execution')",
            "__import__('aegis.execution.base', globals(), locals(), ['Executor'], 0)",
            "__import__('aegis', fromlist=['execution'])",
            "__import__('aegis', globals(), locals(), ('execution',))",
            "import importlib\nimportlib.import_module('aegis.' + 'execution')",
            "import importlib\nimportlib.import_module(f'aegis.{\"execution\"}')",
            "import importlib\nimportlib.import_module('.execution', 'aegis')",
            "import importlib\nimportlib.import_module('..execution', package='aegis.cli')",
            "import importlib\nimportlib.import_module('..execution.base', 'aegis.store')",
            "def later():\n    from aegis.execution import Executor\n    return Executor",
            "try:\n    import aegis.execution\nexcept ImportError:\n    pass",
            "if True:\n    from aegis import execution",
        ],
    )
    def test_catches_every_import_form(self, snippet):
        found = imported_modules(snippet, package="aegis.cli")
        assert imports(found, EXECUTION), snippet
        assert COMPUTED_IMPORT not in found, snippet  # seen by name, not by suspicion

    @pytest.mark.parametrize(
        "snippet, package",
        [
            ("from .. import execution", "aegis.cli"),
            ("from ..execution import Executor", "aegis.cli"),
            ("from ..execution.base import Executor", "aegis.store"),
            ("from . import execution", "aegis"),  # aegis/config.py's own package
            ("from .execution import Executor", "aegis"),
            ("from ... import execution", "aegis.brain.prompts"),
            ("from ...execution.base import Executor", "aegis.brain.prompts"),
        ],
    )
    def test_catches_relative_imports(self, snippet, package):
        assert imports(imported_modules(snippet, package=package), EXECUTION), snippet

    @pytest.mark.parametrize(
        "snippet",
        [
            "import importlib\nimportlib.import_module(name)",
            "import importlib\nname = 'aegis.execu' + 'tion'\nimportlib.import_module(name)",
            "import importlib\nimportlib.import_module('aegis.' + suffix)",
            "import importlib\nimportlib.import_module(f'aegis.{suffix}')",
            "import importlib\nimportlib.import_module(*args)",
            "import importlib\nimportlib.import_module()",
            "import importlib\nimportlib.import_module('.execution', package)",
            "import importlib\nimportlib.import_module('.execution')",
            "__import__(''.join(['aegis', '.execution']))",
            "__import__('execution', globals(), locals(), [], 2)",
            "__import__('execution', level=1)",
            "__import__('aegis', fromlist=names)",
            "__import__('aegis', fromlist=[name])",
            "load = __import__",
            "import importlib\nload = importlib.import_module\nload('aegis.' + 'execution')",
            "from importlib import import_module\nlist(map(import_module, names))",
            "from importlib import import_module as load\nload(name)",
        ],
    )
    def test_a_computed_import_counts_as_importing_anything(self, snippet):
        found = imported_modules(snippet, package="aegis.cli")
        assert COMPUTED_IMPORT in found, snippet
        assert imports(found, EXECUTION) and imports(found, "anthropic"), snippet

    @pytest.mark.parametrize(
        "snippet",
        [
            '"""No module outside aegis.policy may import aegis.execution."""',
            "# import aegis.execution",
            "from aegis.store import insert_proposal",
            "from aegis.policy.engine import evaluate",
            "import aegis.executionary",
            "from aegis import executions",
            "from . import execution",  # resolves to aegis.cli.execution, not aegis.execution
            "from .execution import run",
            "import importlib\nimportlib.import_module('aegis.store')",
            "import importlib\nimportlib.import_module('.db', 'aegis.store')",
            "__import__('json')",
            "__import__('aegis', fromlist=['store'])",
            "__import__('aegis.store', globals(), locals(), [], 0)",
            "note = 'nothing here touches aegis.execution'",  # rule (c), not an import
            "import_module_name = 'x'",  # a look-alike name is not the function
        ],
    )
    def test_does_not_flag_mentions_or_look_alikes(self, snippet):
        found = imported_modules(snippet, package="aegis.cli")
        assert not imports(found, EXECUTION), snippet
        assert COMPUTED_IMPORT not in found, snippet

    def test_what_each_import_statement_yields(self):
        assert imported_modules("import a.b") == {"a.b"}
        assert imported_modules("import a.b as c, d") == {"a.b", "d"}
        assert imported_modules("from a.b import c, d as e") == {"a.b", "a.b.c", "a.b.d"}
        assert imported_modules("from . import c", package="a.b") == {"a.b", "a.b.c"}
        assert imported_modules("from ..x import c", package="a.b") == {"a.x", "a.x.c"}
        assert imported_modules("import_module('a.b')") == {"a.b"}
        assert imported_modules("__import__('a', fromlist=['b', 'c'])") == {"a", "a.b", "a.c"}
        assert imported_modules("x = 1") == set()

    def test_package_resolution(self):
        assert _package_of(AEGIS_DIR / "cli" / "policy.py") == "aegis.cli"
        assert _package_of(AEGIS_DIR / "cli" / "__init__.py") == "aegis.cli"
        assert _package_of(AEGIS_DIR / "config.py") == "aegis"
        assert _package_of(AEGIS_DIR / "__init__.py") == "aegis"
        assert _package_of(AEGIS_DIR / "brain" / "prompts" / "__init__.py") == "aegis.brain.prompts"
        assert _module_name(AEGIS_DIR / "cli" / "policy.py") == "aegis.cli.policy"
        assert _module_name(AEGIS_DIR / "__init__.py") == "aegis"
        assert _module_name(AEGIS_DIR / "store" / "__init__.py") == "aegis.store"
        assert _resolve("execution", 1, "aegis") == "aegis.execution"
        assert _resolve(None, 2, "aegis.cli") == "aegis"
        assert _resolve("execution", 2, "aegis.cli") == "aegis.execution"
        assert _resolve("execution", 2, "aegis.brain.prompts") == "aegis.brain.execution"
        assert _resolve("os", 0, "aegis.cli") == "os"

    def test_folding_reads_only_literal_text(self):
        def fold(expression):
            return _folded(ast.parse(expression, mode="eval").body)

        assert fold("'aegis.execution'") == "aegis.execution"
        assert fold("'aegis.' + 'exec' + 'ution'") == "aegis.execution"
        assert fold("f'aegis.{\"execution\"}'") == "aegis.execution"
        assert fold("'aegis.' 'execution'") == "aegis.execution"
        assert fold("'aegis.' + name") is None and fold("f'aegis.{name}'") is None
        assert fold("f'{\"x\"!r}'") is None and fold("'a' * 2") is None
        assert fold("b'aegis'") is None and fold("1") is None and _folded(None) is None


class TestExecutorNameChecker:
    @pytest.mark.parametrize(
        "snippet, names",
        [
            ("handle = Executor", {"Executor"}),
            ("order = Executor().submit_order(approved)", {"Executor"}),
            ("x: Executor = build()", {"Executor"}),
            ("def run(venue: Executor) -> None: ...", {"Executor"}),
            ("handle = base.Executor", {"Executor"}),
            ("handle = aegis.execution.base.Executor", {"Executor", "execution"}),
            ("venue = PaperExecutor()", {"PaperExecutor"}),
            ("venue = brokers.LiveExecutor(config)", {"LiveExecutor"}),
            ("factory = ExecutorFactory()", {"ExecutorFactory"}),
            ("from concurrent.futures import Executor", {"Executor"}),
            ("from somewhere import Executor as Venue", {"Executor"}),
            ("from somewhere import Venue as PaperExecutor", {"PaperExecutor"}),
            ("from somewhere.Executor import make", {"Executor"}),
            ("import Executor", {"Executor"}),
            ("import brokers.PaperExecutor.live as live", {"PaperExecutor"}),
            ("import brokers as BrokerExecutor", {"BrokerExecutor"}),
            ("def make_Executor(): ...", {"make_Executor"}),
            ("async def PaperExecutor(): ...", {"PaperExecutor"}),
            ("class PaperExecutor: ...", {"PaperExecutor"}),
            ("class Paper(Executor): ...", {"Executor"}),
            ("def run(Executor): ...", {"Executor"}),
            ("def run(*, PaperExecutor=None): ...", {"PaperExecutor"}),
            ("def run(*Executor, **LiveExecutor): ...", {"Executor", "LiveExecutor"}),
            ("run = lambda Executor: None", {"Executor"}),
            ("run(Executor=venue)", {"Executor"}),
            ("try:\n    pass\nexcept ValueError as Executor:\n    pass", {"Executor"}),
            ("match venue:\n    case Executor():\n        pass", {"Executor"}),
            ("match venue:\n    case object() as Executor:\n        pass", {"Executor"}),
            ("match venue:\n    case [*Executor]:\n        pass", {"Executor"}),
            ("match venue:\n    case {**Executor}:\n        pass", {"Executor"}),
            ("match venue:\n    case Venue(Executor=1):\n        pass", {"Executor"}),
            ("def run():\n    global Executor", {"Executor"}),
            ("def outer():\n    def inner():\n        nonlocal PaperExecutor", {"PaperExecutor"}),
            ("type Executor = int", {"Executor"}),
            ("def run[Executor](venue): ...", {"Executor"}),
            ("class Box[*PaperExecutor]: ...", {"PaperExecutor"}),
            ("def run[**LiveExecutor](venue): ...", {"LiveExecutor"}),
            ("for Executor in venues: pass", {"Executor"}),
            ("with open_venue() as Executor: pass", {"Executor"}),
            ("venues = [Executor for Executor in found]", {"Executor"}),
            ("def later():\n    return globals()['x'].PaperExecutor", {"PaperExecutor"}),
            # a pool executor beside a real one: only the real one is reported
            ("pool = ThreadPoolExecutor()\nvenue = PaperExecutor()", {"PaperExecutor"}),
            ("class MyThreadPoolExecutor: ...", {"MyThreadPoolExecutor"}),
        ],
    )
    def test_catches_every_place_a_name_can_stand(self, snippet, names):
        assert executor_names(snippet) == names, snippet

    @pytest.mark.parametrize(
        "snippet, names",
        [
            # what an instance is conventionally kept in: a variable, a parameter, an attribute
            ("executor = build()", {"executor"}),
            ("self.executor.shutdown()", {"executor"}),
            ("def _send(executor, order):\n    return executor.submit_order(order)", {"executor"}),
            ("engine.executor.submit_order(order)", {"executor"}),
            ("held = engine.executor", {"executor"}),
            ("EXECUTOR = build()", {"EXECUTOR"}),
            ("self._executor = venue", {"_executor"}),
            ("paper_executor: object = build()", {"paper_executor"}),
            ("def make_executor(): ...", {"make_executor"}),
            ("def run(*, executor=None): ...", {"executor"}),
            ("run(executor=venue)", {"executor"}),
            ("for executor in venues: pass", {"executor"}),
            ("from aegis.policy.engine import executor", {"executor"}),
            ("from aegis.policy.engine import venue as paper_executor", {"paper_executor"}),
            ("from aegis.policy import executors", {"executors"}),
            ("import brokers.executor.live as live", {"executor"}),
            ("class eXeCuToR: ...", {"eXeCuToR"}),
            # the four names let through are let through exactly as spelled
            ("pool = threadpoolexecutor()", {"threadpoolexecutor"}),
            ("loop.Run_In_Executor(None, work)", {"Run_In_Executor"}),
            ("def run_in_executor_too(): ...", {"run_in_executor_too"}),
            ("venue = executor\nloop.run_in_executor(None, work)", {"executor"}),
        ],
    )
    def test_catches_the_name_in_any_case(self, snippet, names):
        """Rule (b) is case-insensitive: an outside module handed an
        executor — a parameter, an attribute of a policy module — names it,
        whatever the capitals."""
        assert executor_names(snippet) == names, snippet

    @pytest.mark.parametrize(
        "snippet, names",
        [
            ("venue = execution", {"execution"}),
            ("venue = engine.execution", {"execution"}),
            ("base = engine.execution.base", {"execution"}),
            ("del execution", {"execution"}),
            ("self.execution = venue", {"execution"}),
            ("def run(execution):\n    return execution.base", {"execution"}),
            ("from aegis.policy.engine import execution", {"execution"}),
            ("from aegis.policy.engine import execution, executor", {"execution", "executor"}),
            ("from aegis.policy.engine import execution as venue", {"execution"}),
            ("from aegis.policy.engine import venue as execution", {"execution"}),
            ("import brokers as execution", {"execution"}),
            # rule (a)'s findings too: the package named in an import
            ("from aegis import execution", {"execution"}),
            ("import aegis.execution", {"execution"}),
            ("from aegis.execution import base", {"execution"}),
            ("from aegis.execution.base import Executor", {"execution", "Executor"}),
            ("from . import execution", {"execution"}),  # any package's: the bare name is enough
        ],
    )
    def test_catches_the_bare_name_of_the_package(self, snippet, names):
        """Rule (b) reads ``execution`` wherever it is a Name, an Attribute
        or an imported name: once a policy module holds the package, that is
        how an outside module takes it from there — no import of it, no
        ``Executor``, no string."""
        assert executor_names(snippet) == names, snippet
        # none of them is rule (c)'s or rule (f)'s to read
        assert executor_literals(snippet) == set() and order_methods(snippet) == set(), snippet

    @pytest.mark.parametrize(
        "snippet",
        [
            '"""Only aegis.policy may reference an Executor."""',
            'def f():\n    """Hands nothing to an Executor."""\n    return 1',
            "# an Executor in a comment",
            "note = 'the Executor lives in aegis.execution'",  # rule (c), not a name
            "note = 'the executor'\nkind = 'execution'",  # strings: not identifiers
            "loop.run_in_executor(None, work)",
            "async def later():\n    await loop.run_in_executor(pool, work)",
            "from concurrent.futures import ThreadPoolExecutor",
            "from concurrent.futures import ProcessPoolExecutor as Pool",
            "pool = concurrent.futures.InterpreterPoolExecutor()",
            "with ThreadPoolExecutor(max_workers=2) as pool:\n    pool.map(work, items)",
            "execute = True\nexecuted = [order for order in orders]",
            "executes = run.executions",
            # ``execution``: the bare, lower-case identifier only ...
            "class Execution: ...",
            "from aegis.store.models import Execution",
            "row = models.Execution(order_id=1)",
            "def execution_report(): ...",
            "report = order.execution_report",
            "import aegis.executionary",
            # ... and only where it is read, taken as an attribute or imported
            "def execution(): ...",
            "def run(execution=None): ...",
            "run(execution=venue)",
        ],
    )
    def test_does_not_flag_look_alikes_or_the_pool_executors(self, snippet):
        assert executor_names(snippet) == set(), snippet

    def test_the_pool_executors_are_exactly_the_stdlibs(self):
        import asyncio
        import concurrent.futures

        assert POOL_EXECUTORS == {
            "ThreadPoolExecutor", "ProcessPoolExecutor", "InterpreterPoolExecutor",
        }
        for name in POOL_EXECUTORS:
            assert issubclass(getattr(concurrent.futures, name), concurrent.futures.Executor)
        # the abstract base itself is not let through: it is spelled exactly like ours
        assert executor_names("from concurrent.futures import Executor") == {"Executor"}
        # the fourth name let through hands a callable to one of those pools
        assert NOT_AN_EXECUTOR == POOL_EXECUTORS | {"run_in_executor"}
        assert callable(asyncio.AbstractEventLoop.run_in_executor)
        for name in NOT_AN_EXECUTOR:  # each is let through as spelled, and no other way
            assert executor_names(f"value = owner.{name}") == set()
            assert executor_names(f"value = owner.{name.lower()}x") == {f"{name.lower()}x"}
            assert executor_names(f"value = owner.{name.upper()}") == {name.upper()}

    def test_the_bare_name_is_the_packages_own(self):
        assert EXECUTION_NAME == "execution" and EXECUTION == f"{ROOT}.{EXECUTION_NAME}"
        assert EXECUTION_DIR.name == EXECUTION_NAME


class TestExecutorLiteralChecker:
    @pytest.mark.parametrize(
        "snippet",
        [
            "p = 'aegis.execution'",
            "p = 'aegis.execution.base'",
            "p = 'aegis.execution:Executor'",
            "p = 'aegis/execution/base.py'",
            "p = 'aegis\\\\execution\\\\base.py'",
            "p = ROOT / 'execution/base.py'",
            "p = '..\\\\execution\\\\base.py'",
            "p = 'AEGIS.EXECUTION'",
            "note = 'nothing here touches aegis.execution: see the README'",
            "f(f'{root}/aegis/execution/{name}')",
            "def later():\n    return 'aegis.execution'",
            "p = b'aegis/execution/base.py'",
            "p = rb'aegis\\execution'",
            "pickle.loads(b'caegis.execution.base\\nExecutor\\n.')",
            # the class by name alone
            "name = 'Executor'",
            "name = 'PaperExecutor'",
            "name = b'Executor'",
            "venue = getattr(module, 'Executor')",
            "__all__ = ['Executor']",
            "def run(venue: 'Executor') -> None: ...",
            "label = f'{kind}Executor'",
            "pkgutil.resolve_name('somewhere:PaperExecutor')",
            "message = 'handed to the Executor'",
            "pool = 'ThreadPoolExecutor'",  # a string has no pool-executor exemption
            # built from parts
            "p = 'aegis.' + 'execution'",
            "p = 'aegis' + '.' + 'exec' + 'ution'",
            "name = 'Exec' + 'utor'",
            "p = f'aegis.{\"execution\"}'",
            "p = f'{root}/execution'",
            "p = f'aegis{sep}execution'",
            "p = 'aegis.' 'execution'",
        ],
    )
    def test_catches_execution_paths_and_executor_names(self, snippet):
        assert executor_literals(snippet), snippet

    @pytest.mark.parametrize(
        "snippet",
        [
            '"""Only aegis.policy may import aegis.execution or name an Executor."""',
            'def f():\n    """Nothing here touches aegis.execution."""\n    return 1',
            'X = 1\n"""An attribute docstring naming aegis/execution/base.py and Executor."""',
            "# 'aegis.execution' and 'Executor' in a comment",
            "text = 'zero execution authority'",
            "text = 'One (possibly partial) execution of an order.'",
            "p = 'aegis.store'",
            "p = 'aegis/store/repo.py'",
            "blob = b'zero execution authority'",
            "text = 'the executor pool'",  # lower case: not the class
            "text = 'auto-execute tier'",
            "text = 'executed ' + 'orders'",
            "text = f'{count} orders executed'",
            "total = price + fees",  # a BinOp that folds to nothing
        ],
    )
    def test_docstrings_and_plain_words_are_fine(self, snippet):
        assert executor_literals(snippet) == set(), snippet


class TestOrderMethodChecker:
    @pytest.mark.parametrize(
        "snippet, methods",
        [
            # called on whatever was handed over — no ``Executor`` anywhere in sight
            ("venue.submit_order(order)", {"submit_order"}),
            ("def _send(venue, order):\n    return venue.submit_order(order)", {"submit_order"}),
            ("engine.venue.cancel_order(order_id)", {"cancel_order"}),
            ("venue.close_position('SPY')", {"close_position"}),
            ("build().submit_order(order)", {"submit_order"}),
            ("def later():\n    return venue.close_position", {"close_position"}),
            # the method as a value, a bare name, a definition, an argument, a keyword
            ("send = venue.submit_order", {"submit_order"}),
            ("submit_order(order)", {"submit_order"}),
            ("def submit_order(order): ...", {"submit_order"}),
            ("async def cancel_order(order_id): ...", {"cancel_order"}),
            ("class close_position: ...", {"close_position"}),
            ("def run(submit_order): ...", {"submit_order"}),
            ("run(cancel_order=None)", {"cancel_order"}),
            ("send = lambda close_position: None", {"close_position"}),
            # imported
            ("from aegis.policy.engine import submit_order", {"submit_order"}),
            ("from somewhere import send as submit_order", {"submit_order"}),
            ("from somewhere import cancel_order as stop", {"cancel_order"}),
            ("import brokers.close_position", {"close_position"}),
            # in a string: asked for by name, built from parts, or merely mentioned
            ("send = getattr(venue, 'submit_order')", {"submit_order"}),
            ("send = getattr(venue, 'submit_' + 'order')", {"submit_order"}),
            ("send = getattr(venue, f'submit_{\"order\"}')", {"submit_order"}),
            ("send = getattr(venue, 'submit_' 'order')", {"submit_order"}),
            ("stop = operator.methodcaller('cancel_order', order_id)", {"cancel_order"}),
            ("name = b'close_position'", {"close_position"}),
            ("message = 'then call submit_order on it'", {"submit_order"}),
            ("name = 'submit_orders'", {"submit_order"}),  # a string is read for what it contains
            ("__all__ = ['submit_order', 'cancel_order']", {"submit_order", "cancel_order"}),
            # all three at once
            ("venue.submit_order(order)\nvenue.cancel_order(order_id)\n"
             "venue.close_position(symbol)", set(ORDER_METHODS)),
        ],
    )
    def test_catches_every_way_an_order_method_is_named(self, snippet, methods):
        assert order_methods(snippet) == methods, snippet

    @pytest.mark.parametrize(
        "snippet",
        [
            '"""Never submit_order, cancel_order or close_position: the engine alone."""',
            'def f():\n    """Hands nothing to submit_order."""\n    return 1',
            "# venue.submit_order(order)",
            # the interface's fourth method is the store's function too: not on the list
            "rows = repo.get_open_orders(conn)",
            "from aegis.store.repo import get_open_orders",
            "name = 'get_open_orders'",
            # an identifier is read whole
            "orders = submit_orders(batch)",
            "venue.cancel_orders()",
            "book = close_positions(book)",
            "def _submit_order_report(): ...",
            "def submit(order): ...",
            "order = submit\nposition = close",
            "venue.Submit_Order(order)",
            # plain words, and a name nobody can read off the source
            "text = 'submit an order'",
            "text = 'cancel' + ' order'",
            "send = getattr(venue, 'submit_' + suffix)",
        ],
    )
    def test_does_not_flag_look_alikes_or_the_stores_function(self, snippet):
        assert order_methods(snippet) == set(), snippet

    def test_the_list_is_the_interfaces_order_methods(self):
        """Rule (f) reads for what the Executor interface really declares:
        every method of the abstract class is on the list but the one the
        store shares — so a method Phase 6 adds fails here until it is put
        on the list, or named beside ``get_open_orders``."""

        def functions(nodes):
            kinds = (ast.FunctionDef, ast.AsyncFunctionDef)
            return {node.name for node in nodes if isinstance(node, kinds)}

        base = _source(EXECUTION_DIR / "base.py")
        interface = [
            node
            for node in ast.walk(ast.parse(base))
            if isinstance(node, ast.ClassDef) and node.name == EXECUTOR
        ]
        assert len(interface) == 1
        declared = functions(interface[0].body)
        assert ORDER_METHODS == {"submit_order", "cancel_order", "close_position"}
        assert ORDER_METHODS <= declared
        assert declared - ORDER_METHODS == {"get_open_orders"}
        # ... which the store really defines, so no outside module could be told not to name it
        store = ast.parse(_source(AEGIS_DIR / "store" / "repo.py"))
        assert "get_open_orders" in functions(ast.walk(store))
        assert not ORDER_METHODS & functions(ast.walk(store))


class TestExecutionAttributeChecker:
    BASE = f"{EXECUTION}.base"
    ANY = f"aegis.{COMPUTED_ATTRIBUTE}"
    # takes the package off ``aegis``, then the class out of its ``__all__``
    HOLDER = (
        "import aegis\n\n_VENUE = aegis.execution\n_BASE = getattr(_VENUE, _VENUE.__all__[0])\n"
    )

    @pytest.mark.parametrize(
        "snippet, paths",
        [
            # the package, off whatever binds ``aegis``
            ("import aegis\nvenue = aegis.execution", {EXECUTION}),
            ("import aegis.store\nvenue = aegis.execution", {EXECUTION}),
            ("import aegis.cli.db\nvenue = aegis.execution", {EXECUTION}),
            ("import os, aegis\nvenue = aegis.execution", {EXECUTION}),
            ("import aegis as a\nvenue = a.execution", {EXECUTION}),
            ("import aegis\nbase = aegis.execution.base", {EXECUTION, BASE}),
            ("import aegis\nvenue = aegis.execution\nvenue.base", {EXECUTION, BASE}),
            ("import aegis\ndef later():\n    return aegis.execution", {EXECUTION}),
            ("def later():\n    import aegis\n    return aegis.execution", {EXECUTION}),
            ("import aegis\nrun(aegis.execution)", {EXECUTION}),
            ("import aegis\naegis.execution = fake", {EXECUTION}),  # planting one: a reference too
            ("import aegis\ndel aegis.execution", {EXECUTION}),
            # the same attribute, asked for by name
            ("import aegis\nvenue = getattr(aegis, 'execution')", {EXECUTION}),
            ("import aegis\nvenue = getattr(aegis, 'execution', None)", {EXECUTION}),
            ("import aegis\nvenue = getattr(aegis, 'exec' + 'ution')", {EXECUTION}),
            ("import aegis\nvenue = builtins.getattr(aegis, 'execution')", {EXECUTION}),
            ("import aegis\nloaded = hasattr(aegis, 'execution')", {EXECUTION}),
            ("import aegis\nsetattr(aegis, 'execution', fake)", {EXECUTION}),
            ("import aegis\ndelattr(aegis, 'execution')", {EXECUTION}),
            ("import aegis\nbase = getattr(aegis.execution, 'base')", {EXECUTION, BASE}),
            # an alias made by assignment — before or after its use, in any scope
            ("import aegis\npkg = aegis\nvenue = pkg.execution", {EXECUTION}),
            ("import aegis\na = aegis\nb = a\nc = b\nvenue = c.execution", {EXECUTION}),
            ("import aegis\nvenue = c.execution\nc = b\nb = a\na = aegis", {EXECUTION}),
            ("import aegis\na = b = aegis\nvenue = b.execution", {EXECUTION}),
            ("import aegis\npkg: object = aegis\nvenue = pkg.execution", {EXECUTION}),
            ("import aegis\nif (pkg := aegis):\n    venue = pkg.execution", {EXECUTION}),
            ("import aegis\nvenue = (pkg := aegis).execution", {EXECUTION}),
            ("import aegis\ndef a():\n    global pkg\n    pkg = aegis\n"
             "def b():\n    return pkg.execution", {EXECUTION}),
            # the package by another road
            ("import sys\nvenue = sys.modules['aegis'].execution", {EXECUTION}),
            ("import sys\nvenue = sys.modules.get('aegis').execution", {EXECUTION}),
            ("from sys import modules\nvenue = modules['aegis'].execution", {EXECUTION}),
            ("from sys import modules as loaded\nvenue = loaded['aegis'].execution", {EXECUTION}),
            ("import sys as s\nvenue = getattr(s.modules['aegis'], 'execution')", {EXECUTION}),
            ("import sys\npkg = sys.modules['aegis']\nvenue = pkg.execution", {EXECUTION}),
            ("import sys\nvenue = sys.modules['aegis.execution']", {EXECUTION}),
            ("venue = __import__('aegis').execution", {EXECUTION}),
            ("venue = __import__('aegis.store').execution", {EXECUTION}),  # returns ``aegis``
            ("venue = __import__('aegis', fromlist=()).execution", {EXECUTION}),
            ("import importlib\nvenue = importlib.import_module('aegis').execution", {EXECUTION}),
            ("import importlib\nvenue = importlib.import_module('..', 'aegis.cli').execution",
             {EXECUTION}),
            ("from importlib import import_module as load\nvenue = load('aegis').execution",
             {EXECUTION}),
            ("import importlib\npkg = importlib.import_module('aegis')\nvenue = pkg.execution",
             {EXECUTION}),
            # below a binding rule (a) already flags
            ("from aegis import execution as ex\nbase = ex.base", {BASE}),
            ("import aegis.execution as ex\nbase = ex.base", {BASE}),
            ("import aegis.execution\nbase = aegis.execution.base", {EXECUTION, BASE}),
            # every loaded submodule at once
            ("from aegis import *", {"aegis.*"}),
            ("def later():\n    from aegis import *", {"aegis.*"}),
            ("from aegis.execution import *", {f"{EXECUTION}.*"}),
            # an attribute nobody spelled, of the package that holds the execution package
            ("import aegis\nvenue = getattr(aegis, name)", {ANY}),
            ("import aegis\nvenue = getattr(aegis, 'exec' + suffix)", {ANY}),
            ("import aegis\nvenue = vars(aegis)['execution']", {ANY}),
            ("import aegis\nvenue = vars(aegis).get(name)", {ANY, f"{ANY}.get"}),
            ("import aegis\nvenue = aegis.__dict__['execution']", {ANY}),
            ("import aegis\nspace = aegis.__dict__", {ANY}),
            ("import sys\nvenue = getattr(sys.modules['aegis'], name)", {ANY}),
            ("venue = getattr(__import__('aegis'), name)", {ANY}),
            ("import aegis\nbase = getattr(aegis, name).base", {ANY, f"{ANY}.base"}),
            (HOLDER, {EXECUTION, f"{EXECUTION}.__all__", f"{EXECUTION}.{COMPUTED_ATTRIBUTE}"}),
        ],
    )
    def test_catches_every_attribute_path_to_the_package(self, snippet, paths):
        assert execution_attributes(snippet, package="aegis.cli") == paths, snippet

    @pytest.mark.parametrize(
        "snippet, package, paths",
        [
            ("from .. import execution as ex\nbase = ex.base", "aegis.cli", {BASE}),
            ("from . import execution\nbase = execution.base", "aegis", {BASE}),
            ("from .. import *", "aegis.cli", {"aegis.*"}),
            ("from . import *", "aegis", {"aegis.*"}),  # aegis/config.py's own package
            ("from ... import *", "aegis.brain.prompts", {"aegis.*"}),
            ("from ..execution import *", "aegis.store", {f"{EXECUTION}.*"}),
            ("import importlib\nvenue = importlib.import_module('..', __name__).execution",
             "aegis.cli", set()),  # a computed anchor: rule (a) reports the computed import
        ],
    )
    def test_resolves_relative_bindings(self, snippet, package, paths):
        assert execution_attributes(snippet, package=package) == paths, snippet

    @pytest.mark.parametrize(
        "snippet",
        [
            '"""Once loaded, aegis.execution is an attribute of the aegis package."""',
            "# venue = aegis.execution",
            "note = 'aegis.execution'",  # rule (c), not an attribute
            "import aegis.execution",  # rule (a): an import, and no attribute is taken
            "import aegis\nversion = aegis.__version__",
            "import aegis\nstore = aegis.store",
            "import aegis.store\nrows = aegis.store.repo.get_open_orders(conn)",
            "import aegis\nvalue = aegis.executions",
            "import aegis\nvalue = aegis.execution_report",
            "import aegis\nvalue = getattr(aegis, 'executions')",
            "from aegis import store\nvalue = store.execution",  # aegis.store.execution
            "from aegis import config\nvalue = getattr(config, name)",
            "import aegis.config\nvalue = getattr(aegis.config, name, None)",
            "import aegis.store\nspace = vars(aegis.store)",
            "from aegis.store import *",
            "from aegis.policy import *",
            "from . import execution\nexecution.run()",  # aegis.cli.execution, not aegis.execution
            "from . import *",
            "order.execution",  # nothing binds ``order`` to a module
            "self.execution = execution",
            "aegis.execution",  # nor ``aegis``, here: a NameError, not a reference
            "import other\nvalue = other.execution",
            "import other as aegis\nvalue = aegis.execution",
            "import sys\nvalue = sys.modules['aegis.store'].execution",
            "value = __import__('aegis.store', fromlist=['repo']).execution",
            "import importlib\nvalue = importlib.import_module('aegis.store').execution",
            "value = cache.get('aegis').execution",
            "from aegis.policy import engine\ndecision = engine.evaluate(proposal, context, conn)",
            "node = tree\nnode = node.parent",  # feeds itself: bounded, and reaches nothing
        ],
    )
    def test_does_not_flag_other_attributes_or_other_packages(self, snippet):
        assert execution_attributes(snippet, package="aegis.cli") == set(), snippet

    def test_the_three_older_rules_do_not_see_what_this_one_does(self):
        """Why rule (d) exists: no import of the package, no ``Executor``
        identifier, no telling string — and yet the module ends up holding
        the package, and the class through it. Spelled with a dot the
        attribute is rule (b)'s too (the bare identifier ``execution``);
        asked for by name it is this rule's alone."""
        source = self.HOLDER
        assert not imports(imported_modules(source, package="aegis.cli"), EXECUTION)
        assert executor_names(source) == {EXECUTION_NAME}
        assert executor_literals(source) == set() and order_methods(source) == set()
        assert EXECUTION in execution_attributes(source, package="aegis.cli")

        by_name = source.replace("aegis.execution", "getattr(aegis, 'execution')")
        assert by_name != source
        assert not imports(imported_modules(by_name, package="aegis.cli"), EXECUTION)
        assert executor_names(by_name) == set() and executor_literals(by_name) == set()
        assert order_methods(by_name) == set()
        assert EXECUTION in execution_attributes(by_name, package="aegis.cli")

    def test_a_name_stands_for_everything_it_is_ever_bound_to(self):
        def bound(source):
            return _module_bindings(ast.parse(source), "aegis.cli")

        assert bound("import a.b") == {"a": {"a"}}
        assert bound("import a.b as c, d") == {"c": {"a.b"}, "d": {"d"}}
        assert bound("from a.b import c, d as e") == {"c": {"a.b.c"}, "e": {"a.b.d"}}
        assert bound("from .. import store\nfrom . import db") == {
            "store": {"aegis.store"}, "db": {"aegis.cli.db"},
        }
        assert bound("from a import *") == {}
        assert bound("import a\nimport b as a\nx = a") == {"a": {"a", "b"}, "x": {"a", "b"}}
        assert bound("import a\nx = a.b\ny: object = x.c\nz = (w := y)") == {
            "a": {"a"}, "x": {"a.b"}, "y": {"a.b.c"}, "w": {"a.b.c"}, "z": {"a.b.c"},
        }
        assert bound("x = 1\ny = x\nz: int") == {}  # nothing here is bound to a path
        # a name that feeds itself grows once per pass, and the passes are bounded
        assert bound("import a\nx = a\nx = x.up") == {"a": {"a"}, "x": {"a", "a.up", "a.up.up"}}

    def test_what_an_import_call_returns(self):
        def returned(expression):
            return _returned_module(ast.parse(expression, mode="eval").body)

        assert returned("import_module('a.b')") == "a.b"
        assert returned("import_module('.b', 'a')") == "a.b"
        assert returned("import_module('..', package='a.b.c')") == "a.b"
        assert returned("__import__('a')") == "a"
        assert returned("__import__('a.b')") == "a"
        assert returned("__import__('a.b', globals(), locals(), [], 0)") == "a"
        assert returned("__import__('a.b', fromlist=['c'])") == "a.b"
        assert returned("__import__('a.b', fromlist=names)") == "a.b"
        assert returned("import_module(name)") is None
        assert returned("import_module('.b', package)") is None
        assert returned("__import__('b', level=1)") is None

    def test_reaching_is_being_at_or_below_the_target_or_computed_above_it(self):
        assert _reaches("aegis.execution", EXECUTION)
        assert _reaches("aegis.execution.base.Anything", EXECUTION)
        assert _reaches(f"aegis.{COMPUTED_ATTRIBUTE}", EXECUTION)
        assert _reaches(f"aegis.{COMPUTED_ATTRIBUTE}.base", EXECUTION)
        assert _reaches(f"aegis.execution.{COMPUTED_ATTRIBUTE}", EXECUTION)
        assert not _reaches("aegis", EXECUTION) and not _reaches("aegis.executions", EXECUTION)
        assert not _reaches(f"aegis.store.{COMPUTED_ATTRIBUTE}", EXECUTION)
        assert not _reaches(f"other.{COMPUTED_ATTRIBUTE}", EXECUTION)
        assert not _reaches("", EXECUTION) and not _reaches(f".{COMPUTED_ATTRIBUTE}", EXECUTION)


def _asked_by_name(source: str) -> str:
    """``source`` with every ``.execution`` taken by name instead —
    ``x.__getattribute__('execution')`` — so that no identifier spells it:
    the same reference, minus what rule (b) reads."""
    return source.replace(".execution", ".__getattribute__('execution')")


class TestRootPackageChecker:
    ENTRY = f"{MODULE_TABLE}['aegis']"
    TAIL = "HELD = getattr(VENUE, VENUE.__all__[0])\n"

    @pytest.mark.parametrize(
        "snippet, handles",
        [
            # an import that binds the package
            ("import aegis", {"import aegis"}),
            ("import aegis.store", {"import aegis.store"}),  # binds ``aegis`` too
            ("import aegis.cli.db", {"import aegis.cli.db"}),
            ("import os, aegis", {"import aegis"}),
            ("import aegis as a", {"import aegis as a"}),
            ("import aegis.execution", {"import aegis.execution"}),
            ("def later():\n    import aegis\n    return aegis", {"import aegis"}),
            ("try:\n    import aegis.store\nexcept ImportError:\n    pass", {"import aegis.store"}),
            # its entry in the module table
            ("import sys\npkg = sys.modules['aegis']", {ENTRY}),
            ("import sys\npkg = sys.modules.get('aegis')", {ENTRY}),
            ("import sys\npkg = sys.modules.get('aegis', None)", {ENTRY}),
            ("import sys\npkg = sys.modules['ae' + 'gis']", {ENTRY}),
            ("import sys\nvenue = sys.modules['aegis'].execution", {ENTRY}),
            ("import sys as s\npkg = s.modules['aegis']", {ENTRY}),
            ("from sys import modules\npkg = modules['aegis']", {ENTRY}),
            ("from sys import modules as loaded\npkg = loaded.get('aegis')", {ENTRY}),
            ("import os\npkg = os.sys.modules['aegis']", {ENTRY}),
            ("import sys\npkg = getattr(sys, 'modules')['aegis']", {ENTRY}),
            ("import sys\nsys.modules['aegis'] = fake", {ENTRY}),
            # a lookup nobody can read off the source: any module, so this one
            ("import sys\nvalue = sys.modules[name].execution", {COMPUTED_MODULE}),
            ("import sys\npkg = sys.modules[__name__.partition('.')[0]]", {COMPUTED_MODULE}),
            ("import sys\nfrom aegis import store\n"
             "pkg = sys.modules[store.__package__.partition('.')[0]]", {COMPUTED_MODULE}),
            ("import sys\npkg = sys.modules.get(name)", {COMPUTED_MODULE}),
            ("import sys\npkg = sys.modules.get(*args)", {COMPUTED_MODULE}),
            ("import sys\npkg = sys.modules['aegis' + suffix]", {COMPUTED_MODULE}),
            ("from sys import modules as loaded\npkg = loaded[name]", {COMPUTED_MODULE}),
            ("import sys\ndel sys.modules[name]", {COMPUTED_MODULE}),
            # the table handed on instead of looked up
            ("import sys\ntable = sys.modules", {COMPUTED_MODULE}),
            ("import sys\ntable = sys.modules\npkg = table['aegis']", {COMPUTED_MODULE, ENTRY}),
            ("import sys\nrun(sys.modules)", {COMPUTED_MODULE}),
            ("import sys\nloaded = dict(sys.modules)", {COMPUTED_MODULE}),
            ("import sys\nfor name, module in sys.modules.items():\n    pass", {COMPUTED_MODULE}),
            ("import sys\nfor name in sys.modules:\n    pass", {COMPUTED_MODULE}),
            ("import sys\npkg = sys.modules.pop('aegis')", {COMPUTED_MODULE}),
            ("import sys\npkg = sys.modules.__getitem__('aegis')", {COMPUTED_MODULE}),
            ("import sys\npkg = sys.modules.setdefault('aegis', fake)", {COMPUTED_MODULE}),
            # ... or taken off ``sys`` by a name nobody spelled
            ("import sys\nspace = vars(sys)", {COMPUTED_MODULE}),
            ("import sys\ntable = getattr(sys, name)", {COMPUTED_MODULE}),
            ("import sys\npkg = sys.__dict__['modules']['aegis']", {COMPUTED_MODULE}),
            # an attribute called ``.modules`` on something the source does not bind to
            # ``sys``: read as the table all the same, and labelled as what was read
            ("class Registry:\n    def __init__(self):\n        self.modules = []",
             {ATTRIBUTE_MODULES}),
            ("class Registry:\n    def get(self, name):\n        return self.modules[name]",
             {ATTRIBUTE_MODULES}),
            ("found = registry.modules.get(name)", {ATTRIBUTE_MODULES}),
            ("run(registry.modules)", {ATTRIBUTE_MODULES}),
            ("import os\npkg = os.sys.modules[name]", {ATTRIBUTE_MODULES}),
            ("from aegis.cli import trace\npkg = trace.sys.modules[name]", {ATTRIBUTE_MODULES}),
            ("pkg = self.modules['aegis']", {ENTRY}),  # a literal key says which entry
            ("import sys\npkg = sys.modules[name]\nown = self.modules[name]",
             {COMPUTED_MODULE, ATTRIBUTE_MODULES}),
            # an import call that returns it
            ("pkg = __import__('aegis')", {"__import__() returns aegis"}),
            ("pkg = __import__('aegis.store')", {"__import__() returns aegis"}),
            ("pkg = __import__('aegis.store', globals(), locals(), [], 0)",
             {"__import__() returns aegis"}),
            ("import importlib\npkg = importlib.import_module('aegis')",
             {"import_module() returns aegis"}),
            ("import importlib\npkg = importlib.import_module('..', 'aegis.cli')",
             {"import_module() returns aegis"}),
            ("from importlib import import_module as load\npkg = load('aegis')",
             {"load() returns aegis"}),
            # its namespace, or a method bound to it, imported by name
            ("from aegis import __dict__", {"from aegis import __dict__"}),
            ("from aegis import __dict__ as space", {"from aegis import __dict__"}),
            ("from aegis import store, __dict__", {"from aegis import __dict__"}),
            ("from aegis import __getattribute__ as take",
             {"from aegis import __getattribute__"}),
            ("from aegis import __version__", {"from aegis import __version__"}),
            # everything a module that may hold an Executor holds, unnamed
            ("from aegis.policy.engine import *", {"from aegis.policy.engine import *"}),
            ("from aegis.policy import *", {"from aegis.policy import *"}),
            ("from aegis.execution import *", {"from aegis.execution import *"}),
            ("def later():\n    from aegis.policy.engine import *",
             {"from aegis.policy.engine import *"}),
        ],
    )
    def test_catches_every_handle_on_the_package(self, snippet, handles):
        assert root_handles(snippet, package="aegis.cli", module="aegis.cli.trace") == handles

    @pytest.mark.parametrize(
        "snippet, package, handles",
        [
            ("from .. import __dict__", "aegis.cli", {"from aegis import __dict__"}),
            ("from . import __dict__ as space", "aegis", {"from aegis import __dict__"}),
            ("from ... import __dict__", "aegis.brain.prompts", {"from aegis import __dict__"}),
            ("from ..policy.engine import *", "aegis.cli", {"from aegis.policy.engine import *"}),
            ("from .policy import *", "aegis", {"from aegis.policy import *"}),
            ("from . import __dict__", "aegis.cli", set()),  # aegis.cli's, not the package's
            ("from .policy import *", "aegis.cli", set()),  # aegis.cli.policy: the CLI
        ],
    )
    def test_resolves_relative_imports(self, snippet, package, handles):
        assert root_handles(snippet, package=package) == handles, snippet

    @pytest.mark.parametrize(
        "body",
        [
            "def _venue(package):\n    return package.execution\n\n\nVENUE = _venue(aegis)\n",
            "def _venue(package=aegis):\n    return package.execution\n\n\nVENUE = _venue()\n",
            "def _root():\n    return aegis\n\n\nVENUE = _root().execution\n",
            "package, _ = aegis, None\nVENUE = package.execution\n",
            "for package in (aegis,):\n    VENUE = package.execution\n",
            "class _App:\n    package = aegis\n\n\nVENUE = _App.package.execution\n",
            "class _App:\n    def __init__(self):\n        self.package = aegis\n\n"
            "    def venue(self):\n        return self.package.execution\n\n\n"
            "VENUE = _App().venue()\n",
            "PACKAGES = {'root': aegis}\nVENUE = PACKAGES['root'].execution\n",
            "VENUE = (aegis if aegis else None).execution\n",
            "VENUE = (lambda package: package.execution)(aegis)\n",
            "VENUE = globals()['aegis'].execution\n",
            "import operator\n\nVENUE = operator.attrgetter('execution')(aegis)\n",
            "VENUE = aegis.__getattribute__('execution')\n",
        ],
        ids=[
            "parameter", "parameter-default", "return-value", "tuple-unpacking", "for-target",
            "class-attribute", "instance-attribute", "dict-value", "conditional", "lambda",
            "globals", "attrgetter", "getattribute",
        ],
    )
    def test_however_the_package_is_handed_on_it_was_bound_first(self, body):
        """Why rule (e) exists: each of these takes ``.execution`` off the
        package by a road rule (d) does not follow — and (a), (c) and (f)
        have nothing to read. Rule (b) reads the attribute where a dot
        spells it, and nothing once it is asked for by name. None of them
        can start without a binding of ``aegis``, and that is what this rule
        reads."""
        source = "import aegis\n\n" + body + self.TAIL
        by_name = _asked_by_name(source)
        for spelled in (source, by_name):
            assert not imports(imported_modules(spelled, package="aegis.cli"), EXECUTION)
            assert executor_literals(spelled) == set() and order_methods(spelled) == set()
            assert execution_attributes(spelled, package="aegis.cli") == set()
            assert root_handles(spelled, package="aegis.cli") == {"import aegis"}
        dotted = ".execution" in source
        assert executor_names(source) == ({EXECUTION_NAME} if dotted else set())
        assert (by_name != source) == dotted and executor_names(by_name) == set()

    @pytest.mark.parametrize(
        "source, handles",
        [
            ("import sys\n\nVENUE = sys.modules[__name__.partition('.')[0]].execution\n",
             {COMPUTED_MODULE}),
            ("import sys\n\nfrom aegis import store\n\n"
             "VENUE = sys.modules[store.__package__.partition('.')[0]].execution\n",
             {COMPUTED_MODULE}),
            ("from aegis import __dict__ as space\n\nVENUE = space['execution']\n",
             {"from aegis import __dict__"}),
            ("from aegis.policy.engine import *\n\n"
             "HELD = [v for k, v in dict(globals()).items() if k.endswith('utor')][0]\n",
             {"from aegis.policy.engine import *"}),
        ],
        ids=["computed-key", "computed-key-off-a-submodule", "dunder", "star-from-policy"],
    )
    def test_the_package_without_a_binding_of_its_name(self, source, handles):
        by_name = _asked_by_name(source)
        for spelled in (source, by_name):
            assert not imports(imported_modules(spelled, package="aegis.cli"), EXECUTION)
            assert executor_literals(spelled) == set() and order_methods(spelled) == set()
            assert execution_attributes(spelled, package="aegis.cli") == set()
            assert root_handles(spelled, package="aegis.cli") == handles
        # rule (b) reads the attribute where a dot spells it — and nothing asked for by name
        dotted = ".execution" in source
        assert executor_names(source) == ({EXECUTION_NAME} if dotted else set())
        assert (by_name != source) == dotted and executor_names(by_name) == set()

    @pytest.mark.parametrize(
        "snippet, handles",
        [
            ("def _venue():\n    venue = execution\n    return getattr(venue, venue.__all__[0])",
             {"execution"}),
            ("venue = execution.base", {"execution"}),
            ("del execution", {"execution"}),
            ("space = globals()", {"globals()"}),
            ("space = vars()", {"vars()"}),
            ("def later():\n    return locals()", {"locals()"}),
            ("venue = globals()[name]", {"globals()"}),
        ],
    )
    def test_in_the_packages_own_init_its_globals_are_the_package(self, snippet, handles):
        """``aegis/__init__.py`` needs no handle: once anything has loaded
        the execution package, ``execution`` is a global of that module."""
        assert root_handles(snippet, package="aegis", module="aegis") == handles, snippet
        # anywhere else the same text names a global of some other module
        assert root_handles(snippet, package="aegis", module="aegis.config") == set(), snippet
        assert root_handles(snippet, package="aegis.cli", module="aegis.cli.trace") == set()
        # and rule (d) reads none of it
        assert execution_attributes(snippet, package="aegis") == set(), snippet

    @pytest.mark.parametrize(
        "snippet",
        [
            '"""No module outside aegis.policy may import aegis or read sys.modules."""',
            "# import aegis",
            "note = 'import aegis'",
            # below the package: a submodule, or a name out of one
            "from aegis import store",
            "from aegis import config, store",
            "from aegis.store import repo",
            "from aegis.config import get_config",
            "import aegis.store as store",
            "import aegis.cli.db as db",
            "from aegis.policy import engine",
            "from aegis.policy.engine import evaluate",
            "from aegis.store import *",
            "from aegis.store import __doc__",
            "from other import __dict__",
            # other packages
            "import aegisx",
            "import other.aegis",
            "import other as aegis",
            "pkg = __import__('aegis.store', fromlist=['repo'])",
            "import importlib\nstore = importlib.import_module('aegis.store')",
            "import importlib\nstore = importlib.import_module(name)",  # rule (a)'s to report
            # the module table, read for something that is not the package
            "import sys\nstore = sys.modules['aegis.store']",
            "import sys\nstore = sys.modules.get('aegis.store')",
            "import sys\nloaded = 'aegis' in sys.modules",
            "import sys\nmissing = name not in sys.modules",
            "import sys\nfrozen = getattr(sys, 'frozen', False)",
            "import sys\npath = sys.path[0]",
            "value = cache.get('aegis')",
            "modules = load()\nfirst = modules['aegis']",  # a local that happens to be called so
            # the package's namespace is only this module's in aegis/__init__.py
            "space = globals()",
            "execution = report.execution",
        ],
    )
    def test_does_not_flag_what_sits_below_the_package_or_beside_it(self, snippet):
        assert root_handles(snippet, package="aegis.cli", module="aegis.cli.trace") == set()

    def test_a_namespace_call_with_an_argument_is_not_the_packages(self):
        assert root_handles("space = vars(config)", package="aegis", module="aegis") == set()
        assert root_handles("space = vars(**options)", package="aegis", module="aegis") == set()

    def test_what_a_policy_module_holds_is_beyond_a_static_rule(self):
        """The limit, pinned so nobody takes the silence for coverage: an
        outside module may import a policy module, and reading that module's
        namespace finds whatever it holds. No static rule can tell — the
        runtime check of who HOLDS an Executor names the module that keeps
        what it found (``TestWhoHoldsAnExecutor``)."""
        source = (
            "from aegis.policy import engine\n\n"
            "HELD = [v for k, v in vars(engine).items() if k.endswith('utor')][0]\n"
        )
        assert not imports(imported_modules(source, package="aegis.cli"), EXECUTION)
        assert root_handles(source, package="aegis.cli") == set()
        assert execution_attributes(source, package="aegis.cli") == set()
        assert executor_names(source) == set() and executor_literals(source) == set()
        assert order_methods(source) == set()

    @pytest.mark.parametrize(
        "path, handles",
        [(AEGIS_DIR / "__init__.py", {"execution"}), (AEGIS_DIR / "config.py", set())],
        ids=["the-package", "a-module-beside-it"],
    )
    def test_a_file_of_the_tree_is_read_as_the_module_it_is(self, monkeypatch, path, handles):
        """What the tree rule calls: the bare name ``execution`` is a handle
        in ``aegis/__init__.py`` — whose globals are the package's namespace
        — and nowhere else, so the rule must hand each file's module name to
        the checker. Dropping it reads the package's own file as any other."""
        monkeypatch.setitem(globals(), "_source", lambda path: "venue = execution\n")
        assert _handles_in(path) == handles
        assert root_handles("venue = execution\n", package=_package_of(path)) == set()


# --- PART 1: the rules --------------------------------------------------------


class TestOnlyPolicyMayReferenceAnExecutor:
    def test_the_scan_covers_everything_but_the_two_packages(self):
        everything = set(_sources(AEGIS_DIR))
        outside, policy, execution = (
            set(OUTSIDE_SOURCES), set(POLICY_SOURCES), set(_sources(EXECUTION_DIR)),
        )
        # every file is in exactly one of the three, and none is skipped
        assert outside | policy | execution == everything
        assert not (outside & policy or outside & execution or policy & execution)
        assert policy and execution and len(outside) > len(policy)

        top_level = {path.relative_to(AEGIS_DIR).parts[0] for path in OUTSIDE_SOURCES}
        assert top_level == EXPECTED_OUTSIDE, top_level ^ EXPECTED_OUTSIDE
        packages = {
            entry.name for entry in AEGIS_DIR.iterdir() if entry.is_dir() and _sources(entry)
        }
        assert packages == (EXPECTED_OUTSIDE - {"__init__.py", "config.py"}) | {
            "policy", "execution",
        }
        # the policy CLI is outside the engine: it is scanned like every other module
        assert CLI_POLICY.exists() and CLI_POLICY in outside
        assert {AEGIS_DIR / "__init__.py", AEGIS_DIR / "config.py"} <= outside
        for package in ("data", "pricing", "store", "brain", "cli", "notify", "dashboard"):
            assert AEGIS_DIR / package / "__init__.py" in outside, package
        assert AEGIS_DIR / "brain" / "prompts" / "__init__.py" in outside  # nested packages too

    def test_the_rules_have_a_real_target(self):
        """The execution package exists, defines ``Executor`` and its order
        methods, and trips the checkers itself — so a clean scan elsewhere
        means something."""
        base = EXECUTION_DIR / "base.py"
        classes = {
            node.name
            for node in ast.walk(ast.parse(_source(base)))
            if isinstance(node, ast.ClassDef)
        }
        assert EXECUTOR in classes
        init = _source(EXECUTION_DIR / "__init__.py")
        assert imports(imported_modules(init, package=EXECUTION), EXECUTION)
        assert executor_names(init) == {EXECUTOR, EXECUTION_NAME}  # the class, and its package
        assert executor_literals(init) == {EXECUTOR}  # ``__all__``
        assert executor_names(_source(base)) == {EXECUTOR}
        assert order_methods(_source(base)) == ORDER_METHODS and order_methods(init) == set()

    @pytest.mark.parametrize("path", OUTSIDE_SOURCES, ids=_id)
    def test_no_module_outside_policy_imports_execution(self, path):
        found = imported_modules(_source(path), package=_package_of(path))
        assert not imports(found, EXECUTION), f"{_id(path)} imports {EXECUTION}"

    @pytest.mark.parametrize("path", OUTSIDE_SOURCES, ids=_id)
    def test_no_module_outside_policy_names_an_executor(self, path):
        assert executor_names(_source(path)) == set(), _id(path)

    @pytest.mark.parametrize("path", OUTSIDE_SOURCES, ids=_id)
    def test_no_module_outside_policy_spells_one_in_a_string(self, path):
        assert executor_literals(_source(path)) == set(), _id(path)

    @pytest.mark.parametrize("path", OUTSIDE_SOURCES, ids=_id)
    def test_no_module_outside_policy_reaches_execution_through_an_attribute(self, path):
        found = execution_attributes(_source(path), package=_package_of(path))
        assert found == set(), f"{_id(path)} reaches {sorted(found)}"

    @pytest.mark.parametrize("path", OUTSIDE_SOURCES, ids=_id)
    def test_no_module_outside_policy_holds_the_aegis_package_itself(self, path):
        found = _handles_in(path)
        hint = (
            " — an attribute called .modules is read as sys.modules: rename it"
            if ATTRIBUTE_MODULES in found
            else ""
        )
        assert found == set(), f"{_id(path)} holds {sorted(found)}{hint}"

    @pytest.mark.parametrize("path", OUTSIDE_SOURCES, ids=_id)
    def test_no_module_outside_policy_names_an_order_method(self, path):
        found = order_methods(_source(path))
        assert found == set(), f"{_id(path)} names {sorted(found)}"

    def test_the_package_rule_reads_a_real_binding(self):
        """Rule (e) is not silent because it reads nothing: this very file —
        a test, outside ``aegis/`` — binds the package with ``import
        aegis.config``, and the checker says so. And the one module whose
        globals are the package's namespace is scanned as that module."""
        assert root_handles(_source(Path(__file__))) == {"import aegis.config"}
        assert _handles_in(Path(__file__)) == {"import aegis.config"}
        assert _module_name(AEGIS_DIR / "__init__.py") == ROOT
        assert AEGIS_DIR / "__init__.py" in OUTSIDE_SOURCES

    def test_the_package_rule_itself_reads_each_file_as_the_module_it_is(self, monkeypatch):
        """The RULE, not only the helper it calls: handed a source that takes
        the package's attribute by its bare name, the tree rule fails for
        ``aegis/__init__.py`` — read as the package, whose globals that name
        lives in — and passes for a module beside it. A rule that stopped
        passing each file's module name to the checker would pass both."""
        rule = self.test_no_module_outside_policy_holds_the_aegis_package_itself
        monkeypatch.setitem(globals(), "_source", lambda path: "venue = execution\n")
        with pytest.raises(AssertionError, match=r"holds \['execution'\]") as failure:
            rule(AEGIS_DIR / "__init__.py")
        assert str(failure.value).splitlines()[0] == (
            f"{_id(AEGIS_DIR / '__init__.py')} holds ['execution']"
        )
        rule(AEGIS_DIR / "config.py")  # the bare name is nothing there: no failure
        rule(CLI_POLICY)

    def test_the_package_rule_says_what_an_attribute_called_modules_was_read_as(
        self, monkeypatch
    ):
        """The hint in the rule's message: honest code with an attribute
        called ``.modules`` fails closed, and is told why and what to do —
        and only that finding carries the hint."""
        rule = self.test_no_module_outside_policy_holds_the_aegis_package_itself
        registry = "class Registry:\n    def __init__(self):\n        self.modules = []\n"
        monkeypatch.setitem(globals(), "_source", lambda path: registry)
        with pytest.raises(AssertionError, match="rename it") as failure:
            rule(AEGIS_DIR / "config.py")
        assert str(failure.value).splitlines()[0] == (
            f"{_id(AEGIS_DIR / 'config.py')} holds ['{ATTRIBUTE_MODULES}'] — an attribute"
            " called .modules is read as sys.modules: rename it"
        )
        monkeypatch.setitem(globals(), "_source", lambda path: "import aegis\n")
        with pytest.raises(AssertionError, match=r"holds \['import aegis'\]") as failure:
            rule(AEGIS_DIR / "config.py")
        assert "rename it" not in str(failure.value)

    def test_the_attribute_rule_follows_the_trees_own_bindings(self):
        """Rule (d) is not silent because it resolves nothing: pointed at a
        module the policy CLI really takes names from, it reads the chains
        the CLI writes on them."""
        found = execution_attributes(
            _source(CLI_POLICY), package=_package_of(CLI_POLICY), target="aegis.store.models"
        )
        assert "aegis.store.models.EventLevel.WARNING" in found


# --- the runtime backstops ----------------------------------------------------

# Runs in a fresh interpreter. ``argv[1]`` is a JSON job: ``root`` (put first
# on sys.path), then either ``walk`` — a package directory under root whose
# every module is imported by walking the tree, minus those under an
# ``exclude``d package — or ``modules``, a list imported as given; and
# ``extra``, code the self-tests run last. Two checks, each optional — the
# second in two halves, who ASKS for a tracked module and who HOLDS one:
#
# - ``forbidden``: module prefixes that must not be in ``sys.modules``
#   afterwards (reported as ``loaded``);
# - ``track``: module prefixes whose every import is charged to the module that
#   asked for it — the nearest calling frame, outside the import machinery and
#   this harness, that belongs to the package ``tree`` (with no such frame on
#   the stack, the nearest frame of any other module). ``importers`` maps each
#   one to what it asked for, and ``rogue`` keeps those not under an
#   ``allowed`` package. Four hooks see the asking: ``builtins.__import__``
#   (every import statement, whether or not the module is already loaded),
#   ``importlib.__import__`` and ``importlib.import_module`` (the same, by
#   call), and a meta-path finder for a first load that arrives by any other
#   road. Asking for a submodule is asking for the tracked packages above it:
#   they are charged to the asker too — whether this import loaded them or
#   they were there already, so the charge does not depend on who came first
#   — except the package the asker itself lives in. A tracked module that is
#   loaded although no hook saw another module ask for it is charged to
#   ``<unattributed>``: nobody's permission covers it. What a module asks for
#   itself does not count — a package's ``__init__`` asking for its own
#   submodule says nothing about who loaded the package. ``reached`` lists
#   the tracked modules loaded.
#
#   Then, whatever was asked for and by whom: ``holders`` maps every loaded
#   module under ``tree`` and not under an ``allowed`` package to the names of
#   its module-level values that are — or hold, one level into a list, tuple,
#   set or dict value — a module under a tracked prefix (by its ``__name__``,
#   registered or not), or a class, function or instance whose ``__module__``
#   (its own, or its type's) is under one. A package's own loaded submodule
#   is not a holding: ``sys.modules[package + "." + attribute]`` is that
#   very value. This half asks nothing of the import system, so it does not
#   depend on the road a reference took or on who loaded the package first.
#   What it cannot see is a value that is neither a module-level value nor an
#   item of such a list, tuple, set, frozenset or dict: one kept in a
#   function's local, a default argument or a closure, as an attribute of a
#   class or an object (``BOX.held`` — a module-level name leads to it, and
#   the check still does not follow), in any other container, as a dict's
#   key, or two containers deep.
#
# A module that fails to import is reported, never skipped: an unimportable
# module is an unchecked one.
_IMPORT_SCRIPT = r"""
import builtins, importlib, importlib.util, json, sys
from pathlib import Path

job = json.loads(sys.argv[1])
root = Path(job["root"])
sys.path.insert(0, str(root))
tracked, tree, allowed = job.get("track", []), job.get("tree", ""), job.get("allowed", [])


def within(name, prefixes):
    return any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes)


MACHINERY = ("importlib", "_frozen_importlib", "_frozen_importlib_external")
HARNESS = globals()
importers = {}


def charge(names, frame):
    names = [name for name in names if within(name, tracked)]
    nearest = None
    while names and frame is not None:
        module = frame.f_globals.get("__name__") or "<unnamed>"
        if frame.f_globals is not HARNESS and not within(module, MACHINERY):
            if within(module, [tree]):
                nearest = module
                break
            nearest = nearest or module
        frame = frame.f_back
    if names:
        asker = nearest or "<harness>"
        # the packages above a submodule are asked for with it, loaded already
        # or not — all but the asker's own, which it is running inside
        above = {name.rsplit(".", up)[0] for name in names for up in range(1, name.count(".") + 1)}
        mine = {name for name in above if within(name, tracked) and not within(asker, [name])}
        importers.setdefault(asker, set()).update(names, mine)


def absolute(name, globals_, level):
    if not level:
        return name
    globals_ = globals_ or {}
    package = globals_.get("__package__")
    if not isinstance(package, str):
        package = globals_.get("__name__") or ""
        if "__path__" not in globals_:
            package = package.rpartition(".")[0]
    base = package.rsplit(".", level - 1)[0]
    return f"{base}.{name}" if name else base


real_import_module = importlib.import_module


def tracking(real_import):
    def tracking_import(name, globals=None, locals=None, fromlist=(), level=0):
        target, frame = absolute(name, globals, level), sys._getframe(1)
        charge([target], frame)
        module = real_import(name, globals, locals, fromlist, level)
        # ``from package import submodule`` asks for the submodule too
        submodules = [f"{target}.{item}" for item in fromlist or () if isinstance(item, str)]
        charge([name for name in submodules if name in sys.modules], frame)
        return module

    return tracking_import


def tracking_import_module(name, package=None):
    try:
        target = importlib.util.resolve_name(name, package)
    except (ImportError, TypeError, ValueError):
        target = name  # the real call below raises the real error
    charge([target], sys._getframe(1))
    return real_import_module(name, package)


class FirstLoads:
    @staticmethod
    def find_spec(name, path=None, target=None):
        charge([name], sys._getframe(1))
        return None  # never finds anything: the real finders do


def from_tracked(value):
    # a tracked module by its name, anything else by where it says it was defined
    if isinstance(value, type(sys)):
        names = [getattr(value, "__name__", None)]
    else:
        names = [getattr(value, "__module__", None), getattr(type(value), "__module__", None)]
    return any(isinstance(name, str) and within(name, tracked) for name in names)


def held_by(name, module):
    found = []
    for attribute, value in list(vars(module).items()):
        if sys.modules.get(f"{name}.{attribute}") is value:
            continue  # a package's own submodule: it hangs there once anything loads it
        inside = []  # one level into a container
        if isinstance(value, dict):
            inside = list(value.values())
        elif isinstance(value, (list, tuple, set, frozenset)):
            inside = list(value)
        if any(from_tracked(item) for item in [value, *inside]):
            found.append(attribute)
    return sorted(found)


if tracked:
    builtins.__import__ = tracking(builtins.__import__)
    importlib.__import__ = tracking(importlib.__import__)
    importlib.import_module = tracking_import_module
    sys.meta_path.insert(0, FirstLoads)

names = list(job.get("modules", []))
if "walk" in job:
    for path in sorted((root / job["walk"]).rglob("*.py")):
        parts = list(path.relative_to(root).with_suffix("").parts)
        if parts[-1] == "__init__":
            parts.pop()
        name = ".".join(parts)
        if not within(name, job.get("exclude", [])):
            names.append(name)

imported, failed = [], {}
for name in names:
    try:
        importlib.import_module(name)
    except BaseException as exc:
        failed[name] = f"{type(exc).__name__}: {exc}"
    else:
        imported.append(name)
if job.get("extra"):
    exec(job["extra"], {"__name__": "<extra>"})

loaded = sorted(name for name in sys.modules if within(name, job.get("forbidden", [])))
report = {"imported": imported, "failed": failed, "loaded": loaded}
rogue, holders = {}, {}
if tracked:
    for name, module in sorted(sys.modules.items()):
        outside = within(name, [tree]) and not within(name, allowed)
        if outside and isinstance(module, type(sys)):
            held = held_by(name, module)
            if held:
                holders[name] = held
    report["holders"] = holders
    reached = sorted(name for name in sys.modules if within(name, tracked))
    unattributed = {
        name
        for name in reached
        if not any(name in asks for asker, asks in importers.items() if asker != name)
    }
    if unattributed:
        importers["<unattributed>"] = unattributed
    report["reached"] = reached
    report["importers"] = {name: sorted(importers[name]) for name in sorted(importers)}
    rogue = {name: asks for name, asks in report["importers"].items() if not within(name, allowed)}
    report["rogue"] = rogue
print(json.dumps(report))
sys.exit(1 if loaded or failed or rogue or holders else 0)
"""


def _run_imports(**job) -> tuple[int, dict]:
    job.setdefault("root", str(REPO_ROOT))
    # never write bytecode: the harness must leave no __pycache__ behind it
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    result = subprocess.run(
        [sys.executable, "-c", _IMPORT_SCRIPT, json.dumps(job)],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=120, env=env,
    )
    lines = result.stdout.strip().splitlines()
    assert lines, result.stderr
    return result.returncode, json.loads(lines[-1])


def _outside_job(**extra) -> dict:
    return {
        "walk": "aegis",
        "exclude": ["aegis.policy", EXECUTION],
        "track": [EXECUTION],
        "tree": "aegis",
        "allowed": list(MAY_IMPORT_EXECUTION),
        **extra,
    }


def _imported_by(importer: str) -> str:
    """``extra`` code: the import of the Executor, made as the module
    ``importer`` — what ``aegis.policy.engine`` will do itself in Phase 6."""
    return f"exec('from aegis.execution import Executor', {{'__name__': {importer!r}}})\n"


def _toy_tree(root: Path, files: dict[str, str]) -> None:
    for name, source in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")


TOY = {
    "toy/__init__.py": "",
    "toy/data.py": "VALUE = 1\n",
    "toy/cli/__init__.py": "",
    "toy/cli/report.py": "from toy.data import VALUE\n",
    "toy/policy/__init__.py": "",
    # the one package that may — at module level, as the engine will in Phase 6
    "toy/policy/engine.py": "from toy.execution import Executor\n",
    "toy/policy/lazy.py": "def venue():\n    import toy.execution\n    return toy.execution\n",
    "toy/execution/__init__.py": "from toy.execution.base import Executor\n",
    "toy/execution/base.py": "class Executor: ...\n",
}
TOY_JOB = {
    "walk": "toy",
    "exclude": ["toy.policy", "toy.execution"],
    "track": ["toy.execution"],
    "tree": "toy",
    "allowed": ["toy.policy", "toy.execution"],
}
TOY_OUTSIDE = ["toy", "toy.cli", "toy.cli.report", "toy.data"]
TOY_EXECUTION = ["toy.execution", "toy.execution.base"]
# what the execution package's own ``__init__`` asks for, once anything loads it
OWN_IMPORTS = {"toy.execution": ["toy.execution.base"]}
# added to a toy tree, the Phase 6 order: this module is imported before
# ``toy/cli/rogue.py``, so a policy module has loaded the package by then
PHASE_6_ORDER = {"toy/cli/first.py": "import toy.policy.engine\n"}


class TestRuntimeOutsidePolicy:
    def test_only_policy_and_execution_modules_ever_import_the_execution_package(self):
        """Runtime confirmation of the static rules, in a fresh interpreter:
        every module outside the two packages is imported, and every import
        of ``aegis.execution`` that causes is charged to the module that
        asked for it. A computed or otherwise unseen import anywhere outside
        would make that module an importer — and it is not one of the two
        packages that may be.

        The invariant is WHO imports the package, not that nothing loads it:
        ``aegis.cli.policy`` imports the policy engine, and once the engine
        imports the Executor (Phase 6) this very run loads the execution
        package — through ``aegis.policy``, which is allowed.

        And whoever asked, by whatever road: once every outside module is
        imported, none of them HOLDS the package, the Executor or an
        instance of one among its module-level values (``holders``)."""
        code, report = _run_imports(**_outside_job())
        assert report["failed"] == {}
        assert report["rogue"] == {}
        # ... and whatever was imported by whom, no outside module HOLDS an Executor
        assert report["holders"] == {}
        assert all(_within(name, MAY_IMPORT_EXECUTION) for name in report["importers"])
        # the package is loaded exactly when someone was seen asking for it
        assert bool(report["reached"]) == bool(report["importers"])
        assert code == 0
        # the walk in the child saw exactly the files the static rules scanned
        assert sorted(report["imported"]) == sorted(_module_name(p) for p in OUTSIDE_SOURCES)
        assert "aegis.cli.policy" in report["imported"] and "aegis" in report["imported"]
        assert not [name for name in report["imported"] if _within(name, MAY_IMPORT_EXECUTION)]

    @pytest.mark.parametrize(
        "loaded_by", [None, "aegis.policy.engine"], ids=["tree-as-it-is", "loaded-by-policy"]
    )
    @pytest.mark.parametrize(
        "extra, asked",
        [
            ("import aegis.execution", [EXECUTION]),
            ("from aegis import execution", [EXECUTION]),
            # asking for the submodule asks for its package: both are this caller's
            ("import importlib\nimportlib.import_module('aegis.exec' + 'ution.base')",
             [EXECUTION, f"{EXECUTION}.base"]),
            ("import importlib\nimportlib.__import__('aegis.exec' + 'ution')", [EXECUTION]),
        ],
        ids=["import", "from-import", "computed-name", "importlib-dunder-import"],
    )
    def test_the_harness_catches_an_import_in_the_real_tree(self, extra, asked, loaded_by):
        """Self-test: the same run fails once anything outside the two
        packages asks for the package — and says who asked. Twice over: on
        the tree as it is (today the outsider is the first to load the
        package), and with the policy engine having loaded it already, as in
        Phase 6 — when no finder is consulted any more. What the outsider is
        charged with is the same either way, so nothing here is pinned to
        who loaded the package first."""
        if loaded_by:
            extra = _imported_by(loaded_by) + extra
        code, report = _run_imports(**_outside_job(extra=extra))
        assert code == 1 and report["failed"] == {}
        assert report["rogue"] == {"<extra>": asked}
        assert report["holders"] == {}  # the asker is no module of the tree: it is only charged
        # whatever else the package holds by then (Phase 6 adds modules), nothing outside it
        assert {EXECUTION, f"{EXECUTION}.base"} <= set(report["reached"])
        assert all(_within(name, (EXECUTION,)) for name in report["reached"])
        # the package's own import of its submodule is charged to the package: allowed
        assert f"{EXECUTION}.base" in report["importers"][EXECUTION]
        if loaded_by:
            assert EXECUTION in report["importers"][loaded_by]
        allowed = {name for name in report["importers"] if name != "<extra>"}
        assert all(_within(name, MAY_IMPORT_EXECUTION) for name in allowed)

    @pytest.mark.parametrize(
        "importer", ["aegis.policy.engine", "aegis.policy", "aegis.execution.paper"]
    )
    def test_the_same_import_is_allowed_from_the_two_packages_in_the_real_tree(self, importer):
        """Self-test: what Phase 6 will do. The package is loaded during the
        run — which the old "nothing may load it" check would have failed —
        and the run passes, because a policy (or execution) module asked."""
        # ... after which the package hangs off ``aegis``, an outside module: its own
        # submodule, which the check of who holds it must not take for a holding
        hangs = (
            "import sys\n"
            "assert sys.modules['aegis'].execution is sys.modules['aegis.execution']\n"
        )
        extra = _imported_by(importer) + hangs
        code, report = _run_imports(**_outside_job(extra=extra))
        assert report["rogue"] == {} and report["failed"] == {} and report["holders"] == {}
        assert {EXECUTION, f"{EXECUTION}.base"} <= set(report["reached"])
        assert all(_within(name, (EXECUTION,)) for name in report["reached"])
        assert EXECUTION in report["importers"][importer]
        assert code == 0

    def test_a_clean_toy_tree_passes(self, tmp_path):
        """Self-test: the excluded policy package imports execution, and that
        is fine — nothing outside imports either of them."""
        _toy_tree(tmp_path, TOY)
        code, report = _run_imports(root=str(tmp_path), **TOY_JOB)
        assert report == {
            "imported": TOY_OUTSIDE,
            "failed": {},
            "loaded": [],
            "holders": {},
            "reached": [],
            "importers": {},
            "rogue": {},
        }
        assert code == 0

    @pytest.mark.parametrize(
        "source",
        [
            "import toy.execution\n",
            "from toy import execution\n",
            "from ..execution import Executor\n",
            "import importlib\nvenue = importlib.import_module('toy.' + 'exec' + 'ution')\n",
            "name = ''.join(['toy', '.', 'execution'])\n__import__(name)\n",
            "import importlib\nimportlib.__import__('toy.' + 'execution')\n",
        ],
        ids=[
            "import", "from-import", "relative", "import-module", "dunder-import",
            "importlib-dunder-import",
        ],
    )
    def test_an_outside_module_that_imports_execution_is_caught_by_name(self, tmp_path, source):
        """Proof (a): one module outside the two packages asks for the
        execution package at import time — however it spells it — and the
        report names that module."""
        _toy_tree(tmp_path, {**TOY, "toy/cli/rogue.py": source})
        code, report = _run_imports(root=str(tmp_path), **TOY_JOB)
        assert code == 1 and report["failed"] == {}
        assert report["rogue"] == {"toy.cli.rogue": ["toy.execution"]}
        assert report["importers"] == {**OWN_IMPORTS, **report["rogue"]}
        assert report["reached"] == TOY_EXECUTION
        assert "toy.cli.rogue" in report["imported"]

    def test_a_submodule_of_execution_is_caught_too(self, tmp_path):
        source = "from toy.execution.base import Executor\n"
        _toy_tree(tmp_path, {**TOY, "toy/cli/rogue.py": source})
        code, report = _run_imports(root=str(tmp_path), **TOY_JOB)
        assert code == 1
        # asking for the submodule asks for its package: both are charged to the asker
        assert report["rogue"] == {"toy.cli.rogue": ["toy.execution", "toy.execution.base"]}

    @pytest.mark.parametrize(
        "source, importers",
        [
            ("from toy.policy import engine\n", {"toy.policy.engine": ["toy.execution"]}),
            ("import toy.policy.engine\n", {"toy.policy.engine": ["toy.execution"]}),
            ("import importlib\nimportlib.import_module('toy.policy.engine')\n",
             {"toy.policy.engine": ["toy.execution"]}),
            ("from toy.policy.lazy import venue\nvenue()\n",
             {"toy.policy.lazy": ["toy.execution"]}),
        ],
        ids=["from-import", "import", "import-module", "lazy-in-policy"],
    )
    def test_an_outside_module_that_imports_a_policy_module_is_allowed(
        self, tmp_path, source, importers
    ):
        """Proof (b): an outside module imports a policy module, which itself
        imports the execution package — and holds the class it imported. The
        package IS loaded by the run, and the run passes: the importer is the
        policy module, and so is the holder. (Taking the class from there by
        name is another matter: ``TestWhoHoldsAnExecutor``.)"""
        _toy_tree(tmp_path, {**TOY, "toy/cli/policy.py": source})
        code, report = _run_imports(root=str(tmp_path), **TOY_JOB)
        assert report["failed"] == {} and report["rogue"] == {} and report["holders"] == {}
        assert report["importers"] == {**OWN_IMPORTS, **importers}
        assert report["reached"] == TOY_EXECUTION  # loaded — through the package that may
        assert "toy.cli.policy" in report["imported"]
        assert code == 0

    @pytest.mark.parametrize(
        "source, asked",
        [
            ("import toy.execution\n", ["toy.execution"]),
            ("from toy import execution\n", ["toy.execution"]),
            ("from ..execution import Executor\n", ["toy.execution"]),
            ("import importlib\nvenue = importlib.import_module('toy.' + 'exec' + 'ution')\n",
             ["toy.execution"]),
            ("name = ''.join(['toy', '.', 'execution'])\n__import__(name)\n", ["toy.execution"]),
            ("import importlib\nimportlib.__import__('toy.' + 'execution')\n", ["toy.execution"]),
            # a submodule: its package is the asker's too, though nothing had to load it
            ("from toy.execution.base import Executor\n", TOY_EXECUTION),
            ("import importlib\nimportlib.import_module('toy.exec' + 'ution.base')\n",
             TOY_EXECUTION),
        ],
        ids=[
            "import", "from-import", "relative", "import-module", "dunder-import",
            "importlib-dunder-import", "submodule", "submodule-import-module",
        ],
    )
    def test_execution_already_loaded_through_policy_does_not_hide_a_direct_import(
        self, tmp_path, source, asked
    ):
        """Proofs (a) and (b) together — the Phase 6 order: by the time the
        rogue module runs, the package is already in ``sys.modules`` (a policy
        module loaded it), so no finder is consulted — the asking itself is
        what is seen, and it is charged exactly as on a first load."""
        files = {
            **TOY,
            "toy/cli/first.py": "import toy.policy.engine\n",  # imported before rogue.py
            "toy/cli/rogue.py": source,
        }
        _toy_tree(tmp_path, files)
        code, report = _run_imports(root=str(tmp_path), **TOY_JOB)
        assert report["imported"].index("toy.cli.first") < report["imported"].index(
            "toy.cli.rogue"
        )
        assert code == 1 and report["failed"] == {}
        assert report["rogue"] == {"toy.cli.rogue": asked}
        assert report["importers"] == {
            **OWN_IMPORTS,
            "toy.policy.engine": ["toy.execution"],
            "toy.cli.rogue": asked,
        }
        # the same module in a tree where it is the first to load the package
        del files["toy/cli/first.py"]
        first = tmp_path / "first-load"
        _toy_tree(first, files)
        code, report = _run_imports(root=str(first), **TOY_JOB)
        assert code == 1 and report["rogue"] == {"toy.cli.rogue": asked}

    def test_a_reference_taken_off_an_attribute_is_charged_to_nobody_and_held_all_the_same(
        self, tmp_path
    ):
        """Asking and holding are two checks. Once a policy module has loaded
        the package it hangs off ``toy``, and an outside module takes it from
        there — the package, and the class inside it — without asking the
        import system for anything: nothing to charge. The module still HOLDS
        both, and for that the run fails, naming it. Rule (d) reads the
        attribute in the source as well."""
        source = (
            "import toy\n\n"
            "VENUE = toy.execution\n"
            "HELD = [value for value in vars(VENUE).values() if isinstance(value, type)]\n"
        )
        files = {
            **TOY,
            "toy/cli/first.py": "import toy.policy.engine\n",  # imported before rogue.py
            "toy/cli/rogue.py": source,
        }
        # run last in the child, where a failed assertion fails the whole run:
        # the reference is real — the outside module holds the Executor
        held = (
            "import sys\n"
            "rogue, venue = sys.modules['toy.cli.rogue'], sys.modules['toy.execution']\n"
            "assert rogue.VENUE is venue and rogue.HELD == [venue.Executor], rogue.HELD\n"
        )
        _toy_tree(tmp_path, files)
        code, report = _run_imports(root=str(tmp_path), **{**TOY_JOB, "extra": held})
        assert "toy.cli.rogue" in report["imported"] and report["failed"] == {}
        assert report["importers"] == {**OWN_IMPORTS, "toy.policy.engine": ["toy.execution"]}
        assert report["rogue"] == {}  # no import was made: nothing to charge
        # ... and the module is named for what it holds; ``toy`` itself, which the
        # package hangs off as its own submodule, is not
        assert report["holders"] == {"toy.cli.rogue": ["HELD", "VENUE"]}
        assert code == 1
        # rules (a) and (c) read nothing; (b) the bare name, (d) the attribute path
        assert not imports(imported_modules(source, package="toy.cli"), "toy.execution")
        assert executor_names(source) == {EXECUTION_NAME} and executor_literals(source) == set()
        namespace = f"toy.execution.{COMPUTED_ATTRIBUTE}"
        assert execution_attributes(source, package="toy.cli", target="toy.execution") == {
            "toy.execution", namespace, f"{namespace}.values",
        }

    # run last in the child, where a failed assertion fails the whole run: the
    # outside module really holds the Executor
    HOLDS = (
        "import sys\n"
        "rogue, base = sys.modules['toy.cli.rogue'], sys.modules['toy.execution.base']\n"
        "assert rogue.HELD == [base.Executor], rogue.HELD\n"
    )
    # finds the package's spec and executes it: no import statement, no import call
    SPEC_LOAD = (
        "import importlib.util as u\n"
        "import sys\n\n"
        "spec = u.find_spec(__name__.partition('.')[0] + '.exe' + 'cution')\n"
        "VENUE = u.module_from_spec(spec)\n"
        "{register}"
        "spec.loader.exec_module(VENUE)\n"
        "HELD = [value for value in vars(VENUE).values() if isinstance(value, type)]\n"
    )

    # what the copy a rogue module made leaves behind: it holds the copy and the
    # class in it. A copy REGISTERED over a package a policy module had loaded
    # also leaves ``toy`` with the displaced original, no longer its submodule.
    HOLDING = {"toy.cli.rogue": ["HELD", "VENUE"]}
    DISPLACED = {"toy": ["execution"], **HOLDING}

    @pytest.mark.parametrize(
        "register, handles",
        [("", set()), ("sys.modules[spec.name] = VENUE\n", {COMPUTED_MODULE})],
        ids=["not-registered", "registered"],
    )
    def test_a_load_from_a_spec_is_seen_by_the_finder_alone(self, tmp_path, register, handles):
        """Proof of the fourth hook: an outside module finds the package's
        spec under a computed name and executes it — with or without putting
        the module in ``sys.modules``. No ``__import__`` and no
        ``import_module`` is called for the package; only the meta-path
        finder is consulted, and it names the module that asked. Take the
        finder away and this is charged to nobody by name. (It is a first
        load that the finder sees: once the package is in ``sys.modules``,
        ``find_spec`` answers from there and no finder is asked — the next
        test.) Either way the module is named for what it ends up holding."""
        source = self.SPEC_LOAD.format(register=register)
        _toy_tree(tmp_path, {**TOY, "toy/cli/rogue.py": source})
        code, report = _run_imports(root=str(tmp_path), **{**TOY_JOB, "extra": self.HOLDS})
        assert code == 1 and report["failed"] == {}
        assert report["rogue"] == {"toy.cli.rogue": ["toy.execution"]}
        assert report["holders"] == self.HOLDING
        assert report["reached"] == TOY_EXECUTION
        # rules (a) to (d) and (f) read nothing; (e) only the write to the module table
        assert not imports(imported_modules(source, package="toy.cli"), "toy.execution")
        assert executor_names(source) == set() and executor_literals(source) == set()
        assert order_methods(source) == set()
        assert execution_attributes(source, package="toy.cli", target="toy.execution") == set()
        assert root_handles(source, package="toy.cli") == handles

    @pytest.mark.parametrize(
        "load, register, holders",
        [
            ("SPEC_LOAD", "", HOLDING),
            ("SPEC_LOAD", "sys.modules[spec.name] = VENUE\n", DISPLACED),
            ("FILE_LOAD", "", HOLDING),
            ("FILE_LOAD", "sys.modules[name] = VENUE\n", DISPLACED),
        ],
        ids=["spec", "spec-registered", "file-path", "file-path-registered"],
    )
    def test_a_second_load_asks_nobody_and_its_holder_is_named_all_the_same(
        self, tmp_path, load, register, holders
    ):
        """The Phase 6 order, where the asking goes unseen: a policy module
        has loaded the package before the rogue module runs, so its load
        from a spec (``find_spec`` answers from ``sys.modules``: no finder is
        asked) or from the file path is charged to nobody — and nothing is
        unattributed either, the package having been asked for by the module
        that may. What the rogue module ends up holding does not depend on
        any of that: the copy of the package, and the Executor in it."""
        source = getattr(self, load).format(register=register)
        _toy_tree(tmp_path, {**TOY, **PHASE_6_ORDER, "toy/cli/rogue.py": source})
        code, report = _run_imports(root=str(tmp_path), **{**TOY_JOB, "extra": self.HOLDS})
        assert report["imported"].index("toy.cli.first") < report["imported"].index(
            "toy.cli.rogue"
        )
        assert report["failed"] == {}
        assert report["rogue"] == {}  # the import hooks saw nothing to charge the rogue with
        assert "toy.cli.rogue" not in report["importers"]
        assert report["holders"] == holders
        assert code == 1

    LAZY = "def late():\n    import toy.execution\n    return toy.execution.Executor\n"

    @pytest.mark.parametrize(
        "source, extra",
        [
            (LAZY + "\n\nVENUE = late()\n", None),
            (LAZY, "import toy.cli.rogue\ntoy.cli.rogue.late()\n"),
            ("def late():\n    from toy.policy import engine\n    import toy.execution\n",
             "from toy.cli.rogue import late\nlate()\n"),
        ],
        ids=["called-at-import", "called-later-in-the-run", "after-a-policy-import"],
    )
    def test_a_lazy_import_in_an_outside_module_is_caught_when_it_runs(
        self, tmp_path, source, extra
    ):
        """Proof (c): the import hides inside a function of an outside module.
        Once the function is called during the run — by the module itself or
        by anyone else — the import is charged to the module that wrote it."""
        _toy_tree(tmp_path, {**TOY, "toy/cli/rogue.py": source})
        job = {**TOY_JOB, "extra": extra} if extra else TOY_JOB
        code, report = _run_imports(root=str(tmp_path), **job)
        assert code == 1 and report["failed"] == {}
        assert report["rogue"] == {"toy.cli.rogue": ["toy.execution"]}

    def test_a_lazy_import_that_never_runs_is_the_static_rules_to_catch(self, tmp_path):
        """The runtime backstop sees what runs; rule (a) reads what is written."""
        _toy_tree(tmp_path, {**TOY, "toy/cli/rogue.py": self.LAZY})
        code, report = _run_imports(root=str(tmp_path), **TOY_JOB)
        assert code == 0 and report["rogue"] == {} and report["reached"] == []
        assert imports(imported_modules(self.LAZY, package="toy.cli"), "toy.execution")

    PLANTED = "import sys, types\nsys.modules['toy.execution'] = types.ModuleType('planted')\n"
    # executes the package's ``__init__`` straight from its file: nothing is asked for
    FILE_LOAD = (
        "import importlib.util as u\n"
        "import sys\n"
        "from pathlib import Path\n\n"
        "here = Path(__file__).resolve().parent.parent / ('exe' + 'cution')\n"
        "name = __name__.partition('.')[0] + '.' + here.name\n"
        "spec = u.spec_from_file_location(\n"
        "    name, here / '__init__.py', submodule_search_locations=[str(here)]\n"
        ")\n"
        "VENUE = u.module_from_spec(spec)\n"
        "{register}"
        "spec.loader.exec_module(VENUE)\n"
        "HELD = [value for value in vars(VENUE).values() if isinstance(value, type)]\n"
    )

    @pytest.mark.parametrize(
        "rogue, extra",
        [
            (None, PLANTED),
            (FILE_LOAD.format(register="sys.modules[name] = VENUE\n"), HOLDS),
            (FILE_LOAD.format(register=""), HOLDS),
        ],
        ids=["planted", "file-path-load", "file-path-load-not-registered"],
    )
    def test_a_load_nobody_was_seen_asking_for_is_nobodys_to_allow(self, tmp_path, rogue, extra):
        """Self-test: a module that sits under the package's name although no
        other module asked for it is charged to no allowed importer — one
        planted in ``sys.modules``, or the real package loaded from its file
        path by an outside module, which then holds the Executor. The
        package's own ``__init__`` asks for its submodule (and, when it is
        not registered, for itself) while it runs: that vouches for the
        submodule, never for the package."""
        _toy_tree(tmp_path, {**TOY, "toy/cli/rogue.py": rogue} if rogue else TOY)
        code, report = _run_imports(root=str(tmp_path), **{**TOY_JOB, "extra": extra})
        assert code == 1 and report["failed"] == {}
        assert report["rogue"] == {"<unattributed>": ["toy.execution"]}
        if rogue:
            assert "toy.cli.rogue" in report["imported"]
            assert report["reached"] == TOY_EXECUTION
            assert "toy.execution.base" in report["importers"]["toy.execution"]
            # nobody's to allow — and the module that made the load is named for holding it
            assert report["holders"] == self.HOLDING
        else:
            assert report["holders"] == {}  # planted where no outside module keeps it

    def test_the_harness_reports_a_module_it_could_not_import(self, tmp_path):
        """Self-test: a module that cannot be imported was not checked, so the run fails."""
        _toy_tree(tmp_path, {**TOY, "toy/broken.py": "raise RuntimeError('no')\n"})
        code, report = _run_imports(root=str(tmp_path), **TOY_JOB)
        assert code == 1
        assert report["failed"] == {"toy.broken": "RuntimeError: no"} and report["rogue"] == {}


# The toy tree in its Phase 6 shape: the policy engine holds the package, the
# class and an instance at module level (and one in a container), and hands
# orders to it. Policy may. Nothing outside may take any of it from there.
PHASE_6 = {
    **TOY,
    "toy/execution/base.py": (
        "class Executor:\n    def submit_order(self, order):\n        return ('SENT', order)\n"
    ),
    "toy/policy/engine.py": (
        "from toy import execution\n"
        "from toy.execution import Executor\n\n"
        "executor = Executor()\n"
        "VENUES = {'paper': executor}\n\n\n"
        "class Dry(Executor):\n"
        "    pass\n\n\n"
        "dry = Dry()\n\n\n"
        "def evaluate(order):\n"
        "    return executor.submit_order(order)\n"
    ),
}
PHASE_6_ENGINE = {"toy.policy.engine": ["toy.execution"]}  # who asks for the package there
# the root package under a name nobody spelled, and its attribute asked for by name:
# what rules (b), (d) and (e) would read, written so that they read nothing
TOY_ROOT = "__name__.partition('.')[0]"
BY_NAME = "'exe' + 'cution'"
CLASSES = "HELD = [value for value in vars(VENUE).values() if isinstance(value, type)]\n"
# for a form that takes the package off ``toy``: on a first load something must
# have loaded it, so the module asks a policy module to — which it may
LOADS_IT_FIRST = "from toy.policy import engine as _first\n"


def _static_findings(source: str) -> dict[str, set[str]]:
    """What each of the six static rules reads in a toy module, written as
    it would be in the real tree — keyed by the rule's letter, the silent
    ones left out. Empty when no static rule sees anything."""
    written = source.replace("toy", ROOT)
    package = f"{ROOT}.cli"
    found = {
        "a": {name for name in imported_modules(written, package=package)
              if name == COMPUTED_IMPORT or _within(name, (EXECUTION,))},
        "b": executor_names(written),
        "c": executor_literals(written),
        "d": execution_attributes(written, package=package),
        "e": root_handles(written, package=package, module=f"{package}.rogue"),
        "f": order_methods(written),
    }
    return {rule: findings for rule, findings in found.items() if findings}


class TestWhoHoldsAnExecutor:
    """The second half of the PART 1 backstop: whatever road a reference
    took, and whoever loaded the package first, a module outside the two
    packages that HOLDS the execution package, the Executor or an instance
    of one at module level is named in ``holders`` and fails the run."""

    # run last in the child, where a failed assertion fails the whole run: every
    # name reported really is — or holds — the package, the class or an instance
    REAL = (
        "import sys\n"
        "rogue, venue = sys.modules['toy.cli.rogue'], sys.modules['toy.execution']\n"
        "base = sys.modules['toy.execution.base']\n\n\n"
        "def real(value):\n"
        "    known = value is venue or value is base or value is base.Executor\n"
        "    return known or isinstance(value, base.Executor)\n\n\n"
        "for name in {held!r}:\n"
        "    value = getattr(rogue, name)\n"
        "    inside = list(value.values()) if isinstance(value, dict) else value\n"
        "    inside = inside if isinstance(inside, list) else []\n"
        "    assert real(value) or any(real(item) for item in inside), name\n"
        "sent = getattr(rogue, 'SENT', None)\n"
        "assert sent in (None, ('SENT', 'model output')), sent\n"
    )
    ORDERS = pytest.mark.parametrize("order", ["first-load", "phase-6-order"])

    def _tree(self, order: str, name: str, source: str) -> dict[str, str]:
        earlier = PHASE_6_ORDER if order == "phase-6-order" else {}
        return {**PHASE_6, **earlier, f"toy/cli/{name}.py": source}

    def _loaded_in_order(self, order: str, name: str, report: dict) -> bool:
        """Whether ``toy.cli.<name>`` was imported — after the module that
        has a policy module load the package (the Phase 6 order), or with no
        such module in the tree (a first load)."""
        imported = report["imported"]
        if order == "phase-6-order":
            return imported.index("toy.cli.first") < imported.index(f"toy.cli.{name}")
        return "toy.cli.first" not in imported and f"toy.cli.{name}" in imported

    @ORDERS
    @pytest.mark.parametrize(
        "source, off_the_root, held, static",
        [
            # NEW-1-R1-R1 (A): a library call that returns the root package
            pytest.param(
                "import pkgutil\n\n"
                f"VENUE = getattr(pkgutil.resolve_name({TOY_ROOT}), {BY_NAME})\n" + CLASSES,
                True, ["HELD", "VENUE"], {}, id="resolve-name",
            ),
            # NEW-1-R1-R1 (B): ``sys`` handed to a library call that reads the module table
            pytest.param(
                "import operator\nimport sys\n\n"
                f"_ROOT = operator.attrgetter('modules')(sys)[{TOY_ROOT}]\n"
                f"VENUE = getattr(_ROOT, {BY_NAME})\n" + CLASSES,
                True, ["HELD", "VENUE"], {}, id="sys-handed-on",
            ),
            # NEW-1-R1-R1 (D): the import function taken out of the builtins by a string
            pytest.param(
                f"VENUE = getattr(__builtins__['__imp' + 'ort__']({TOY_ROOT}), {BY_NAME})\n"
                + CLASSES,
                True, ["HELD", "VENUE"], {}, id="import-from-builtins",
            ),
            pytest.param(
                f"VENUE = getattr(eval('__imp' + 'ort__')({TOY_ROOT}), {BY_NAME})\n" + CLASSES,
                True, ["HELD", "VENUE"], {}, id="import-by-eval",
            ),
            # finding 2.2: the package and an instance taken off a policy module by name,
            # and an order sent — rules (b) and (f) read the names now
            pytest.param(
                "from toy.policy.engine import execution, executor\n\n"
                "SENT = executor.submit_order('model output')\n"
                "VENUE = execution\n",
                False, ["VENUE", "execution", "executor"],
                {"b": {"execution", "executor"}, "f": {"submit_order"}},
                id="names-from-a-policy-module",
            ),
            pytest.param(
                "from toy.policy import engine\n\n"
                "HELD = engine.executor\n"
                "BASE = engine.execution.base\n",
                False, ["BASE", "HELD"], {"b": {"execution", "executor"}},
                id="attributes-of-a-policy-module",
            ),
            pytest.param(
                "from toy.policy.engine import Executor as Venue\n",
                False, ["Venue"], {"b": {"Executor"}}, id="the-class-from-a-policy-module",
            ),
            # ... and the same taken by a name no rule can read
            pytest.param(
                "from toy.policy import engine\n\n"
                f"VENUE = getattr(engine, {BY_NAME})\n"
                "HELD = getattr(engine, 'exe' + 'cutor')\n",
                False, ["HELD", "VENUE"], {}, id="asked-of-a-policy-module-by-name",
            ),
            # what a policy module holds, read out of its namespace
            pytest.param(
                "from toy.policy import engine\n\n"
                "HELD = [v for k, v in vars(engine).items() if k.endswith('utor')]\n",
                False, ["HELD"], {}, id="namespace-of-a-policy-module",
            ),
            pytest.param(
                "from toy.policy.engine import evaluate\n\nSPACE = evaluate.__globals__\n",
                False, ["SPACE"], {}, id="globals-of-a-policy-function",
            ),
        ],
    )
    def test_an_outside_module_that_holds_one_is_named_whatever_the_road(
        self, tmp_path, order, source, off_the_root, held, static
    ):
        """Each of these asks the import system for nothing it may not have
        — the tracked package is never imported by the rogue module — so the
        check of who ASKS is silent, and most are written so that no static
        rule reads anything either. The module ends up holding the package,
        the class or an instance all the same, and is named for it: on a
        first load (the module itself is the first to have a policy module
        load the package) and in the Phase 6 order (a policy module loaded
        it earlier in the run)."""
        if off_the_root and order == "first-load":
            source = LOADS_IT_FIRST + source
        _toy_tree(tmp_path, self._tree(order, "rogue", source))
        proof = self.REAL.format(held=held)
        code, report = _run_imports(root=str(tmp_path), **{**TOY_JOB, "extra": proof})
        assert self._loaded_in_order(order, "rogue", report) and report["failed"] == {}
        # who asks: the policy engine alone, which may
        assert report["rogue"] == {}
        assert report["importers"] == {**OWN_IMPORTS, **PHASE_6_ENGINE}
        # who holds: the rogue module, by the names it keeps them under
        assert report["holders"] == {"toy.cli.rogue": held}
        assert code == 1
        assert _static_findings(source) == static

    @ORDERS
    @pytest.mark.parametrize(
        "source",
        [
            "from toy.policy import engine\n",
            "import toy.policy.engine as engine\n",
            "from toy.policy.engine import evaluate\n\nRESULT = evaluate('approved')\n",
            "from toy.policy import engine\n\nRESULT = engine.evaluate('approved')\n"
            "COUNT = len(engine.VENUES)\n",
        ],
        ids=["from-import", "import", "a-policy-function", "calls-into-policy"],
    )
    def test_what_a_policy_module_holds_is_its_own(self, tmp_path, order, source):
        """Allowed, in either order: the policy engine holds the package,
        the class and an instance at module level (and in a container), and
        an outside module that merely imports it — even calls into it, and
        gets an order sent BY the engine — holds none of them."""
        _toy_tree(tmp_path, self._tree(order, "report", source))
        holds = (
            "import sys\n"
            "engine, venue = sys.modules['toy.policy.engine'], sys.modules['toy.execution']\n"
            "assert engine.execution is venue and engine.Executor is venue.Executor\n"
            "assert isinstance(engine.executor, venue.Executor)\n"
            "assert engine.VENUES == {'paper': engine.executor}\n"
            "result = getattr(sys.modules['toy.cli.report'], 'RESULT', None)\n"
            "assert result in (None, ('SENT', 'approved')), result\n"
        )
        code, report = _run_imports(root=str(tmp_path), **{**TOY_JOB, "extra": holds})
        assert self._loaded_in_order(order, "report", report) and report["failed"] == {}
        assert report["reached"] == TOY_EXECUTION  # loaded, and held — by the package that may
        assert report["importers"] == {**OWN_IMPORTS, **PHASE_6_ENGINE}
        assert report["rogue"] == {} and report["holders"] == {}
        assert code == 0
        assert _static_findings(source) == {}

    def test_a_holding_is_read_one_level_into_a_container(self, tmp_path):
        """What "holds" means, pinned on both sides: the value itself, or an
        item of a list, tuple, set or frozenset, or a value of a dict.

        Not a dict's key and not an item two containers deep; not an
        attribute of a class or of an object the module defines, a default
        argument, a closure or a function's local; not any other container
        (a ``functools.partial``, a ``SimpleNamespace``, a ``deque``) — and
        not an instance of a subclass written in a policy module, which says
        it was defined there. Each of the unreported names really leads to
        an Executor (checked in the child): the check is one level deep by
        ruling, and this is where it stops."""
        source = (
            "import collections\nimport functools\nimport types\n\n"
            "from toy.policy import engine\n\n"
            "_held = getattr(engine, 'exe' + 'cutor')\n"
            "ALONE, LIST, TUPLE = _held, [1, _held], (_held, 1)\n"
            "SET, FROZEN, DICT = {_held}, frozenset({_held}), {'paper': _held}\n"
            "KEY, DEEP, DEEPER = {_held: 'paper'}, [[_held]], {'venues': {'paper': _held}}\n"
            "METHOD, KIND = getattr(_held, '_'.join(['submit', 'order'])), type(_held)\n"
            "SUBCLASS = engine.dry\n\n\n"
            "class Box:\n"
            "    held = _held\n\n"
            "    def __init__(self):\n"
            "        self.kept = _held\n\n\n"
            "def send(order, venue=_held):\n"
            "    return venue\n\n\n"
            "def _closing(venue):\n"
            "    return lambda: venue\n\n\n"
            "BOX = Box()\n"
            "CLOSURE = _closing(_held)\n"
            "PARTIAL = functools.partial(send, 'model output', _held)\n"
            "SPACE = types.SimpleNamespace(held=_held)\n"
            "QUEUE = collections.deque([_held])\n"
            "del _held\n\n\n"
            "def late():\n"
            "    local = getattr(engine, 'exe' + 'cutor')\n"
            "    return type(local).__name__\n\n\n"
            "NAME = late()\n"
        )
        # each unreported name really leads to an Executor — or, for the
        # subclass, to an instance of one written in the policy module
        unseen = (
            "import sys\n"
            "rogue, base = sys.modules['toy.cli.rogue'], sys.modules['toy.execution.base']\n"
            "kept = {\n"
            "    'KEY': list(rogue.KEY)[0], 'DEEP': rogue.DEEP[0][0],\n"
            "    'DEEPER': rogue.DEEPER['venues']['paper'], 'SUBCLASS': rogue.SUBCLASS,\n"
            "    'Box': rogue.Box.held, 'BOX': rogue.BOX.kept,\n"
            "    'send': rogue.send.__defaults__[0], 'CLOSURE': rogue.CLOSURE(),\n"
            "    'PARTIAL': rogue.PARTIAL.args[1], 'SPACE': rogue.SPACE.held,\n"
            "    'QUEUE': rogue.QUEUE[0],\n"
            "}\n"
            "for name, value in kept.items():\n"
            "    assert hasattr(rogue, name) and isinstance(value, base.Executor), name\n"
            "assert rogue.PARTIAL() is rogue.BOX.kept and rogue.NAME\n"
        )
        _toy_tree(tmp_path, {**PHASE_6, "toy/cli/rogue.py": source})
        code, report = _run_imports(root=str(tmp_path), **{**TOY_JOB, "extra": unseen})
        assert report["failed"] == {} and report["rogue"] == {}
        assert report["holders"] == {
            "toy.cli.rogue": ["ALONE", "DICT", "FROZEN", "KIND", "LIST", "METHOD", "SET", "TUPLE"],
        }
        assert code == 1
        assert _static_findings(source) == {}

    @pytest.mark.parametrize(
        "holder, held",
        [
            ("aegis.cli.trace", {"aegis.cli.trace": ["HELD", "VENUE"]}),
            ("aegis.cli.policy", {"aegis.cli.policy": ["HELD", "VENUE"]}),
            ("aegis.store.repo", {"aegis.store.repo": ["HELD", "VENUE"]}),
            ("aegis", {"aegis": ["HELD", "VENUE"]}),  # beside its own submodule, which is not one
            ("aegis.policy.engine", {}),
            ("aegis.policy", {}),
        ],
    )
    def test_the_harness_names_a_holder_in_the_real_tree(self, holder, held):
        """Self-test on the real tree, in the Phase 6 order: the policy
        engine has imported the Executor, and a module is then handed the
        package and the class without asking for either. Outside the two
        packages that fails the run, naming the module; inside ``aegis.policy``
        it is what Phase 6 does."""
        hands_over = (
            "import sys\n"
            f"holder, venue = sys.modules[{holder!r}], sys.modules['aegis.execution']\n"
            "holder.VENUE, holder.HELD = venue, [venue.Executor]\n"
        )
        extra = _imported_by("aegis.policy.engine") + hands_over
        code, report = _run_imports(**_outside_job(extra=extra))
        assert report["failed"] == {} and report["rogue"] == {}  # nobody outside asked
        assert report["holders"] == held
        assert code == (1 if held else 0)

    DORMANT = (
        "def _late():\n"
        "    load = __builtins__['__imp' + 'ort__']\n"
        f"    return load({TOY_ROOT} + '.exe' + 'cution', None, None, ['base'])\n"
    )
    LOCAL = (
        "import pkgutil\n\n\n"
        "def _send(order):\n"
        f"    venue = getattr(pkgutil.resolve_name({TOY_ROOT}), {BY_NAME})\n"
        "    held = [value for value in vars(venue).values() if isinstance(value, type)][0]()\n"
        "    return getattr(held, '_'.join(['submit', 'order']))(order)\n\n\n"
        "SENT = _send('model output')\n"
    )

    def test_the_documented_limit_a_dormant_obfuscated_reference(self, tmp_path):
        """The limit, pinned so nobody takes the silence for coverage
        (NEW-1-R1-R1 (C)): a name assembled from string pieces and resolved
        through ``__builtins__`` inside a function nobody calls during the
        run. No static rule reads it, nothing is asked for, nothing is held:
        the run passes. Only once it runs is the asking seen."""
        assert _static_findings(self.DORMANT) == {}
        _toy_tree(tmp_path, {**PHASE_6, "toy/cli/rogue.py": self.DORMANT})
        code, report = _run_imports(root=str(tmp_path), **TOY_JOB)
        assert "toy.cli.rogue" in report["imported"] and report["failed"] == {}
        assert report["rogue"] == {} and report["holders"] == {} and report["reached"] == []
        assert code == 0
        # called — by anyone — it is an import like any other, charged to the module it is in
        called = "from toy.cli.rogue import _late\nassert _late().base.Executor\n"
        code, report = _run_imports(root=str(tmp_path), **{**TOY_JOB, "extra": called})
        assert report["rogue"] == {"toy.cli.rogue": TOY_EXECUTION}
        assert code == 1

    def test_the_documented_limit_a_reference_held_in_a_local(self, tmp_path):
        """The holding check's blind spot, pinned likewise: in the Phase 6
        order an outside module reaches the package by a road no rule reads,
        keeps it in a function's local, and really sends an order. Nothing
        is asked for and no module-level name leads to an Executor: the run
        passes. Deliberate evasion of a tripwire — not something it claims
        to stop."""
        assert _static_findings(self.LOCAL) == {}
        _toy_tree(tmp_path, self._tree("phase-6-order", "rogue", self.LOCAL))
        sent = (
            "import sys\n"
            "assert sys.modules['toy.cli.rogue'].SENT == ('SENT', 'model output')\n"
        )
        code, report = _run_imports(root=str(tmp_path), **{**TOY_JOB, "extra": sent})
        assert "toy.cli.rogue" in report["imported"] and report["failed"] == {}
        assert report["importers"] == {**OWN_IMPORTS, **PHASE_6_ENGINE}
        assert report["rogue"] == {} and report["holders"] == {}
        assert code == 0


# --- PART 2: the checkers -----------------------------------------------------


def forbidden_imports(source: str, *, package: str = "") -> set[str]:
    """What ``source`` imports that no policy module may: anything under
    ``FORBIDDEN_IN_POLICY``, or a computed import (which could be any of them)."""
    found = imported_modules(source, package=package)
    return {
        name for name in found if name == COMPUTED_IMPORT or _within(name, FORBIDDEN_IN_POLICY)
    }


def outside_allowlist(source: str, allowed: frozenset[str], *, package: str = "") -> set[str]:
    """Every import in ``source`` that is not on ``allowed``.

    ``import x`` needs ``x`` itself on the list; ``from m import a`` needs
    ``m`` — or ``m.a``, for a submodule taken from its package (``from
    collections import abc``). A dynamic import is checked by its literal
    target and never passes when computed."""
    tree = ast.parse(source)
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names if alias.name not in allowed)
        elif isinstance(node, ast.ImportFrom):
            base = _resolve(node.module, node.level, package)
            if base in allowed:
                continue
            names = (f"{base}.{alias.name}" if base else alias.name for alias in node.names)
            found.update(name for name in names if name not in allowed)
    found.update(name for name in _dynamic_imports(tree) if name not in allowed)
    return found


def _is_config_type(name: str) -> bool:
    return inspect.isclass(getattr(aegis.config, name, None))


def config_values(source: str, *, package: str = "") -> set[str]:
    """What ``source`` takes from ``aegis.config`` that is not a type: a
    function or constant by name (``get_config``, ``REPO_ROOT``), or the
    module itself, through which all of them are reachable."""
    found: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            found.update(CONFIG for alias in node.names if _within(alias.name, (CONFIG,)))
        elif isinstance(node, ast.ImportFrom):
            base = _resolve(node.module, node.level, package)
            for alias in node.names:
                if f"{base}.{alias.name}" == CONFIG:
                    found.add(CONFIG)
                elif base == CONFIG and not _is_config_type(alias.name):
                    found.add(alias.name)
    return found


def _clock_owners(tree: ast.AST, package: str) -> set[str]:
    """Every name ``datetime`` / ``date`` goes by in a tree: their own, each
    ``from datetime import datetime as dt`` alias, and each name a plain
    assignment hands one of them to (``moment = datetime``)."""
    owners = set(CLOCK_OWNERS)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and _resolve(node.module, node.level, package) == (
            "datetime"
        ):
            owners.update(a.asname or a.name for a in node.names if a.name in CLOCK_OWNERS)
    grew = True
    while grew:
        grew = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                targets, value = node.targets, node.value
            elif isinstance(node, (ast.AnnAssign, ast.NamedExpr)):
                targets, value = [node.target], node.value
            else:
                continue
            if value is None or _terminal_name(value) not in owners:
                continue
            for target in targets:
                if isinstance(target, ast.Name) and target.id not in owners:
                    owners.add(target.id)
                    grew = True
    return owners


def clock_reads(source: str, *, package: str = "") -> set[str]:
    """Every way ``source`` reads a clock, described (empty when clean):
    any reference to ``utcnow`` (imported, called or handed on); a call of
    ``.now()`` / ``.today()`` on anything (or of a bare ``now`` / ``today``);
    a REFERENCE to ``.now`` / ``.today`` / ``.utcnow`` on ``datetime`` or
    ``date`` — under any name the source gives them, spelled out or through
    ``getattr`` — because a clock handed on (``default_factory=datetime.now``,
    ``_now = datetime.now``) is read by whoever calls it; a reference to one
    of those three on anything that is not a bare name (``context.now.now``,
    ``type(x).now``, ``[datetime][0].now`` — the same clock, off an instance
    or an expression); and the ``time``
    module in any form: ``import time``, ``from time import …`` (a star
    included), a call of anything taken from it, ``__import__("time")`` or
    ``import_module("time")`` whatever is then done with it, and an import
    call whose target cannot be read off the source, which could be that
    one. Attribute access on a NAME that is not the datetime classes is not
    a read: ``context.now`` is a value the context was given."""
    tree = ast.parse(source)
    time_modules: set[str] = set()  # local names bound to the time module
    time_functions: set[str] = set()  # local names bound to something taken from it
    owners = _clock_owners(tree, package)
    found: set[str] = set()
    for name in _dynamic_imports(tree):  # ``__import__("time").time()`` names no module
        if name == COMPUTED_IMPORT or _within(name, ("time",)):
            found.add(f"import {name} by a call")
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "time":
                    # The module itself is the finding: ``tick = time.time``
                    # and ``getattr(time, "time")()`` call nothing by name.
                    time_modules.add(alias.asname or alias.name)
                    found.add("import time")
        elif isinstance(node, ast.ImportFrom):
            base = _resolve(node.module, node.level, package)
            for alias in node.names:
                if base == "time":
                    time_functions.add(alias.asname or alias.name)
                    found.add(f"import time.{alias.name}")
                if alias.name == "utcnow":
                    found.add("import utcnow")
    for node in ast.walk(tree):
        if _terminal_name(node) == "utcnow":
            found.add("reference utcnow")
        if isinstance(node, ast.Attribute) and node.attr in CLOCK_ATTRIBUTES:
            owner = _terminal_name(node.value)
            if owner in owners:
                found.add(f"reference {owner}.{node.attr}")
            elif not isinstance(node.value, ast.Name):
                # Taken off an expression, not a name: ``context.now.now`` (every
                # policy module holds a datetime INSTANCE in ``context.now``),
                # ``type(x).now``, ``[datetime][0].now`` — no owner to follow.
                found.add(f"reference {UNNAMED_OWNER}.{node.attr}")
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = _terminal_name(func)
        if name == "getattr" and len(node.args) > 1 and _terminal_name(node.args[0]) in owners:
            owner, attribute = _terminal_name(node.args[0]), _folded(node.args[1])
            if attribute is None:
                found.add(f"computed attribute of {owner}")
            elif attribute in CLOCK_ATTRIBUTES:
                found.add(f"reference {owner}.{attribute}")
        if name in CLOCK_METHODS:
            found.add(f"call {name}()")
        elif isinstance(func, ast.Name) and name in time_functions:
            found.add(f"call time.{name}()")
        elif isinstance(func, ast.Attribute):
            owner = func.value
            if isinstance(owner, ast.Name) and owner.id in time_modules:
                found.add(f"call time.{name}()")
            elif name in TIME_FUNCTIONS:
                found.add(f"call {name}()")
    return found


class TestForbiddenImportChecker:
    @pytest.mark.parametrize(
        "snippet, package",
        [
            ("import anthropic", "aegis.policy"),
            ("from anthropic import Anthropic", "aegis.policy"),
            ("import anthropic.types as types", "aegis.policy"),
            ("from aegis.brain.llm import AnthropicLLM", "aegis.policy"),
            ("from aegis import brain", "aegis.policy"),
            ("from ..brain.cycle import run_cycle", "aegis.policy"),
            ("from .. import brain", "aegis.policy"),
            ("import alpaca", "aegis.policy"),
            ("from alpaca.trading.client import TradingClient", "aegis.policy"),
            ("from aegis.data.clients import trading_client", "aegis.policy"),
            ("from aegis.data import clients", "aegis.policy"),
            ("from ..data.clients import trading_client", "aegis.policy"),
            ("from aegis.data.news import get_news", "aegis.policy"),
            ("from aegis.data import news", "aegis.policy"),
            ("import requests", "aegis.policy"),
            ("from requests.exceptions import RequestException", "aegis.policy"),
            ("import httpx", "aegis.policy"),
            ("import httpx2", "aegis.policy"),
            ("import urllib", "aegis.policy"),
            ("import urllib.request", "aegis.policy"),
            ("from urllib import request", "aegis.policy"),
            ("from urllib.parse import urlparse", "aegis.policy"),
            ("import socket", "aegis.policy"),
            ("import random", "aegis.policy"),
            ("from random import random", "aegis.policy"),
            ("import secrets", "aegis.policy"),
            ("import subprocess", "aegis.policy"),
            ("def later():\n    import random\n    return random.random()", "aegis.policy"),
            ("import importlib\nimportlib.import_module('anthropic')", "aegis.policy"),
            ("__import__('random')", "aegis.policy"),
            ("import importlib\nimportlib.import_module(name)", "aegis.policy"),
        ],
    )
    def test_catches_every_forbidden_import(self, snippet, package):
        assert forbidden_imports(snippet, package=package), snippet

    @pytest.mark.parametrize(
        "snippet",
        [
            '"""No model calls: never anthropic, aegis.brain or alpaca."""',
            "# import random",
            "import math",
            "from datetime import datetime, timedelta",
            "from aegis.data.models import Quote, utcnow",
            "from aegis.data.errors import DataError",
            "from aegis.data.market import get_spot",  # the fetcher rule's business, not this one's
            "from aegis.store.repo import record_decision",
            "from aegis.policy.measures import known",
            "from . import measures",
            "from .models import PolicyContext",
            "import urllib3",  # another top-level name; not on the list
            "import randomness",
            "from aegis.brainstorm import idea",
            "note = 'never import anthropic or random here'",
        ],
    )
    def test_does_not_flag_what_policy_may_import(self, snippet):
        assert forbidden_imports(snippet, package="aegis.policy") == set(), snippet

    def test_every_listed_module_is_caught_on_its_own(self):
        """Dropping any one name from ``FORBIDDEN_IN_POLICY`` fails here."""
        assert FORBIDDEN_IN_POLICY == (
            "anthropic", "aegis.brain", "alpaca", "aegis.data.clients", "aegis.data.news",
            "requests", "httpx", "httpx2", "urllib", "socket", "random", "secrets", "subprocess",
        )
        for module in FORBIDDEN_IN_POLICY:
            assert forbidden_imports(f"import {module}") == {module}
            assert forbidden_imports(f"import {module}.sub") == {f"{module}.sub"}
            parent, _, name = module.rpartition(".")
            if parent:
                assert module in forbidden_imports(f"from {parent} import {name}")


class TestAllowlistChecker:
    @pytest.mark.parametrize(
        "snippet, outside",
        [
            ("from aegis.store.repo import record_decision", {"aegis.store.repo.record_decision"}),
            ("from aegis.store import repo", {"aegis.store.repo"}),
            ("import aegis.store.repo", {"aegis.store.repo"}),
            ("from aegis.store.db import connect", {"aegis.store.db.connect"}),
            ("from aegis.store import db, models", {"aegis.store.db"}),
            ("import sqlite3", {"sqlite3"}),
            ("import time", {"time"}),
            ("from time import monotonic", {"time.monotonic"}),
            ("import os", {"os"}),
            ("import os.path", {"os.path"}),
            ("from os import environ", {"os.environ"}),
            ("import uuid", {"uuid"}),
            ("from uuid import uuid4", {"uuid.uuid4"}),
            ("import collections", {"collections"}),
            ("from collections import OrderedDict", {"collections.OrderedDict"}),
            ("import aegis.policy", {"aegis.policy"}),
            ("from aegis.policy import context", {"aegis.policy.context"}),
            ("from aegis.policy.engine import decide", {"aegis.policy.engine.decide"}),
            ("from . import context", {"aegis.policy.context"}),
            ("from .engine import decide", {"aegis.policy.engine.decide"}),
            ("from aegis.data.market import get_spot", {"aegis.data.market.get_spot"}),
            ("from aegis.pricing.position import analyze_position",
             {"aegis.pricing.position.analyze_position"}),
            ("import zoneinfo, math", {"zoneinfo"}),
            ("def later():\n    import json\n    return json", {"json"}),
            ("__import__('os')", {"os"}),
            ("import importlib\nimportlib.import_module(name)", {"importlib", COMPUTED_IMPORT}),
            ("from aegis.policy import *", {"aegis.policy.*"}),
        ],
    )
    def test_catches_everything_off_the_list(self, snippet, outside):
        assert outside_allowlist(snippet, RULES_ALLOWED, package="aegis.policy") == outside

    @pytest.mark.parametrize(
        "snippet",
        [
            "from __future__ import annotations",
            "import math",
            "from math import isfinite",
            "import datetime",
            "from datetime import date, datetime, timedelta, timezone",
            "import collections.abc",
            "from collections.abc import Callable, Iterable",
            "from collections import abc",
            "import typing",
            "from typing import Any",
            "from aegis.policy.models import PolicyContext, RuleResult",
            "from aegis.policy import models",
            "from .models import PolicyContext",
            "from . import models",
            "from aegis.config import RiskLimits",
            "from aegis.data.models import OptionSnapshot, Quote, parse_occ_symbol",
            "from aegis.pricing.models import PositionSummary",
            "from aegis.store.models import OrderSide, OrderType, ProposalLeg",
            "from aegis.policy.measures import exceeds, known, money",
            "from aegis.policy import measures",
            "from . import measures",
            '"""Reads no store: never aegis.store.repo, sqlite3, time, os or uuid."""',
        ],
    )
    def test_the_allowlist_itself_passes(self, snippet):
        assert outside_allowlist(snippet, RULES_ALLOWED, package="aegis.policy") == set(), snippet

    def test_measures_may_not_import_itself_or_the_rules(self):
        assert RULES_ALLOWED - MEASURES_ALLOWED == {"aegis.policy.measures"}
        snippet = "from aegis.policy.measures import known"
        assert outside_allowlist(snippet, MEASURES_ALLOWED) == {"aegis.policy.measures.known"}
        assert outside_allowlist("from aegis.policy import rules", RULES_ALLOWED) == {
            "aegis.policy.rules",
        }

    @pytest.mark.parametrize(
        "snippet, values",
        [
            ("from aegis.config import get_config", {"get_config"}),
            ("from aegis.config import RiskLimits, load_config", {"load_config"}),
            ("from aegis.config import REPO_ROOT", {"REPO_ROOT"}),
            ("from aegis.config import require_env as env", {"require_env"}),
            ("from aegis.config import load_env", {"load_env"}),
            ("from aegis.config import nothing_by_this_name", {"nothing_by_this_name"}),
            ("from aegis.config import *", {"*"}),
            ("import aegis.config", {CONFIG}),
            ("import aegis.config as config", {CONFIG}),
            ("from aegis import config", {CONFIG}),
            ("from .. import config", {CONFIG}),
            ("from ..config import get_config", {"get_config"}),
            ("from aegis.config import RiskLimits", set()),
            ("from aegis.config import AegisConfig, AutoExecuteConfig, RiskLimits", set()),
            ("from ..config import RiskLimits", set()),
            ("from aegis.configuration import get_config", set()),
            ('"""Never aegis.config.get_config: limits come from the context."""', set()),
        ],
    )
    def test_from_the_config_module_only_types(self, snippet, values):
        assert config_values(snippet, package="aegis.policy") == values, snippet


class TestClockChecker:
    @pytest.mark.parametrize(
        "snippet",
        [
            "from aegis.data.models import utcnow",
            "from aegis.data.models import utcnow as clock\nstamp = clock()",
            "stamp = utcnow()",
            "stamp = models.utcnow()",
            "stamp = datetime.utcnow()",
            "factory = utcnow",  # handed on, not called: still a clock
            "field = Field(default_factory=utcnow)",
            "stamp = datetime.now()",
            "stamp = datetime.now(timezone.utc)",
            "stamp = datetime.datetime.now(tz=UTC)",
            "day = date.today()",
            "day = datetime.date.today()",
            "stamp = now()",
            "day = today()",
            "import time\nstamp = time.time()",
            "import time\nstamp = time.monotonic()",
            "import time as clock\nstamp = clock.time()",
            "import time\nwhen = time.localtime()",
            "import time\ntime.sleep(1)",
            "from time import time\nstamp = time()",
            "from time import monotonic\nstamp = monotonic()",
            "from time import monotonic as tick\nstamp = tick()",
            "stamp = clock.monotonic()",
            "stamp = timer.perf_counter()",
            "stamp = clock.time_ns()",
            "def later():\n    return datetime.now()",
            "stamp = datetime.fromtimestamp(time_source.monotonic())",
            # the time module brought in by a call: no name is ever bound to it
            '__import__("time").time()',
            'as_of = context.now if __import__("time").time() else context.now',
            "clock = __import__('time')\nstamp = clock.time()",
            "stamp = getattr(__import__('time'), 'time')()",
            "import importlib\nstamp = importlib.import_module('time').monotonic()",
            "from importlib import import_module as load\nstamp = load('time').time()",
            "__import__('ti' + 'me').time()",
            "__import__(name).time()",
            "load = __import__\nstamp = load('time').time()",
            # a clock handed on instead of called by name: read by whoever calls it
            "stamp: datetime = Field(default_factory=datetime.now)",
            "_now = datetime.now\nstamp = _now()",
            "from time import *\nstamp = time()",
            "import time\ntick = time.time\nstamp = tick()",
            "import time\nstamp = getattr(time, 'time')()",
            "day = date.today",
            "_now = datetime.datetime.now",
            "from datetime import datetime as dt\n_now = dt.now",
            "import datetime as d\n_today = d.date.today",
            "moment = datetime\n_now = moment.now",
            "_now = getattr(datetime, 'now')",
            "_now = getattr(datetime, 'n' + 'ow')",
            "_now = getattr(datetime, name)",
            "import time",  # the module itself: nothing in the pure core needs it
            "import time as clock",
            "from time import sleep",
            "def later():\n    import time\n    return time",
        ],
    )
    def test_catches_every_clock_read(self, snippet):
        assert clock_reads(snippet, package="aegis.policy"), snippet

    @pytest.mark.parametrize(
        "snippet, reads",
        [
            ("stamp: datetime = Field(default_factory=datetime.now)", {"reference datetime.now"}),
            ("_now = datetime.now\nstamp = _now()", {"reference datetime.now"}),
            ("day = date.today", {"reference date.today"}),
            ("_now = datetime.datetime.now", {"reference datetime.now"}),
            ("from datetime import datetime as dt\n_now = dt.now", {"reference dt.now"}),
            ("from datetime import date as d\n_today = d.today", {"reference d.today"}),
            ("moment = datetime\n_now = moment.now", {"reference moment.now"}),
            ("a = datetime\nb = a\n_now = b.now", {"reference b.now"}),
            ("_now = getattr(datetime, 'now')", {"reference datetime.now"}),
            ("_now = getattr(datetime, 'to' + 'day')", {"reference datetime.today"}),
            ("_now = getattr(datetime, name)", {"computed attribute of datetime"}),
            ("f = datetime.utcnow", {"reference utcnow", "reference datetime.utcnow"}),
            ("stamp = datetime.now()", {"call now()", "reference datetime.now"}),
            ("from time import *\nstamp = time()", {"import time.*"}),
            ("import time\ntick = time.time\nstamp = tick()", {"import time"}),
            ("import time\nstamp = getattr(time, 'time')()", {"import time"}),
            ("import time as clock", {"import time"}),
            ("from time import sleep", {"import time.sleep"}),
            ("import time\nstamp = time.time()", {"import time", "call time.time()"}),
            # the same clock taken off an instance or an expression: no name to follow
            ("_now = context.now.now", {f"reference {UNNAMED_OWNER}.now"}),
            ("day = context.now.today", {f"reference {UNNAMED_OWNER}.today"}),
            ("f = type(context.now).now", {f"reference {UNNAMED_OWNER}.now"}),
            ("f = context.now.__class__.now", {f"reference {UNNAMED_OWNER}.now"}),
            ("f = [datetime][0].now", {f"reference {UNNAMED_OWNER}.now"}),
            ("f = context.now.utcnow",
             {"reference utcnow", f"reference {UNNAMED_OWNER}.utcnow"}),
            ("stamp = context.now.now()", {"call now()", f"reference {UNNAMED_OWNER}.now"}),
            # what the datetime classes offer that is no clock
            ("moment = getattr(datetime, 'fromisoformat')(text)", set()),
            ("kind = datetime\nmoment = kind.fromisoformat(text)", set()),
            ("last = datetime.max\nfirst = date.min", set()),
        ],
    )
    def test_a_clock_handed_on_is_a_clock_read(self, snippet, reads):
        """A wall-clock read need not be a call by name: ``_now = datetime.now``
        followed by ``_now()`` reads the clock in a module that imports
        ``datetime`` legitimately — so the reference is the finding, and the
        ``time`` module, which the pure core has no use for, is one outright."""
        assert clock_reads(snippet, package="aegis.policy") == reads, snippet

    @pytest.mark.parametrize(
        "snippet",
        [
            "a, b = datetime, 1\n_now = a.now",  # the class reaches a name by unpacking
            "def later(clock=datetime):\n    return clock.now",  # ... or as a default
        ],
    )
    def test_the_stated_limit_of_the_clock_rule_is_pinned(self, snippet):
        """What the rule does not follow, so nobody takes the silence for
        coverage: the datetime class bound to a bare name by something other
        than an import or a plain assignment, and the clock handed on off
        that name. (Called on the spot it IS seen — ``a.now()`` is a call of
        ``now``; and a read whose value is used fails the behaviour tests.)"""
        assert clock_reads(snippet, package="aegis.policy") == set(), snippet
        called = snippet.replace(".now", ".now()")
        assert clock_reads(called, package="aegis.policy") == {"call now()"}, called

    def test_the_clockless_modules_take_no_clock_attribute_off_an_expression(self):
        """The expression rule costs nothing: every ``.now`` in the clockless
        modules is ``context.now`` — an attribute of a bare name."""
        seen = 0
        for path in CLOCKLESS_SOURCES:
            for node in ast.walk(ast.parse(_source(path))):
                if isinstance(node, ast.Attribute) and node.attr in CLOCK_ATTRIBUTES:
                    assert isinstance(node.value, ast.Name), f"{_id(path)}:{node.lineno}"
                    assert node.value.id == "context", f"{_id(path)}:{node.lineno}"
                    seen += 1
        assert seen >= 10  # the rule is not vacuous: the rules do read ``context.now``

    def test_the_clockless_modules_import_no_time_module_today(self):
        """The rule costs nothing: no clockless policy module imports ``time``
        in any form, and each still imports ``datetime`` (so "it imports no
        clock module at all" is not why the references rule passes)."""
        for path in CLOCKLESS_SOURCES:
            found = imported_modules(_source(path), package=_package_of(path))
            assert not imports(found, "time"), _id(path)
        for name in ("rules.py", "engine.py", "models.py"):
            found = imported_modules(_source(POLICY_DIR / name), package="aegis.policy")
            assert imports(found, "datetime"), name

    @pytest.mark.parametrize(
        "snippet, reads",
        [
            ('__import__("time").time()', {"import time by a call"}),
            ('as_of = context.now if __import__("time").time() else context.now',
             {"import time by a call"}),
            ("clock = __import__('time')", {"import time by a call"}),
            ("import importlib\nimportlib.import_module('time').monotonic()",
             {"import time by a call", "call monotonic()"}),
            ("from importlib import import_module as load\nload('ti' + 'me')",
             {"import time by a call"}),
            ("__import__(name).time()", {f"import {COMPUTED_IMPORT} by a call"}),
            ("load = __import__", {f"import {COMPUTED_IMPORT} by a call"}),
            # a module that is no clock, by a call that can be read: not this checker's
            ("__import__('math').floor(1.5)", set()),
            ("import importlib\nimportlib.import_module('datetime').time(9, 30)", set()),
            ("__import__('timeit')", set()),
        ],
    )
    def test_the_time_module_imported_by_a_call_is_a_clock_read(self, snippet, reads):
        """Finding 3.23: ``__import__("time").time()`` binds no name the
        checker could follow, so the import call itself is the finding — as
        is one whose target nobody can read."""
        assert clock_reads(snippet, package="aegis.policy") == reads, snippet

    @pytest.mark.parametrize(
        "snippet",
        [
            "stamp = context.now",  # attribute ACCESS: the context's own instant
            "if until > context.now:\n    pass",
            "later = context.now + timedelta(hours=limits.halt_fallback_hours)",
            "decided_at = evaluation.decided_at\nas_of = report.as_of",
            "day = context.market_date\ntoday = day",
            "now = context.now",  # a local called ``now`` is a value, not a clock
            "def f(now):\n    return now",
            "from datetime import date, datetime, time, timedelta, timezone",
            "from datetime import time\nopening = time(9, 30)",
            "import datetime\nopening = datetime.time(9, 30)",
            "start = datetime.combine(day, time.min, tzinfo=zone)",
            "stamp = datetime.fromisoformat(text)",
            "moment = value.astimezone(timezone.utc)",
            '"""Never utcnow(), datetime.now() or time.time(): only context.now."""',
            "# stamp = datetime.now()",
            "text = 'now'",
            "knows = known(value)",
            # ``.now`` / ``.today`` on something that is not the datetime classes
            "when = context.now\nlater = when",
            "stamp = row.now",
            "seen = evaluation.today",
            "from datetime import datetime, time\nopening = time(9, 30)",
            "kind = datetime\nmoment = kind.fromisoformat(text)",
            "forever = datetime.max.replace(tzinfo=timezone.utc)",
            "name = getattr(rule, '__name__', None)",
        ],
    )
    def test_reading_the_context_is_not_reading_a_clock(self, snippet):
        assert clock_reads(snippet, package="aegis.policy") == set(), snippet


# --- PART 2: the rules --------------------------------------------------------


def _policy_id(path: Path) -> str:
    return str(path.relative_to(POLICY_DIR))


CONTEXT = POLICY_DIR / "context.py"
CLOCKLESS_SOURCES = [path for path in POLICY_SOURCES if path != CONTEXT]


class TestPolicyIsPureDeterministicPython:
    def test_the_walk_covers_the_whole_package(self):
        found = {_policy_id(path) for path in POLICY_SOURCES}
        assert EXPECTED_POLICY_MODULES <= found, EXPECTED_POLICY_MODULES - found
        assert CONTEXT in POLICY_SOURCES and CONTEXT not in CLOCKLESS_SOURCES
        for name in ("rules.py", "measures.py", "engine.py", "limits.py"):
            assert POLICY_DIR / name in CLOCKLESS_SOURCES

    @pytest.mark.parametrize("path", POLICY_SOURCES, ids=_policy_id)
    def test_no_policy_module_imports_a_model_a_broker_a_network_or_chance(self, path):
        found = forbidden_imports(_source(path), package=_package_of(path))
        assert found == set(), f"{_id(path)} imports {sorted(found)}"

    @pytest.mark.parametrize("path", POLICY_SOURCES, ids=_policy_id)
    def test_only_context_imports_the_live_fetchers(self, path):
        found = imported_modules(_source(path), package=_package_of(path))
        touches = [module for module in LIVE_FETCHERS if imports(found, module)]
        if path == CONTEXT:
            assert touches == list(LIVE_FETCHERS)  # the rule is not vacuous
        else:
            assert touches == [], f"{_id(path)} imports {touches}"

    @pytest.mark.parametrize(
        "name, allowed",
        [("measures.py", MEASURES_ALLOWED), ("rules.py", RULES_ALLOWED)],
        ids=["measures.py", "rules.py"],
    )
    def test_the_pure_core_imports_only_its_allowlist(self, name, allowed):
        path = POLICY_DIR / name
        outside = outside_allowlist(_source(path), allowed, package=_package_of(path))
        assert outside == set(), f"{_id(path)} imports {sorted(outside)}"
        assert config_values(_source(path), package=_package_of(path)) == set()
        # in particular: no store access, no clock, no environment, no chance
        found = imported_modules(_source(path), package=_package_of(path))
        for module in ("aegis.store.repo", "aegis.store.db", "sqlite3", "time", "os", "uuid"):
            assert not imports(found, module), f"{_id(path)} imports {module}"

    def test_the_allowlists_are_the_specs(self):
        assert MEASURES_ALLOWED == {
            "__future__", "math", "datetime", "collections.abc", "typing",
            "aegis.policy.models", "aegis.config", "aegis.data.models", "aegis.pricing.models",
            "aegis.store.models",
        }
        assert RULES_ALLOWED == MEASURES_ALLOWED | {"aegis.policy.measures"}
        rules = imported_modules(_source(POLICY_DIR / "rules.py"), package="aegis.policy")
        assert imports(rules, "aegis.policy.measures")  # the extra entry is really used

    @pytest.mark.parametrize("path", CLOCKLESS_SOURCES, ids=_policy_id)
    def test_no_clock_is_read_outside_context(self, path):
        found = clock_reads(_source(path), package=_package_of(path))
        assert found == set(), f"{_id(path)}: {sorted(found)}"

    def test_context_is_where_the_clock_is_read(self):
        """The one module that may: so the checker demonstrably sees a real read."""
        found = clock_reads(_source(CONTEXT), package="aegis.policy")
        assert {"import utcnow", "reference utcnow"} <= found

    def test_the_rules_still_read_the_contexts_now(self):
        """``context.now`` is attribute access — allowed, and really used:
        the clock rule does not pass because the rules ignore time."""
        tree = ast.parse(_source(POLICY_DIR / "rules.py"))
        reads = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute)
            and node.attr == "now"
            and isinstance(node.value, ast.Name)
            and node.value.id == "context"
        ]
        assert reads


class TestRuntimePureCore:
    def test_the_pure_core_loads_no_model_and_no_broker(self):
        """In a fresh interpreter, importing the rules, the measures, the
        engine and the limits report loads neither ``anthropic`` nor any
        ``aegis.brain*`` nor any ``alpaca*`` module — directly or through
        anything they import."""
        code, report = _run_imports(
            modules=list(PURE_MODULES), forbidden=list(NEVER_LOADED_BY_THE_PURE_CORE)
        )
        assert report == {"imported": list(PURE_MODULES), "failed": {}, "loaded": []}
        assert code == 0

    @pytest.mark.parametrize(
        "extra, prefix",
        [
            ("import anthropic", "anthropic"),
            ("import aegis.brain.models", "aegis.brain"),
            ("import alpaca.common.exceptions", "alpaca"),
        ],
        ids=["anthropic", "brain", "alpaca"],
    )
    def test_the_harness_catches_each_forbidden_load(self, extra, prefix):
        """Self-test: each of the three is seen once loaded."""
        code, report = _run_imports(
            modules=list(PURE_MODULES), forbidden=list(NEVER_LOADED_BY_THE_PURE_CORE), extra=extra
        )
        assert code == 1
        assert report["loaded"] and all(_within(name, (prefix,)) for name in report["loaded"])

    def test_the_context_builder_is_the_module_that_reaches_the_broker_sdk(self):
        """Self-test on the real tree: ``context.py`` imports the live
        fetchers, and the harness sees what that loads — so its silence about
        the four pure modules means they load none of it."""
        code, report = _run_imports(
            modules=["aegis.policy.context"], forbidden=list(NEVER_LOADED_BY_THE_PURE_CORE)
        )
        assert code == 1 and report["failed"] == {}
        assert "alpaca" in report["loaded"]
        others = ("anthropic", "aegis.brain")
        assert not [name for name in report["loaded"] if _within(name, others)]
