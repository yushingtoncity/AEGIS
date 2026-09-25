"""Governance, enforced statically and at runtime: the brain has no execution authority.

Every ``.py`` under ``aegis/brain`` and the brain CLI is parsed with ``ast``
and held to four static rules:

1. No import of ``aegis.execution`` in any static form — ``import a.b``,
   ``from a.b import c``, ``from a import b``, relative imports resolved
   against the module's package, a literal ``import_module``/``__import__``
   argument, or a string literal that *is* the module path.
2. No dynamic-import machinery at all, since a computed name is exactly what
   rule 1 cannot see through and the brain has no legitimate use for it: no
   import of ``importlib`` (any submodule), ``pkgutil``, ``runpy``, ``imp``
   or ``zipimport``, nor of the modules that hand out the same door by
   another name — ``builtins``, ``pickle``/``shelve``/``marshal`` (a module
   or code object from bytes), ``ctypes``, ``code``/``codeop``, ``pydoc``,
   ``unittest`` (``mock.patch`` imports its target by name),
   ``logging.config`` (it resolves dotted names), ``annotationlib``, the
   statement runners ``timeit``, ``bdb``, ``pdb``, ``profile``, ``cProfile``,
   ``trace`` and ``doctest`` (each an ``exec`` by another name) — or that
   start another interpreter, which no audit hook in this one can see:
   ``os``, ``subprocess``, ``multiprocessing`` (and ``concurrent``, whose
   ``futures.process`` hands it out as ``mp``), ``pty``, ``webbrowser``;
   no use of the builtins ``__import__``, ``exec``, ``eval`` or ``compile``
   (a method that happens to share a name, like ``re.compile``, is fine),
   nor of ``globals``/``vars``/``locals``; no ``getattr``/``setattr``/
   ``hasattr``/``delattr`` except called with a literal attribute name
   (``getattr(builtins, "ex" + "ec")`` and ``reduce(getattr, …)`` are out);
   no attribute or imported name among the loader entry points
   (``import_module``, ``resolve_name``, ``spec_from_file_location``,
   ``module_from_spec``, ``run_module``, ``run_path`` …), the resolvers of a
   dotted name (pydantic's ``ImportString``, ``get_type_hints`` and
   ``get_annotations`` evaluating a string annotation), the process
   spawners (``Popen``, ``posix_spawn``, ``fork``, ``execv`` …) or the
   reflection escape hatches back to the builtins or a module table
   (``__builtins__``, ``__globals__``, ``__dict__``, ``__self__``,
   ``sys.modules``, ``attrgetter`` …); no string literal naming one of them
   (``getattr(builtins, "__import__")``). Any of these is also recorded as a
   *dynamic import*, which counts as importing anything at all.
3. No string or bytes literal that pairs ``execution`` with ``aegis`` or a
   path separator — ``"aegis.execution"``, ``"aegis/execution/base.py"``,
   ``"aegis.execution:Executor"``, ``b"caegis.execution.base\\nExecutor"``.
   Docstrings (and any other bare string statement, which no code can use)
   may mention the package freely.
4. No broker access: paper-only still places orders, so no import of
   ``alpaca`` (any submodule) or ``aegis.data.clients`` (which hands out
   Alpaca's ``TradingClient``), and no name, attribute, imported name or
   string naming an order capability (``trading_client``,
   ``submit_order``, ``cancel_orders``, ``close_all_positions``,
   ``exercise_options_position`` …). Beyond that, only ``snapshot.py`` may
   import from ``aegis.data`` anything but its ``models`` and ``errors`` —
   and it only the three live fetchers.

Each rule's checker is tested against synthetic snippets, so a form it
missed shows up here rather than in production. The static rules are a
tripwire, not a sandbox — Python always has one more door — so the runtime
rule is the backstop: a fresh interpreter installs an audit hook first,
makes every order-placing method of Alpaca's ``TradingClient`` (and
``aegis.data.clients.trading_client``) record and refuse, and then runs the
brain the way production does as well as the way tests do — a full
``FakeLLM`` cycle; ``AnthropicLLM.complete`` over a stand-in SDK client on
its success path (after a retried 429), a refusal and a truncation; the
no-trade and halted outcomes; and every brain CLI command (``once``,
``stage scan|thesis|proposal``, ``usage``), ``once`` and ``stage scan`` also
through the default ``build_market_snapshot`` with the data layer's
fetchers replaced in-process by fixture-backed fakes. Nothing under
``aegis/execution`` may be imported, opened, compiled or executed — by any
module name, and by any spelling of its path (relative, through ``..`` or a
symlink, differently cased) — nor may any ``aegis.execution*`` module be
loaded afterwards, a child process be started, or an order method be
called; the hook proves it was installed before the brain was imported,
and the harness proves it catches each of those loads, during the cycle as
well as after it. Only ``llm.py`` imports ``anthropic`` and only
``snapshot.py`` imports the live fetchers, pinned statically.
``tests/conftest.py`` makes the package unimportable in-process during
every ``test_brain_*`` test besides.
"""

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from aegis.config import REPO_ROOT

BRAIN_DIR = REPO_ROOT / "aegis" / "brain"
CLI_BRAIN = REPO_ROOT / "aegis" / "cli" / "brain.py"
EXECUTION_DIR = REPO_ROOT / "aegis" / "execution"

EXECUTION = "aegis.execution"
LIVE_FETCHERS = ("aegis.data.market", "aegis.data.account", "aegis.data.news")
EXPECTED_MODULES = {
    "__init__.py", "errors.py", "models.py", "schemas.py", "llm.py", "testing.py", "prompts/__init__.py",
    "snapshot.py", "stage.py", "scan.py", "thesis.py", "proposal.py", "cycle.py",
}

DYNAMIC_IMPORT = "<dynamic import>"
"""Recorded for any dynamic-import machinery (rule 2), whatever its argument:
a computed name could resolve to anything, so it counts as importing
everything (``imports``)."""

DYNAMIC_MODULES = (
    "importlib", "pkgutil", "runpy", "imp", "zipimport",
    # the same door by another name: a handle on the builtins, a module or code object
    # made from bytes, raw memory, an interpreter loop, an import by dotted name
    "builtins", "pickle", "_pickle", "shelve", "marshal", "ctypes", "code", "codeop", "pydoc",
    "unittest", "logging.config", "annotationlib",
    # a statement string run for you: exec by another name
    "timeit", "bdb", "pdb", "profile", "cProfile", "trace", "doctest",
    # another interpreter, which loads whatever it likes where no audit hook of ours can see
    # (concurrent.futures.process hands out multiprocessing as ``mp``)
    "os", "posix", "nt", "subprocess", "_posixsubprocess", "_winapi", "multiprocessing", "concurrent",
    "pty", "webbrowser",
)
DYNAMIC_BUILTINS = frozenset({"__import__", "exec", "eval", "compile"})
# hand out a namespace, so a computed key reaches any of the above
NAMESPACE_BUILTINS = frozenset({"globals", "vars", "locals"})
# fine with a literal attribute name; with a computed one they reach anything
ATTRIBUTE_BUILTINS = frozenset({"getattr", "setattr", "hasattr", "delattr"})
DYNAMIC_ATTRIBUTES = frozenset(
    {
        "import_module", "resolve_name", "spec_from_file_location", "module_from_spec",
        "run_module", "run_path",
        # the rest of the loader surface, for the same reason
        "exec_module", "load_module", "spec_from_loader", "SourceFileLoader",
        # the ways back to the builtins, a module table or raw code without importing anything
        "__builtins__", "__globals__", "__dict__", "__self__", "__subclasses__", "__code__",
        "__loader__", "__spec__", "__getattribute__", "modules", "attrgetter", "methodcaller",
        "getattr_static", "FunctionType", "CodeType",
        # a dotted name resolved (pydantic's ImportString) or a string annotation evaluated
        "ImportString", "get_type_hints", "get_annotations", "ForwardRef",
        # process spawners, whatever module they are reached through
        "Popen", "popen", "posix_spawn", "posix_spawnp", "fork", "forkpty", "startfile",
        "execv", "execve", "execl", "execle", "execlp", "execlpe", "execvp", "execvpe",
        "spawnv", "spawnve", "spawnl", "spawnle", "spawnlp", "spawnlpe", "spawnvp", "spawnvpe",
        "create_subprocess_exec", "create_subprocess_shell", "subprocess_exec", "subprocess_shell",
        "ProcessPoolExecutor",
    }
)
# Rule 4: the broker. alpaca-py's TradingClient places orders (paper or not), and
# aegis.data.clients.trading_client() hands one out.
BROKER_MODULES = ("alpaca", "aegis.data.clients")
ORDER_NAMES = frozenset(
    {
        "trading_client", "TradingClient", "submit_order", "replace_order_by_id",
        "cancel_order_by_id", "cancel_orders", "close_position", "close_all_positions",
        "exercise_options_position",
    }
)
# What any brain module may import from the data layer: its typed models and errors.
DATA_MODELS = ("aegis.data.models", "aegis.data.errors")
_BUILTINS_MODULES = frozenset({"builtins", "__builtins__"})
# a string literal naming any of these is a getattr/subscript away from using it
_DYNAMIC_STRINGS = DYNAMIC_BUILTINS | NAMESPACE_BUILTINS | DYNAMIC_ATTRIBUTES | _BUILTINS_MODULES


def _brain_sources() -> list[Path]:
    return sorted(BRAIN_DIR.rglob("*.py"))


def _package_of(path: Path) -> str:
    """The dotted package a module's relative imports resolve against."""
    relative = path.relative_to(REPO_ROOT).with_suffix("")
    parts = list(relative.parts)
    if parts[-1] == "__init__":
        parts.pop()
    else:
        parts.pop()  # a plain module belongs to its directory's package
    return ".".join(parts)


def _resolve(module: str | None, level: int, package: str) -> str:
    """``from <module> import`` with ``level`` leading dots, as an absolute name."""
    if level == 0:
        return module or ""
    base = package.split(".")
    if level > 1:
        base = base[: len(base) - (level - 1)]
    return ".".join(part for part in [*base, module or ""] if part)


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


def _literal_attribute_calls(tree: ast.AST) -> set[int]:
    """ids of the ``getattr``-family names that are called with a literal
    attribute name (``getattr(exc, "response", None)``) — the only use rule 2
    allows them."""
    return {
        id(node.func)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in ATTRIBUTE_BUILTINS
        and len(node.args) >= 2
        and isinstance(node.args[1], ast.Constant)
        and isinstance(node.args[1].value, str)
    }


def dynamic_machinery(source: str, *, package: str = "") -> set[str]:
    """Rule 2: every dynamic-import construct in ``source``, described (empty when clean)."""
    tree = ast.parse(source)
    found = {
        f"import {name}"
        for name in _static_imports(tree, package)
        if any(name == m or name.startswith(m + ".") for m in DYNAMIC_MODULES)
    }
    bare = _bare_strings(tree)
    literal_calls = _literal_attribute_calls(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            if node.id in DYNAMIC_BUILTINS | NAMESPACE_BUILTINS:
                found.add(f"builtin {node.id}")
            elif node.id in ATTRIBUTE_BUILTINS and id(node) not in literal_calls:
                found.add(f"computed {node.id}")  # a non-literal name, or passed around as a value
            elif node.id in DYNAMIC_ATTRIBUTES:
                found.add(f"name {node.id}")  # __builtins__, __loader__, an imported attrgetter
        elif isinstance(node, ast.Attribute):
            if node.attr in DYNAMIC_ATTRIBUTES:
                found.add(f"attribute {node.attr}")
            elif node.attr in DYNAMIC_BUILTINS and (
                node.attr != "compile"  # re.compile is a method, not the builtin
                or isinstance(node.value, ast.Name) and node.value.id in _BUILTINS_MODULES
            ):
                found.add(f"builtin {node.attr}")
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name in DYNAMIC_ATTRIBUTES | ATTRIBUTE_BUILTINS | NAMESPACE_BUILTINS:
                    found.add(f"import {alias.name}")  # from sys import modules as m
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in bare:
            if node.value in _DYNAMIC_STRINGS:
                found.add(f"string {node.value!r}")
    return found


def execution_literals(source: str) -> set[str]:
    """Rule 3: every string or bytes literal (docstrings aside) pairing
    ``execution`` with ``aegis`` or a path separator. Bytes are read as
    latin-1, so ``pickle.loads(b"caegis.execution.base\\n…")`` is seen too."""
    tree = ast.parse(source)
    bare = _bare_strings(tree)
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, (str, bytes)) and id(node) not in bare:
            value = node.value.decode("latin-1") if isinstance(node.value, bytes) else node.value
            text = value.lower()
            if "execution" in text and ("aegis" in text or "/" in text or "\\" in text):
                found.add(value)
    return found


def broker_access(source: str, *, package: str = "") -> set[str]:
    """Rule 4: every way ``source`` reaches the broker, described (empty when
    clean) — an import of ``alpaca`` or ``aegis.data.clients``, or a name,
    attribute, imported name or string literal (docstrings aside) that names
    an order capability."""
    tree = ast.parse(source)
    found = {
        f"import {name}"
        for name in _static_imports(tree, package)
        if any(name == m or name.startswith(m + ".") for m in BROKER_MODULES)
    }
    bare = _bare_strings(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in ORDER_NAMES:
            found.add(f"name {node.id}")
        elif isinstance(node, ast.Attribute) and node.attr in ORDER_NAMES:
            found.add(f"attribute {node.attr}")
        elif isinstance(node, ast.ImportFrom):
            found.update(f"import {alias.name}" for alias in node.names if alias.name in ORDER_NAMES)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in bare:
            if node.value in ORDER_NAMES:
                found.add(f"string {node.value!r}")
    return found


def data_layer_imports(found: set[str]) -> set[str]:
    """The ``aegis.data`` modules among ``found`` (the package itself included)."""
    return {name for name in found if name == "aegis.data" or name.startswith("aegis.data.")}


def _within(name: str, modules: tuple[str, ...]) -> bool:
    return any(name == m or name.startswith(m + ".") for m in modules)


def imported_modules(source: str, *, package: str = "") -> set[str]:
    """Every module name a source file imports, in any form (rule 1), plus
    ``DYNAMIC_IMPORT`` when it uses any dynamic-import machinery (rule 2).

    A literal first argument of an ``import_module``/``__import__`` call is
    recorded by name, and so is a string literal that is a module path under
    ``aegis.execution`` — a docstring that mentions it is neither.
    """
    tree = ast.parse(source)
    found = _static_imports(tree, package)
    if dynamic_machinery(source, package=package):
        found.add(DYNAMIC_IMPORT)
    bare = _bare_strings(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else func.id if isinstance(func, ast.Name) else None
            first = node.args[0] if node.args else None
            if name in {"import_module", "__import__"} and isinstance(first, ast.Constant) and isinstance(first.value, str):
                found.add(first.value)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in bare:
            if node.value == EXECUTION or node.value.startswith(EXECUTION + "."):
                found.add(node.value)
    return found


def imports(found: set[str], module: str) -> bool:
    """Whether ``module`` or anything under it is in ``found`` — or ``found``
    holds a dynamic import, whose target cannot be known statically."""
    if DYNAMIC_IMPORT in found:
        return True
    return any(name == module or name.startswith(module + ".") for name in found)


def _source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _imports_of(path: Path) -> set[str]:
    return imported_modules(_source(path), package=_package_of(path))


ALL_SOURCES = [*_brain_sources(), CLI_BRAIN]


def _id(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT))


# --- the checkers themselves --------------------------------------------------


class TestImportChecker:
    @pytest.mark.parametrize(
        "snippet",
        [
            "import aegis.execution",
            "import aegis.execution as ex",
            "import os, aegis.execution",
            "from aegis.execution import submit_order",
            "from aegis.execution.broker import Broker",
            "from aegis import execution",
            "from aegis import store, execution",
            "from aegis import execution as ex",
            "import importlib\nimportlib.import_module('aegis.execution')",
            "import importlib\nimportlib.import_module('aegis.execution.broker')",
            "from importlib import import_module\nimport_module('aegis.execution')",
            "__import__('aegis.execution')",
            "name = 'aegis.execution'\nimport importlib\nmod = importlib.import_module(name)",
            "import importlib\nimportlib.import_module('aegis.' + 'execution')",
            "import importlib\nname = 'aegis.execu' + 'tion'\nimportlib.import_module(name)",
            "__import__(''.join(['aegis', '.execution']))",
            "def later():\n    from aegis.execution import submit_order\n    return submit_order",
            "try:\n    import aegis.execution\nexcept ImportError:\n    pass",
            "import pkgutil\npkgutil.resolve_name('aegis.execution:Executor')",
            "def later():\n    import pkgutil\n    return pkgutil.resolve_name('aegis.execution:Executor')",
        ],
    )
    def test_catches_every_import_form(self, snippet):
        assert imports(imported_modules(snippet, package="aegis.brain"), EXECUTION), snippet

    @pytest.mark.parametrize(
        "snippet, package",
        [
            ("from .. import execution", "aegis.brain"),
            ("from ..execution import submit_order", "aegis.brain"),
            ("from ... import execution", "aegis.brain.prompts"),
            ("from ...execution.broker import Broker", "aegis.brain.prompts"),
        ],
    )
    def test_catches_relative_imports(self, snippet, package):
        assert imports(imported_modules(snippet, package=package), EXECUTION), snippet

    @pytest.mark.parametrize(
        "snippet",
        [
            '"""No module under aegis.brain may import aegis.execution."""',
            "# import aegis.execution",
            "from aegis.store import insert_proposal",
            "from aegis.brain.proposal import run_proposal",
            "import aegis.executionary",
            "from . import execution",  # resolves to aegis.brain.execution, not aegis.execution
            "note = 'the brain never touches aegis.execution: see the README'",  # rule 3, not an import
        ],
    )
    def test_does_not_flag_mentions_or_look_alikes(self, snippet):
        found = imported_modules(snippet, package="aegis.brain")
        assert not imports(found, EXECUTION), snippet
        assert DYNAMIC_IMPORT not in found

    def test_package_resolution(self):
        assert _package_of(BRAIN_DIR / "cycle.py") == "aegis.brain"
        assert _package_of(BRAIN_DIR / "__init__.py") == "aegis.brain"
        assert _package_of(BRAIN_DIR / "prompts" / "__init__.py") == "aegis.brain.prompts"
        assert _package_of(CLI_BRAIN) == "aegis.cli"
        assert _resolve("llm", 1, "aegis.brain") == "aegis.brain.llm"
        assert _resolve(None, 2, "aegis.brain") == "aegis"
        assert _resolve("execution", 2, "aegis.brain.prompts") == "aegis.brain.execution"

    def test_the_walk_covers_the_whole_package(self):
        found = {str(path.relative_to(BRAIN_DIR)) for path in _brain_sources()}
        assert EXPECTED_MODULES <= found, EXPECTED_MODULES - found
        assert EXECUTION_DIR.is_dir()  # the rule has a real target
        assert CLI_BRAIN.exists()
        # every rule is parametrised over ALL_SOURCES: the brain CLI must be in it
        assert CLI_BRAIN in ALL_SOURCES and set(_brain_sources()) <= set(ALL_SOURCES)


class TestDynamicMachineryChecker:
    @pytest.mark.parametrize(
        "snippet",
        [
            "import importlib",
            "import importlib.util",
            "import importlib.machinery as machinery",
            "from importlib import util",
            "from importlib import import_module",
            "from importlib.machinery import SourceFileLoader",
            # a file-path load: no import statement names the package, no literal is its dotted name
            "import importlib.util as u\nspec = u.spec_from_file_location('x', p)\nm = u.module_from_spec(spec)\n"
            "spec.loader.exec_module(m)",
            "import pkgutil",
            "import pkgutil\npkgutil.resolve_name('aegis.execution:Executor')",
            "import runpy\nrunpy.run_path(str(ROOT / 'aegis' / 'execution' / 'base.py'))",
            "import runpy\nrunpy.run_module('aegis.execution')",
            "import imp",
            "import zipimport",
            "__import__('os')",
            "exec('import os')",
            "exec(compile(source, 'x', 'exec'))",
            "value = eval('1 + 1')",
            "code = compile('x = 1', 'x', 'exec')",
            "run = eval",
            "import builtins\nbuiltins.exec('x = 1')",
            "__builtins__.__import__('os')",
            "getattr(builtins, '__import__')('os')",
            "getattr(loader, 'exec_module')(module)",
            "loader.load_module('x')",
            "resolve = pkg.resolve_name",
            "m = mod.import_module",
            "def later():\n    import importlib\n    return importlib.import_module(name)",
            # a builtin reached through a computed attribute name (verifier E1, E5)
            "def _later():\n    import builtins\n"
            "    return getattr(builtins, '__imp' + 'ort__')('aegis.' + 'execu' + 'tion')",
            "import builtins, os\n"
            "_p = os.path.join(os.path.dirname(__file__), os.pardir, 'execu' + 'tion', 'base.py')\n"
            "with open(_p) as _f:\n"
            "    getattr(builtins, 'ex' + 'ec')(getattr(builtins, 'comp' + 'ile')(_f.read(), 'x', 'ex' + 'ec'), {})",
            "import builtins as _b\ngetattr(_b, 'ex' + 'ec')(open('aegis/exe' + 'cution/base.py').read(), {})",
            "from builtins import exec as run",
            "getattr(__builtins__, 'ex' + 'ec')(source)",
            "getattr(thing, name)",
            "hasattr(thing, name)",
            "setattr(thing, name, value)",
            "delattr(thing, name)",
            "getattr(*args)",
            "from functools import reduce\nimport sys\nreduce(getattr, ['mod' + 'ules'], sys)",
            "fetch = getattr",
            # the ways back to the builtins or a module table without an import
            "print.__self__.exec(source)",
            "len.__self__",
            "run = vars(len.__self__)['ex' + 'ec']",
            "globals()['__buil' + 'tins__']",
            "locals()",
            "handler.__globals__['__buil' + 'tins__']",
            "import sys\nsys.modules['buil' + 'tins']",
            "from sys import modules as table",
            "table = sys.__dict__",
            "object.__getattribute__(sys, 'mod' + 'ules')",
            "().__class__.__base__.__subclasses__()",
            "code = handler.__code__",
            "loader = __loader__",
            "loader = __spec__.loader",
            "import operator\nrun = operator.attrgetter('ex' + 'ec')",
            "from operator import methodcaller",
            "import inspect\ninspect.getattr_static(thing, name)",
            "import types\ntypes.FunctionType(code, {})()",
            "table = getattr(sys, 'modules')",
            # a module or code object from bytes, raw memory, an interpreter, an import by name
            "import pickle as _pk\n_pk.loads(b'caegis.execution.base\\nExecutor\\n.')",
            "import _pickle",
            "import shelve",
            "import marshal\ncode = marshal.loads(blob)",
            "import ctypes",
            "import code\ncode.InteractiveInterpreter().runsource(source)",
            "import codeop",
            "import pydoc\npydoc.locate('aegis.' + 'execution')",
            # each entry on its own, so dropping any one from its list fails here (tests-lens D4)
            "import runpy",
            "code = compile(source, name, mode)",
            "spec = util.spec_from_file_location(name, path)",
            "module = util.module_from_spec(spec)",
            "helpers.run_module(name)",
            "helpers.run_path(path)",
            # a dotted name resolved by a library, or a string annotation evaluated
            "from pydantic import ImportString\nTypeAdapter(ImportString).validate_python('aegis' + '.exe' + 'cution.base')",
            "import pydantic\nadapter = TypeAdapter(pydantic.ImportString)",
            "import typing\ntyping.get_type_hints(holder)",
            "from typing import get_type_hints",
            "hints = get_annotations(holder, eval_str=True)",
            "ref = ForwardRef(name)",
            "import annotationlib",
            "from unittest import mock\nmock.patch('aegis' + '.exe' + 'cution.base.Executor').start()",
            "import unittest.mock",
            "import logging.config\nlogging.config.dictConfig(conf)",
            # a statement string run by another name than exec
            "import timeit as _ti\n_ti.timeit('import aegis.ex' + 'ecution.base', number=1)",
            "import bdb\nbdb.Bdb().run(source)",
            "import pdb",
            "import profile",
            "import cProfile\ncProfile.run(source)",
            "import trace\ntrace.Trace(count=False).run(source)",
            "import doctest",
            # another interpreter: nothing in this one sees what it loads
            "import subprocess, sys\nsubprocess.run([sys.executable, '-c', 'import aegis.exec' + 'ution'])",
            "import os\nos.system(cmd)",
            "from os import system",
            "import posix",
            "import multiprocessing",
            # multiprocessing without its name (the x26 mutant)
            "import concurrent.futures.process as _cfp\n_cfp.mp.get_context('spawn').Process(target=run).start()",
            "from concurrent import futures",
            "import _posixsubprocess",
            "import _winapi",
            "import pty",
            "import webbrowser",
            "handle = Popen(argv)",
            "child.popen(cmd)",
            "posix_module.posix_spawn(path, argv, env)",
            "pid = worker.fork()",
            "os_module.execv(path, argv)",
            "loop.create_subprocess_exec(*argv)",
            "coro = loop.subprocess_exec(factory, *argv)",
            "coro = loop.subprocess_shell(factory, cmd)",
            "pool = futures.ProcessPoolExecutor()",
        ],
    )
    def test_catches_every_dynamic_form(self, snippet):
        assert dynamic_machinery(snippet, package="aegis.brain"), snippet
        found = imported_modules(snippet, package="aegis.brain")
        assert DYNAMIC_IMPORT in found
        assert imports(found, EXECUTION) and imports(found, "anthropic"), snippet

    @pytest.mark.parametrize(
        "snippet",
        [
            "import re\n_PATTERN = re.compile(r'x')",  # a method that shares a builtin's name
            "pattern.compile('x')",
            '"""Never exec, eval or compile model output; never importlib."""',
            "text = 'evaluate the execution of the plan'",
            "from aegis.store import add_reasoning",
            "import json\njson.loads('{}')",
            "execute = True",
            # getattr & co. with a literal name are how the brain reads SDK objects
            "response = getattr(exc, 'response', None)",
            "if hasattr(item, 'text'):\n    setattr(item, 'seen', True)",
            "import sys\nprint('x', file=sys.stderr)\nsys.exit(1)",
            "import types\nns = types.SimpleNamespace(a=1)",
            "text = 'the modules of the brain'",
            # llm.py's request carries a system prompt: a field, not os.system
            "kwargs = {'system': request.system}",
            "import sys\nsys.stdout.reconfigure(errors='backslashreplace')",
        ],
    )
    def test_does_not_flag_ordinary_code(self, snippet):
        assert dynamic_machinery(snippet, package="aegis.brain") == set(), snippet


class TestExecutionLiteralChecker:
    @pytest.mark.parametrize(
        "snippet",
        [
            "p = 'aegis.execution'",
            "p = 'aegis.execution.base'",
            "p = 'aegis.execution:Executor'",
            "p = 'aegis/execution/base.py'",
            "p = 'aegis\\\\execution\\\\base.py'",
            "p = ROOT / 'execution/base.py'",
            "p = 'AEGIS.EXECUTION'",
            "note = 'the brain never touches aegis.execution: see the README'",
            "f(f'{root}/aegis/execution/{name}')",
            "def later():\n    return 'aegis.execution'",
            # bytes literals too: a pickle names a module and a class
            "pickle.loads(b'caegis.execution.base\\nExecutor\\n.')",
            "p = b'aegis/execution/base.py'",
            "p = rb'aegis\\execution'",
            "p = '..\\\\execution\\\\base.py'",  # a Windows-style path with no 'aegis' in it
        ],
    )
    def test_catches_execution_paths(self, snippet):
        assert execution_literals(snippet), snippet

    @pytest.mark.parametrize(
        "snippet",
        [
            '"""No module under aegis.brain may import aegis.execution."""',
            'def f():\n    """Nothing here touches aegis.execution."""\n    return 1',
            'X = 1\n"""An attribute docstring naming aegis/execution/base.py."""',
            "text = 'zero execution authority'",
            "p = 'aegis.store'",
            "# 'aegis.execution' in a comment",
            "blob = b'zero execution authority'",
        ],
    )
    def test_docstrings_and_plain_words_are_fine(self, snippet):
        assert execution_literals(snippet) == set(), snippet


class TestBrokerChecker:
    @pytest.mark.parametrize(
        "snippet, package",
        [
            ("from aegis.data.clients import trading_client", "aegis.brain"),
            ("import aegis.data.clients", "aegis.brain"),
            ("from aegis.data import clients", "aegis.brain"),
            ("from ..data.clients import trading_client", "aegis.brain"),
            ("import alpaca", "aegis.brain"),
            ("from alpaca.trading.client import TradingClient", "aegis.brain"),
            ("from alpaca.trading.requests import MarketOrderRequest", "aegis.brain"),
            ("client.submit_order(order)", "aegis.brain"),
            ("client.cancel_orders()", "aegis.brain"),
            ("client.cancel_order_by_id(order_id)", "aegis.brain"),
            ("client.replace_order_by_id(order_id, request)", "aegis.brain"),
            ("client.close_position(symbol)", "aegis.brain"),
            ("client.close_all_positions(cancel_orders=True)", "aegis.brain"),
            ("client.exercise_options_position(symbol)", "aegis.brain"),
            ("make = trading_client", "aegis.brain"),
            ("method = 'submit_order'", "aegis.brain"),
        ],
    )
    def test_catches_every_broker_door(self, snippet, package):
        assert broker_access(snippet, package=package), snippet

    @pytest.mark.parametrize(
        "snippet",
        [
            '"""The brain never calls submit_order or trading_client()."""',
            "from aegis.data.models import OptionType",
            "text = 'no order is ever submitted'",
            "from aegis.store.repo import insert_proposal",
            "def later():\n    return 'orders are the policy engine\\'s business'",
        ],
    )
    def test_does_not_flag_ordinary_code(self, snippet):
        assert broker_access(snippet, package="aegis.brain") == set(), snippet

    @pytest.mark.parametrize(
        "snippet, reaches_beyond_models",
        [
            ("from aegis.data.models import utcnow", False),
            ("from aegis.data.errors import DataError", False),
            ("from aegis.data.cache import TTLCache", True),
            ("from aegis.data import market", True),
            ("import aegis.data", True),  # the package itself: a door to whatever it exports
            ("from ..data.news import get_news", True),
        ],
    )
    def test_the_data_layer_scope(self, snippet, reaches_beyond_models):
        found = data_layer_imports(imported_modules(snippet, package="aegis.brain"))
        assert any(not _within(name, DATA_MODELS) for name in found) is reaches_beyond_models, found


# --- the rules ----------------------------------------------------------------


class TestGovernance:
    @pytest.mark.parametrize("path", ALL_SOURCES, ids=_id)
    def test_no_brain_module_imports_execution(self, path):
        assert not imports(_imports_of(path), EXECUTION), f"{path} imports {EXECUTION}"

    @pytest.mark.parametrize("path", ALL_SOURCES, ids=_id)
    def test_no_brain_module_uses_dynamic_import_machinery(self, path):
        """There is no legitimate dynamic import under the brain: a computed name
        or a file-path load is exactly what the static rule cannot see through."""
        assert dynamic_machinery(_source(path), package=_package_of(path)) == set(), path

    @pytest.mark.parametrize("path", ALL_SOURCES, ids=_id)
    def test_no_brain_module_names_an_execution_path(self, path):
        assert execution_literals(_source(path)) == set(), path

    @pytest.mark.parametrize("path", ALL_SOURCES, ids=_id)
    def test_only_llm_imports_anthropic(self, path):
        uses_sdk = imports(_imports_of(path), "anthropic")
        assert uses_sdk == (path == BRAIN_DIR / "llm.py"), path

    def test_llm_really_does_import_anthropic(self):
        assert imports(_imports_of(BRAIN_DIR / "llm.py"), "anthropic")  # the rule is not vacuous

    @pytest.mark.parametrize("path", _brain_sources(), ids=lambda p: str(p.relative_to(BRAIN_DIR)))
    def test_only_snapshot_imports_the_live_fetchers(self, path):
        found = _imports_of(path)
        touches = [module for module in LIVE_FETCHERS if imports(found, module)]
        if path == BRAIN_DIR / "snapshot.py":
            assert set(touches) == set(LIVE_FETCHERS)
        else:
            assert touches == [], f"{path} imports {touches}"

    @pytest.mark.parametrize("path", ALL_SOURCES, ids=_id)
    def test_no_brain_module_reaches_the_broker(self, path):
        """Paper-only still places orders: no brain module may hold the broker client."""
        assert broker_access(_source(path), package=_package_of(path)) == set(), path

    @pytest.mark.parametrize("path", ALL_SOURCES, ids=_id)
    def test_only_snapshot_imports_the_data_layer_beyond_its_models(self, path):
        found = data_layer_imports(_imports_of(path))
        allowed = (*DATA_MODELS, *LIVE_FETCHERS) if path == BRAIN_DIR / "snapshot.py" else DATA_MODELS
        assert {name for name in found if not _within(name, allowed)} == set(), path

    def test_the_package_is_unimportable_in_every_brain_test(self):
        """tests/conftest.py refuses aegis.execution in-process for every test_brain_* module
        (the snapshot and stage tests included, not only the cycle's)."""
        with pytest.raises(ImportError, match="the brain tried to import aegis.execution"):
            __import__("aegis.execution.base")
        assert not any(name == EXECUTION or name.startswith(EXECUTION + ".") for name in sys.modules)

    def test_cli_brain_reaches_live_data_only_through_the_snapshot_builder(self):
        found = _imports_of(CLI_BRAIN)
        assert not any(imports(found, module) for module in LIVE_FETCHERS)
        assert "aegis.brain.snapshot.build_market_snapshot" in found


# --- the runtime rule ---------------------------------------------------------

# Runs in a fresh interpreter: an audit hook goes in before anything else is
# imported and records every import, open, compile and exec that touches
# aegis/execution — by module name, or by file path whatever name a loader
# gives it — and every child process started. A path is compared the way the
# filesystem resolves it: made absolute (a relative path is relative to the
# CWD, the repo root here), ``..`` and symlinks resolved, casefolded (APFS and
# NTFS match names case-insensitively). The hook also notes that it saw the
# brain's own modules load (``seen``), which proves it was in place before
# them. Every order method of Alpaca's TradingClient, and
# aegis.data.clients.trading_client, then records and refuses. Then the brain
# runs the way production does and the way tests do: one FakeLLM cycle; the
# real AnthropicLLM over a stand-in SDK client (success after a retried 429,
# a refusal, a truncation); the no-trade and halted outcomes; every brain CLI
# command with an injected FakeLLM and snapshot — ``once``, ``stage`` on both
# paths (prior stages from a ``--cycle``, and run first), ``usage`` — and
# ``once`` and ``stage scan`` once more through the DEFAULT snapshot builder,
# the data layer's fetchers replaced in-process by fixture-backed fakes.
# ``argv[3]``, when given, is extra code the self-tests use to prove the
# harness catches a load: run after all that, or — with ``argv[4] ==
# "during"`` — inside the FakeLLM cycle's first call.
_RUNTIME_SCRIPT = r'''
import io, json, os, sys, warnings
from contextlib import redirect_stdout
from pathlib import Path


def _resolved(path):
    """``path`` as the filesystem resolves it (its text as given, if it cannot be)."""
    try:
        return os.path.realpath(os.path.abspath(path))
    except (OSError, ValueError):
        return path


_EXECUTION = _resolved(sys.argv[1]).casefold()
_WATCHED = ("aegis.brain.cycle", "aegis.cli.brain")
_PROCESS_EVENTS = (
    "subprocess.Popen", "os.system", "os.exec", "os.posix_spawn", "os.spawn", "os.fork",
    "os.forkpty", "os.startfile", "pty.spawn",
    # the only event a multiprocessing "spawn" or "forkserver" start raises (POSIX, Windows)
    "_posixsubprocess.fork_exec", "_winapi.CreateProcess",
)
touched, spawned, seen = [], [], []


def _path_args(event, args):
    """The arguments of ``event`` that name a file: what ``open`` opens, the
    filename ``compile`` is given, the file an ``exec``'d code object came from."""
    if event == "open":
        candidates = args[:1]
    elif event == "compile":
        candidates = args[1:2]
    elif event == "exec":
        candidates = [getattr(arg, "co_filename", None) for arg in args[:1]]
    else:  # import: (module, filename, sys.path, sys.meta_path, sys.path_hooks)
        candidates = args[1:2]
    for arg in candidates:
        if isinstance(arg, (str, bytes, os.PathLike)):
            try:
                yield os.fsdecode(arg)
            except (TypeError, ValueError):
                continue


def audit(event, args):
    if event in _PROCESS_EVENTS:
        spawned.append(f"{event}: {str(args[:2])[:200]}")
        return
    if event not in ("import", "open", "compile", "exec"):
        return
    if event == "import" and isinstance(args[0], str):
        if args[0] in _WATCHED:
            seen.append(args[0])
        if args[0] == "aegis.execution" or args[0].startswith("aegis.execution."):
            touched.append(f"import: {args[0]}")
            return
    for path in _path_args(event, args):
        resolved = _resolved(path)
        folded = resolved.casefold()
        if folded == _EXECUTION or folded.startswith(_EXECUTION + os.sep):
            touched.append(f"{event}: {path} -> {resolved}")
            return


sys.addaudithook(audit)

# Every door the broker SDK has to an order records and refuses — before the brain loads.
orders = []


def _refused(name):
    def refuse(*args, **kwargs):
        orders.append(name)
        raise RuntimeError(f"order capability used: {name}")
    return refuse


with warnings.catch_warnings():  # alpaca-py's own websockets.legacy deprecation
    warnings.simplefilter("ignore", DeprecationWarning)
    import aegis.data.clients
    from alpaca.trading.client import TradingClient
for _name in (
    "submit_order", "replace_order_by_id", "cancel_order_by_id", "cancel_orders",
    "close_position", "close_all_positions", "exercise_options_position",
):
    setattr(TradingClient, _name, _refused(_name))
aegis.data.clients.trading_client = _refused("trading_client")

import aegis.brain, aegis.brain.cycle, aegis.brain.llm, aegis.brain.snapshot
import aegis.brain.testing, aegis.brain.prompts, aegis.cli.brain
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
import anthropic, httpx2
from aegis.brain.errors import BrainError
from aegis.brain.llm import AnthropicLLM
from aegis.brain.models import MarketSnapshot
from aegis.brain.testing import FIXTURES_DIR, FakeLLM
from aegis.config import get_config
from aegis.data.models import AccountState, Bar, ChainSnapshot, MarketClock, NewsItem, OptionSnapshot, Quote
from aegis.store.db import open_store

extra = sys.argv[3] if len(sys.argv) > 3 else None
during = len(sys.argv) > 4 and sys.argv[4] == "during"
scope = {"EXECUTION_DIR": sys.argv[1], "TMP": sys.argv[2]}


class _DuringFake(FakeLLM):
    """A FakeLLM that, asked to, runs the self-test's extra code inside its first call."""

    def complete(self, request):
        global extra
        if during and extra is not None:
            code, extra = extra, None
            exec(code, scope)
        return super().complete(request)


ok = {"scan": ["scan_ok.json"], "thesis": ["thesis_ok.json"], "proposal": ["proposal_ok.json"]}
no_idea = {"scan": ["scan_ok.json"], "thesis": ["thesis_empty.json"]}
text = {name: (FIXTURES_DIR / name).read_text(encoding="utf-8") for name in ok["scan"] + ok["thesis"] + ok["proposal"]}
snapshot = MarketSnapshot.model_validate(
    json.loads((FIXTURES_DIR / "market_snapshot.json").read_text(encoding="utf-8"))
)
db = Path(sys.argv[2])
config = get_config()
conn = open_store(db / "brain.db")
cycle = aegis.brain.cycle.run_cycle(conn, _DuringFake.from_fixtures(ok), snapshot=snapshot)
conn.close()

# The production client: AnthropicLLM over a stand-in for the SDK's messages API.
def _message(body, stop_reason="end_turn"):
    refusal = SimpleNamespace(category="policy", explanation="declined") if stop_reason == "refusal" else None
    usage = SimpleNamespace(input_tokens=10, output_tokens=5, cache_creation_input_tokens=None,
                            cache_read_input_tokens=None)
    return SimpleNamespace(content=[SimpleNamespace(type="text", text=body)], stop_reason=stop_reason,
                           stop_details=refusal, usage=usage, model="claude-harness", _request_id="req_h")


class _Sdk:
    def __init__(self, *outcomes):
        self.outcomes, self.messages = list(outcomes), self

    def create(self, **kwargs):
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


rate_limited = anthropic.RateLimitError(
    "rate limited", body=None,
    response=httpx2.Response(429, request=httpx2.Request("POST", "https://api.anthropic.com/v1/messages")),
)
branches = {}


def _branch(name, llm, cfg=config):
    conn = open_store(db / f"{name}.db")
    try:
        branches[name] = aegis.brain.cycle.run_cycle(conn, llm, config=cfg, snapshot=snapshot).outcome
    except BrainError as exc:
        branches[name] = f"failed: {type(exc).__name__}"
    finally:
        conn.close()


def _anthropic(*outcomes):
    return AnthropicLLM(client=_Sdk(*outcomes), sleep=lambda seconds: None)


_branch("anthropic", _anthropic(rate_limited, *(_message(text[name]) for name in ("scan_ok.json", "thesis_ok.json", "proposal_ok.json"))))
_branch("refusal", _anthropic(_message("", "refusal")))
_branch("max_tokens", _anthropic(_message('{"symbols": [', "max_tokens")))
_branch("no_trade", FakeLLM.from_fixtures(no_idea))
tiny = config.model_copy(update={"brain": config.brain.model_copy(update={"daily_token_budget": 1})})
_branch("halted", FakeLLM.from_fixtures(ok), tiny)

# The live snapshot path, fetchers replaced in-process by fixture-backed fakes (no network).
fixtures = FIXTURES_DIR.parent
now = datetime.now(timezone.utc)
expiry = (now + timedelta(days=30)).date()


def _fixture(name):
    return json.loads((fixtures / name).read_text(encoding="utf-8"))


def _chain(symbol, *args, **kwargs):
    full = _fixture("option_snapshot_full.json")
    contracts = [
        OptionSnapshot.from_alpaca(f"{symbol}{expiry:%y%m%d}C{int(strike * 1000):08d}", symbol, full)
        for strike in (630.0, 640.0, 650.0)
    ]
    return ChainSnapshot(underlying=symbol, expiration=expiry, spot=638.9, contracts=contracts)


fetchers = {
    "get_market_clock": lambda: MarketClock(is_open=True, next_open=now + timedelta(hours=16),
                                            next_close=now + timedelta(hours=2)),
    "get_account_state": lambda: AccountState.from_alpaca(_fixture("account.json")["account"],
                                                          _fixture("account.json")["positions"]),
    "get_spot": lambda symbol: Quote.from_alpaca(symbol, quote=_fixture("stock_quote.json")["quote"],
                                                 trade=_fixture("stock_quote.json")["trade"]),
    "get_bars": lambda symbol, *a, **k: [Bar.from_alpaca(symbol, b) for b in _fixture("bars.json")["bars"]],
    "get_option_chain": _chain,
    "get_news": lambda symbol, *a, **k: [NewsItem.from_alpaca(n) for n in _fixture("news.json")["news"]],
}
for name, fake in fetchers.items():
    setattr(aegis.brain.snapshot, name, fake)

fixture_builder = lambda symbols=None, **_: snapshot
commands = {  # name: (argv, canned outputs, snapshot builder — None is the default live one)
    "once": (["once"], ok, fixture_builder),
    "stage scan": (["stage", "scan"], {"scan": ok["scan"]}, fixture_builder),
    "stage thesis --cycle": (["stage", "thesis", "--cycle", cycle.cycle_id], {"thesis": ok["thesis"]}, None),
    "stage proposal": (["stage", "proposal"], ok, fixture_builder),
    "usage": (["usage"], {}, None),
    "once (live snapshot)": (["once"], no_idea, None),
    "stage scan (live snapshot)": (["stage", "scan"], {"scan": ok["scan"]}, None),
}
cli = {}
with redirect_stdout(io.StringIO()):
    for name, (argv, canned, builder) in commands.items():
        cli[name] = aegis.cli.brain.main(
            [*argv, "--db", str(db / "brain.db")], llm=FakeLLM.from_fixtures(canned), snapshot_builder=builder
        )
if extra is not None and not during:
    exec(extra, scope)
loaded = sorted(m for m in sys.modules if m == "aegis.execution" or m.startswith("aegis.execution."))
print(json.dumps({
    "outcome": cycle.outcome, "cli": cli, "branches": branches, "seen": sorted(set(seen)),
    "loaded": loaded, "touched": touched, "spawned": spawned, "orders": orders,
}))
failed = cycle.outcome != "proposal" or any(code != 0 for code in cli.values())
sys.exit(1 if loaded or touched or spawned or orders or failed else 0)
'''

CLI_OK = {
    "once": 0, "stage scan": 0, "stage thesis --cycle": 0, "stage proposal": 0, "usage": 0,
    "once (live snapshot)": 0, "stage scan (live snapshot)": 0,
}
BRANCHES_OK = {
    "anthropic": "proposal", "refusal": "failed: LLMResponseError", "max_tokens": "failed: LLMResponseError",
    "no_trade": "no_trade", "halted": "halted",
}
REPORT_OK = {
    "outcome": "proposal", "cli": CLI_OK, "branches": BRANCHES_OK, "seen": ["aegis.brain.cycle", "aegis.cli.brain"],
    "loaded": [], "touched": [], "spawned": [], "orders": [],
}

# A path that differs only in case names the same file here (APFS, NTFS)?
_CASE_INSENSITIVE_FS = Path(str(EXECUTION_DIR / "base.py").swapcase()).exists()


def _run_runtime_harness(tmp_path: Path, extra: str | None = None, *, during: bool = False) -> tuple[int, dict]:
    argv = [sys.executable, "-c", _RUNTIME_SCRIPT, str(EXECUTION_DIR), str(tmp_path)]
    if extra is not None:
        argv += [extra, "during" if during else "after"]
    # never write bytecode: a self-test load must not leave a __pycache__ under aegis/execution
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    result = subprocess.run(argv, cwd=REPO_ROOT, capture_output=True, text=True, timeout=120, env=env)
    lines = result.stdout.strip().splitlines()
    assert lines, result.stderr
    return result.returncode, json.loads(lines[-1])


def _pycache_state() -> dict[str, int] | None:
    """What ``aegis/execution/__pycache__`` holds (file name -> mtime), None when absent —
    compared before and after a harness run, since an ordinary import elsewhere (an IDE, a
    reviewer, the Phase 6 tests) may leave one there that is none of the harness's doing."""
    cache = EXECUTION_DIR / "__pycache__"
    if not cache.is_dir():
        return None
    return {path.name: path.stat().st_mtime_ns for path in cache.iterdir()}


def _names_execution(report: dict) -> bool:
    spellings = {str(EXECUTION_DIR).casefold(), os.path.realpath(EXECUTION_DIR).casefold()}
    return any(
        EXECUTION in item or any(dir_ in item.casefold() for dir_ in spellings) for item in report["touched"]
    )


# A module registered under the package's name from memory: no file is opened and it
# leaves sys.modules again, so only the hook's module-name branch can see it.
_IN_MEMORY_LOAD = (
    "import sys, types\n"
    "class _Finder:\n"
    "    def find_spec(self, name, path=None, target=None):\n"
    "        if name == 'aegis.exec' + 'ution':\n"
    "            from importlib.machinery import ModuleSpec\n"
    "            return ModuleSpec(name, self)\n"
    "    def create_module(self, spec):\n"
    "        return types.ModuleType(spec.name)\n"
    "    def exec_module(self, module):\n"
    "        pass\n"
    "sys.meta_path.insert(0, _Finder())\n"
    "import aegis.execution\n"
    "del sys.modules['aegis.execution']\n"
)


class TestRuntime:
    def test_a_cycle_and_the_cli_never_touch_execution(self, tmp_path):
        """Runtime confirmation of the static rules, in a fresh interpreter: a lazy,
        computed, file-path or child-process load inside the cycle, the production client,
        the live snapshot path or any CLI command would be seen by the audit hook or left
        in ``sys.modules``, and an order method would record itself. ``seen`` proves the
        hook was installed before the brain's modules loaded."""
        code, report = _run_runtime_harness(tmp_path)
        assert report == REPORT_OK
        assert code == 0

    @pytest.mark.parametrize(
        "extra, expect_loaded",
        [
            ("import aegis.execution.base", True),
            (
                "import importlib.util as u\n"
                "spec = u.spec_from_file_location('x', EXECUTION_DIR + '/base.py')\n"
                "module = u.module_from_spec(spec)\n"
                "spec.loader.exec_module(module)",
                False,  # registered under 'x', never in sys.modules as aegis.execution*
            ),
            ("import runpy\nrunpy.run_path(EXECUTION_DIR + '/base.py')", False),
            ("exec(compile(open(EXECUTION_DIR + '/base.py').read(), 'x', 'exec'), {})", False),
            # the same file by another spelling of its path (verifier E5/E6, mut1)
            ("exec(open('aegis/execution/base.py').read(), {})", False),  # relative to the CWD
            (
                "import os\n"
                "path = os.path.join(os.path.dirname(EXECUTION_DIR), 'brain', os.pardir, 'execution', 'base.py')\n"
                "exec(compile(open(path).read(), 'x', 'exec'), {})",
                False,
            ),
            (
                "import os\n"
                "link = os.path.join(TMP, 'innocent.py')\n"
                "os.symlink(os.path.join(EXECUTION_DIR, 'base.py'), link)\n"
                "exec(open(link).read(), {})",
                False,
            ),
            pytest.param(
                "import os\nexec(open(os.path.join(EXECUTION_DIR, 'base.py').swapcase()).read(), {})",
                False,
                marks=pytest.mark.skipif(not _CASE_INSENSITIVE_FS, reason="case-sensitive filesystem"),
            ),
            # a module named inside bytes: rule 3 sees the literal, this sees the import
            ("import pickle\npickle.loads(b'caegis.execution.base\\nExecutor\\n.')", True),
            # by name alone: no file, gone from sys.modules again (the name branch on its own)
            (_IN_MEMORY_LOAD, False),
        ],
        ids=[
            "import", "spec-from-file", "run-path", "exec-compile",
            "relative-path", "dot-dot-path", "symlink", "other-case", "pickle-bytes", "in-memory-module",
        ],
    )
    def test_the_harness_catches_a_load_by_any_name(self, tmp_path, extra, expect_loaded):
        """Self-test: the harness fails when the run loads aegis/execution —
        by its module name, or by file path under another name entirely and
        however the path is spelled."""
        before = _pycache_state()
        code, report = _run_runtime_harness(tmp_path, extra)
        assert code == 1
        assert report["outcome"] == "proposal" and report["cli"] == CLI_OK
        assert report["touched"] and _names_execution(report), report["touched"]
        assert bool(report["loaded"]) is expect_loaded
        # the harness itself writes no bytecode under aegis/execution
        assert _pycache_state() == before

    def test_the_harness_catches_a_load_during_the_cycle(self, tmp_path):
        """Self-test: a load made while the cycle runs (inside the FakeLLM's first call), not
        only one made after it — the hook is active while the code under test runs."""
        extra = "exec(compile(open(EXECUTION_DIR + '/base.py').read(), 'x', 'exec'), {})"
        code, report = _run_runtime_harness(tmp_path, extra, during=True)
        assert code == 1
        assert report["outcome"] == "proposal" and report["seen"] == REPORT_OK["seen"]
        assert report["touched"] and _names_execution(report), report["touched"]

    @pytest.mark.parametrize(
        "extra, event",
        [
            ("import subprocess, sys\nsubprocess.run([sys.executable, '-c', 'pass'], check=True)", "subprocess.Popen"),
            ("import os\nos.system('true')", "os.system"),
            # a "spawn" start raises neither of the above, only the raw fork_exec / CreateProcess
            (
                "import multiprocessing as mp\np = mp.get_context('spawn').Process(target=print)\np.start(); p.join()",
                "_winapi.CreateProcess" if sys.platform == "win32" else "_posixsubprocess.fork_exec",
            ),
        ],
        ids=["subprocess", "os-system", "multiprocessing-spawn"],
    )
    def test_the_harness_catches_a_child_process(self, tmp_path, extra, event):
        """Self-test: a child interpreter could load the package where no audit hook of this
        one can see, so starting any process — by subprocess, the shell or multiprocessing —
        fails the run."""
        code, report = _run_runtime_harness(tmp_path, extra)
        assert code == 1
        assert report["outcome"] == "proposal" and report["cli"] == CLI_OK
        assert any(item.startswith(event) for item in report["spawned"]), report["spawned"]

    @pytest.mark.parametrize(
        "extra, order",
        [
            (
                "from alpaca.trading.client import TradingClient\n"
                "try:\n    TradingClient.submit_order(None, None)\nexcept RuntimeError:\n    pass",
                "submit_order",
            ),
            (
                "import aegis.data.clients as clients\n"
                "try:\n    clients.trading_client()\nexcept RuntimeError:\n    pass",
                "trading_client",
            ),
        ],
        ids=["submit-order", "trading-client"],
    )
    def test_the_harness_catches_an_order(self, tmp_path, extra, order):
        """Self-test: an order method (or the broker client factory) records itself even
        when the refusal it raises is swallowed."""
        code, report = _run_runtime_harness(tmp_path, extra)
        assert code == 1
        assert report["orders"] == [order] and report["touched"] == []
