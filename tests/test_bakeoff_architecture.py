"""The bakeoff stays offline: it never reaches the order path, and the live
system never reaches it.

- No ``bakeoff`` module imports the execution package, the policy
  dispatcher, the brain cycle, the CLIs or the notifiers, by statement or at
  runtime (a fresh interpreter imports every bakeoff module and inspects
  ``sys.modules``).
- No module under ``aegis`` imports ``bakeoff``.
- The Gemini SDK is optional: importing the bakeoff never loads it.
- ``AnthropicProvider`` sends exactly the live thesis call's shape.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import anthropic
import httpx2 as httpx
import pytest

from aegis.brain.llm import AnthropicLLM
from aegis.config import REPO_ROOT, get_config
from bakeoff.providers import AnthropicProvider, CallRequest, ProviderError, RateLimited

BAKEOFF_DIR = REPO_ROOT / "bakeoff"
AEGIS_DIR = REPO_ROOT / "aegis"

FORBIDDEN = (
    "aegis.execution",
    "aegis.policy.dispatch",
    "aegis.brain.cycle",
    "aegis.cli",
    "aegis.notify",
    "aegis.approve_web",
)
"""The order path and everything that runs the live cycle or talks to the operator."""


def _modules(directory: Path) -> list[Path]:
    return sorted(p for p in directory.rglob("*.py") if "__pycache__" not in p.parts)


def _imported_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
            names.update(f"{node.module}.{alias.name}" for alias in node.names)
        elif isinstance(node, ast.Call) and getattr(node.func, "id", None) == "__import__":
            names.add("<dynamic __import__>")
        elif isinstance(node, ast.Attribute) and node.attr == "import_module":
            names.add("<importlib.import_module>")
    return names


def _hits(name: str) -> bool:
    return any(name == f or name.startswith(f + ".") for f in FORBIDDEN)


def test_no_bakeoff_module_imports_the_order_path():
    offenders = {
        str(path.relative_to(REPO_ROOT)): sorted(n for n in _imported_names(path) if _hits(n) or n.startswith("<"))
        for path in _modules(BAKEOFF_DIR)
    }
    assert {k: v for k, v in offenders.items() if v} == {}


def test_nothing_under_aegis_imports_the_bakeoff():
    offenders = [
        str(path.relative_to(REPO_ROOT))
        for path in _modules(AEGIS_DIR)
        if any(n == "bakeoff" or n.startswith("bakeoff.") for n in _imported_names(path))
    ]
    assert offenders == []


def test_importing_every_bakeoff_module_loads_no_order_path_and_no_gemini_sdk():
    modules = [
        ".".join(p.relative_to(REPO_ROOT).with_suffix("").parts).removesuffix(".__init__")
        for p in _modules(BAKEOFF_DIR)
    ]
    script = (
        "import importlib, json, sys\n"
        f"for m in {modules!r}: importlib.import_module(m)\n"
        "print(json.dumps(sorted(sys.modules)))\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", script], cwd=REPO_ROOT, capture_output=True, text=True, check=True, timeout=120
    )
    loaded = json.loads(out.stdout)
    assert [m for m in loaded if _hits(m)] == []
    assert [m for m in loaded if m == "google.genai" or m.startswith("google.genai.")] == []


def test_the_policy_import_is_the_pure_engine_only():
    # The stand in judge uses decide(), which writes nothing; never evaluate()
    # (which records) and never the dispatcher (which places orders).
    source = (BAKEOFF_DIR / "standin.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    called = {
        node.func.id for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "decide" in called and "evaluate" not in called
    assert "from aegis.policy.engine import decide\n" in source


# --- the baseline provider ------------------------------------------------------------------


class _Messages:
    def __init__(self, outcome):
        self.outcome = outcome
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return self.outcome


def _message(text='{"candidates": [], "no_idea_reason": "quiet"}', stop="end_turn"):
    return SimpleNamespace(
        stop_reason=stop,
        usage=SimpleNamespace(input_tokens=5000, output_tokens=900, cache_creation_input_tokens=0, cache_read_input_tokens=0),
        content=[SimpleNamespace(type="text", text=text)],
        model="claude-opus-5-5",
        stop_details=None,
    )


def _provider(outcome):
    messages = _Messages(outcome)
    llm = AnthropicLLM(max_retries=0, client=SimpleNamespace(messages=messages))
    stage = get_config().brain.stages.thesis
    provider = AnthropicProvider(
        name="baseline", model=stage.model, effort=stage.effort, max_output_tokens=stage.max_tokens, llm=llm
    )
    return provider, messages, stage


def test_the_baseline_call_has_the_live_thesis_shape():
    provider, messages, stage = _provider(_message())
    out = provider.complete(CallRequest(system="SYS", user="USER", json_schema={"type": "object"}))
    assert messages.kwargs == {
        "model": stage.model,
        "max_tokens": stage.max_tokens,
        "system": "SYS",
        "messages": [{"role": "user", "content": "USER"}],
        "output_config": {"format": {"type": "json_schema", "schema": {"type": "object"}}, "effort": stage.effort},
    }
    assert (out.input_tokens, out.output_tokens, out.thinking_tokens, out.billable_output_tokens) == (5000, 900, None, 900)


def test_a_truncated_answer_is_still_a_billed_response():
    provider, _, _ = _provider(_message(text='{"candidates": [', stop="max_tokens"))
    out = provider.complete(CallRequest(system="s", user="u", json_schema={}))
    assert out.stop_reason == "max_tokens" and out.text == '{"candidates": [' and out.input_tokens == 5000


def test_a_rate_limit_is_retryable_and_other_errors_are_not():
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    limited = anthropic.RateLimitError(
        "slow down", response=httpx.Response(429, request=request, headers={"retry-after": "7"}), body=None
    )
    provider, _, _ = _provider(limited)
    with pytest.raises(RateLimited) as raised:
        provider.complete(CallRequest(system="s", user="u", json_schema={}))
    assert raised.value.retry_after == 7.0
    bad = anthropic.BadRequestError("bad", response=httpx.Response(400, request=request), body=None)
    provider, _, _ = _provider(bad)
    with pytest.raises(ProviderError) as raised:
        provider.complete(CallRequest(system="s", user="u", json_schema={}))
    assert not isinstance(raised.value, RateLimited)


def test_the_baseline_refuses_an_llm_with_its_own_retries():
    llm = AnthropicLLM(max_retries=2, client=SimpleNamespace(messages=_Messages(_message())))
    with pytest.raises(ValueError, match="owns retries"):
        AnthropicProvider(name="b", model="m", effort=None, max_output_tokens=10, llm=llm)


def test_build_providers_reads_the_baseline_from_config_yaml():
    from bakeoff.__main__ import build_providers
    from bakeoff.config import load_bakeoff_config

    class Lister:
        def list(self):
            return []

    llm = AnthropicLLM(max_retries=0, client=SimpleNamespace(messages=_Messages(_message()), models=Lister()))
    gemini = SimpleNamespace(models=SimpleNamespace())
    proposers, critic, _ = build_providers(load_bakeoff_config(), get_config(), anthropic_llm=llm, gemini=gemini)
    stage = get_config().brain.stages.thesis
    base = proposers[0]
    assert (base.model, base.effort, base.max_output_tokens) == (stage.model, stage.effort, stage.max_tokens)
    assert [p.name for p in proposers] == ["baseline", "neon-medium", "neon-high"]
    assert [(p.model, p.thinking_level) for p in proposers[1:]] == [("gemini-4-neon", "medium"), ("gemini-4-neon", "high")]
    assert critic.name == "neon-critic" and critic.thinking_level == "medium"
