"""``FakeLLM``: an ``LLMClient`` that replays canned text, for tests and a --fake CLI.

It lives in the package (not under ``tests/``) so the test suite and a
future ``--fake`` flag on ``aegis.cli.brain`` share one implementation.
Responses are queued per stage and popped in order; an exception in a
queue is raised in place of a response (to script a rate-limit or a
``BrainError``), and running out is a ``RuntimeError`` — a test that makes
more calls than it scripted should fail loudly, not loop. Every request is
kept on ``calls`` so tests can inspect what the stages asked for. Never
imports ``anthropic``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from aegis.brain.llm import LLMRequest, LLMResponse
from aegis.config import REPO_ROOT
from aegis.store.models import ReasoningStage

FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures" / "brain"


class FakeLLM:
    """Canned ``LLMClient``: one queue of outputs per stage, consumed in order.

    ``responses`` maps a stage (enum or its value) to the texts — or
    exceptions — to hand out; ``tokens_in``/``tokens_out``/``latency_ms``
    and ``model`` (the answering model) are stamped on every response.
    ``model`` is what the reasoning rows record as ``model_name`` (via
    ``StageUsage.response_model``); cost estimates still price the
    *requested* model from the stage config.
    """

    def __init__(
        self,
        responses: Mapping[ReasoningStage | str, Sequence[str | BaseException]] | None = None,
        *,
        tokens_in: int = 100,
        tokens_out: int = 50,
        latency_ms: float = 1.0,
        model: str = "fake-model",
    ) -> None:
        self._queues: dict[str, list[str | BaseException]] = {}
        for stage, entries in (responses or {}).items():
            self._queues.setdefault(ReasoningStage(stage).value, []).extend(entries)
        self._tokens_in = tokens_in
        self._tokens_out = tokens_out
        self._latency_ms = latency_ms
        self._model = model
        self.calls: list[LLMRequest] = []

    @classmethod
    def from_fixtures(
        cls,
        mapping: Mapping[str, Sequence[str]],
        *,
        fixtures_dir: Path = FIXTURES_DIR,
        **kwargs: object,
    ) -> FakeLLM:
        """Queue the TEXT of fixture files (``{"scan": ["scan_ok.json"], ...}``).

        The file's text is the canned output verbatim, so ``malformed.json``
        may hold something that is not JSON at all.
        """
        responses = {
            stage: [(fixtures_dir / name).read_text(encoding="utf-8") for name in names]
            for stage, names in mapping.items()
        }
        return cls(responses, **kwargs)  # type: ignore[arg-type]

    def remaining(self, stage: ReasoningStage | str) -> int:
        """How many canned entries are still queued for ``stage``."""
        return len(self._queues.get(ReasoningStage(stage).value, []))

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.calls.append(request)
        queue = self._queues.get(request.stage.value, [])
        if not queue:
            raise RuntimeError(f"FakeLLM: no canned response left for stage {request.stage.value}")
        entry = queue.pop(0)
        if isinstance(entry, BaseException):
            raise entry
        return LLMResponse(
            text=entry,
            model=self._model,
            tokens_in=self._tokens_in,
            tokens_out=self._tokens_out,
            latency_ms=self._latency_ms,
            stop_reason="end_turn",
            request_id=f"fake-{len(self.calls)}",
            attempts=1,
        )
