"""Shared builders for the bakeoff tests: a real cycle written by the brain
with canned model outputs, and canned providers. No network, no keys."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from aegis.brain.cycle import run_cycle
from aegis.brain.models import MarketSnapshot
from aegis.brain.testing import FakeLLM
from aegis.config import AegisConfig
from aegis.store.db import open_store
from aegis.store.models import ReasoningStage
from bakeoff.providers import CallRequest, ProviderError, ProviderResponse

FIXTURES = Path(__file__).parent / "fixtures" / "brain"

OK = {"scan": ["scan_ok.json"], "thesis": ["thesis_ok.json"], "proposal": ["proposal_ok.json"]}


def snapshot() -> MarketSnapshot:
    return MarketSnapshot.model_validate(
        json.loads((FIXTURES / "market_snapshot.json").read_text(encoding="utf-8"))
    )


def thesis_ok() -> str:
    return (FIXTURES / "thesis_ok.json").read_text(encoding="utf-8")


def thesis_empty() -> str:
    return (FIXTURES / "thesis_empty.json").read_text(encoding="utf-8")


def write_cycles(db_path: Path, config: AegisConfig, outputs: Sequence[dict]) -> list[FakeLLM]:
    """Run one live-shaped cycle per ``outputs`` entry into a fresh store; the
    fakes keep the exact requests the thesis stage sent."""
    conn = open_store(db_path)
    fakes = []
    try:
        for mapping in outputs:
            fake = FakeLLM.from_fixtures(mapping)
            run_cycle(conn, fake, config=config, snapshot=snapshot())
            fakes.append(fake)
    finally:
        conn.close()
    return fakes


def thesis_request(fake: FakeLLM):
    return next(r for r in fake.calls if r.stage is ReasoningStage.THESIS)


class FakeProvider:
    """A canned ``Provider``: hands out ``responses`` in order (text, or an
    exception to raise), stamping fixed usage on every answer."""

    def __init__(
        self,
        name: str,
        model: str,
        responses: Sequence[str | BaseException],
        *,
        input_tokens: int = 1000,
        output_tokens: int = 200,
        thinking_tokens: int | None = 300,
        max_output_tokens: int = 4000,
        latency_ms: float = 1500.0,
    ) -> None:
        self.name = name
        self.model = model
        self.max_output_tokens = max_output_tokens
        self._responses = list(responses)
        self._usage = (input_tokens, output_tokens, thinking_tokens)
        self.latency_ms = latency_ms
        self.requests: list[CallRequest] = []

    def complete(self, request: CallRequest) -> ProviderResponse:
        self.requests.append(request)
        if not self._responses:
            raise ProviderError(f"{self.name}: no canned response left")
        entry = self._responses.pop(0)
        if isinstance(entry, BaseException):
            raise entry
        inp, out, think = self._usage
        return ProviderResponse(
            text=entry,
            model=self.model,
            input_tokens=inp,
            output_tokens=out,
            thinking_tokens=think,
            billable_output_tokens=out + (think or 0),
            stop_reason="STOP",
            latency_ms=self.latency_ms,
        )
