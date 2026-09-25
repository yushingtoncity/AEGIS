"""The cycle orchestrator and the brain CLI, driven by ``FakeLLM`` and the
canned outputs under ``tests/fixtures/brain`` against tmp_path stores — no
network, no API key, no live data.

``run_cycle`` is checked through what it leaves in the store (reasoning
rows, the proposal and its legs, the events) and what it returns; the CLI
through ``main([...])`` with an injected ``FakeLLM`` and a canned snapshot
builder, output via capsys.
"""

import io
import json
import re
import sys
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace

import anthropic
import httpx2
import pytest

from aegis.brain.cycle import (
    MARKET_SNAPSHOT_EVENT,
    describe_candidate,
    first_offer,
    market_snapshot_event,
    proposal_records,
    reasoning_row,
    run_cycle,
    snapshot_from_event,
)
from aegis.brain.errors import BrainError, BudgetExceeded
from aegis.brain.llm import AnthropicLLM, LLMResponseError, estimate_cost
from aegis.brain.models import (
    HeadlineItem,
    MarketSnapshot,
    ScanBrief,
    StageUsage,
    ThesisOutput,
    TradeProposal,
)
from aegis.brain.prompts import build_thesis_prompt
from aegis.brain.proposal import run_proposal
from aegis.brain.scan import merge_staleness_warnings
from aegis.brain.testing import FIXTURES_DIR, FakeLLM
from aegis.cli import brain as cli_brain
from aegis.cli import trace as cli_trace
from aegis.config import AegisConfig, BrainConfig, get_config
from aegis.data.models import OptionType
from aegis.store.db import open_store
from aegis.store.errors import StoreError
from aegis.store.models import (
    Event,
    EventLevel,
    Instrument,
    OrderSide,
    OrderType,
    Reasoning,
    ReasoningStage,
    TokenUsage,
)
from aegis.store.repo import (
    add_reasoning,
    get_cycle_reasoning,
    get_cycle_token_usage,
    get_proposal,
    get_proposal_legs,
    get_proposal_trace,
    get_recent_events,
    get_recent_proposals,
    get_token_usage,
    log_event,
)

OK = {"scan": ["scan_ok.json"], "thesis": ["thesis_ok.json"], "proposal": ["proposal_ok.json"]}
STAGES = [ReasoningStage.SCAN, ReasoningStage.THESIS, ReasoningStage.PROPOSAL]

# A proposal-stage trade on the QQQ equity candidate (thesis_ok's second idea),
# for the tests that make the SPY spread unresolvable.
EQUITY_TRADE = json.dumps(
    {
        "outcome": "trade",
        "proposal": {
            "symbol": "QQQ",
            "instrument": "equity",
            "side": "buy",
            "quantity": 5,
            "order_type": "limit",
            "limit_price": 561.30,
            "thesis": "Relative strength into the 20-day high on the capex catalyst.",
            "confidence": 0.5,
            "invalidation": "A daily close below 555 before 2026-08-07.",
            "legs": [],
        },
        "no_trade_reason": None,
    }
)


@pytest.fixture
def snapshot(fixture) -> MarketSnapshot:
    return MarketSnapshot.model_validate(fixture("brain/market_snapshot.json"))


@pytest.fixture
def closed_snapshot(fixture) -> MarketSnapshot:
    return MarketSnapshot.model_validate(fixture("brain/market_snapshot_closed.json"))


@pytest.fixture
def config() -> AegisConfig:
    """The repo's config.yaml: real stage models, budgets and price placeholders."""
    return get_config()


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "brain.db"


@pytest.fixture
def conn(db_path):
    conn = open_store(db_path)
    yield conn
    conn.close()


def _text(name: str) -> str:
    return (FIXTURES_DIR / name).read_text(encoding="utf-8")


def _fake(**kwargs) -> FakeLLM:
    return FakeLLM.from_fixtures(OK, **kwargs)


def _events(conn) -> list[Event]:
    """Every event, oldest first."""
    return list(reversed(get_recent_events(conn, limit=100)))


def _kinds(conn) -> list[str]:
    return [event.kind for event in _events(conn)]


def _event(conn, kind: str) -> Event:
    (event,) = [e for e in _events(conn) if e.kind == kind]
    return event


def _seed(conn, cycle_id: str, tokens_in: int, tokens_out: int = 0, **overrides) -> Reasoning:
    fields = dict(cycle_id=cycle_id, stage=ReasoningStage.SCAN, content="{}", tokens_in=tokens_in,
                  tokens_out=tokens_out, model_name="claude-haiku-4-5-20251001")
    fields.update(overrides)
    return add_reasoning(conn, Reasoning(**fields))


def _with_contract(snapshot: MarketSnapshot, occ: str, **update) -> MarketSnapshot:
    """The snapshot with one SPY contract's fields replaced."""
    spy, qqq = snapshot.symbols
    assert spy.chain is not None
    contracts = tuple(c.model_copy(update=update) if c.symbol == occ else c for c in spy.chain.contracts)
    chain = spy.chain.model_copy(update={"contracts": contracts})
    return snapshot.model_copy(update={"symbols": (spy.model_copy(update={"chain": chain}), qqq)})


def _builder(snapshot: MarketSnapshot, seen: list | None = None):
    """A stand-in for ``build_market_snapshot`` that records how it was called."""

    def build(symbols=None, *, config=None, now=None):
        if seen is not None:
            seen.append((symbols, config, now))
        return snapshot

    return build


class _TruncatingClient:
    """An SDK client (``.messages.create``) whose every answer stopped at
    ``max_tokens``: billed, but unusable."""

    def __init__(self, *, tokens_in: int, tokens_out: int):
        self.calls: list[dict] = []
        self.messages = self
        self._usage = SimpleNamespace(
            input_tokens=tokens_in, output_tokens=tokens_out,
            cache_creation_input_tokens=None, cache_read_input_tokens=None,
        )

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text='{"symbols": [')],
            stop_reason="max_tokens", stop_details=None, usage=self._usage,
            model="claude-haiku-answered", _request_id="req_truncated",
        )


class _SnapshotIds:
    """An ``LLMClient`` that answers the way the API may for an alias: each
    response names a dated snapshot of the model requested
    (``claude-opus-5-5`` → ``claude-opus-5-5-20260301``); an id that already
    carries a date comes back as it is."""

    def __init__(self, inner: FakeLLM):
        self.inner = inner

    def complete(self, request):
        response = self.inner.complete(request)
        dated = request.model if re.search(r"-\d{8}$", request.model) else f"{request.model}-20260301"
        return response.model_copy(update={"model": dated})


# --- run_cycle ----------------------------------------------------------------


class TestRunCycle:
    def test_full_cycle_writes_three_rows_one_proposal_with_legs_and_the_events(
        self, conn, snapshot, config
    ):
        llm = _fake(tokens_in=100, tokens_out=50, latency_ms=1.5)
        result = run_cycle(conn, llm, config=config, snapshot=snapshot)

        assert result.outcome == "proposal"
        assert result.proposal_id is not None and result.no_trade_reason is None
        assert result.halt_reason is None
        assert result.started_at.tzinfo is not None and result.finished_at >= result.started_at
        assert [call.stage for call in llm.calls] == STAGES
        assert {call.cycle_id for call in llm.calls} == {result.cycle_id}

        rows = get_cycle_reasoning(conn, result.cycle_id)
        assert [row.stage for row in rows] == STAGES
        assert {row.cycle_id for row in rows} == {result.cycle_id}
        assert {row.proposal_id for row in rows} == {result.proposal_id}
        assert [row.content for row in rows] == [_text(name) for name in ("scan_ok.json", "thesis_ok.json", "proposal_ok.json")]
        assert {(row.tokens_in, row.tokens_out, row.latency_ms) for row in rows} == {(100, 50, 1.5)}
        assert [row.model_name for row in rows] == ["fake-model"] * 3  # the model that ANSWERED

        proposal = get_proposal(conn, result.proposal_id)
        assert proposal is not None
        assert proposal.cycle_id == result.cycle_id
        assert (proposal.symbol, proposal.instrument, proposal.side) == ("SPY", Instrument.OPTION, OrderSide.BUY)
        assert (proposal.quantity, proposal.order_type, proposal.limit_price) == (2.0, OrderType.LIMIT, 4.55)
        assert proposal.confidence == 0.58
        assert proposal.raw_model_output == _text("proposal_ok.json")
        assert proposal.model_name == config.brain.stages.proposal.model
        assert proposal.prompt_version == "proposal-v1+system-v1"
        assert [(l.leg_index, l.side, l.option_type, l.strike, l.quantity, l.expiration, l.symbol) for l in get_proposal_legs(conn, proposal.id)] == [
            (0, OrderSide.BUY, OptionType.CALL, 640.0, 2.0, date(2026, 8, 21), "SPY260821C00640000"),
            (1, OrderSide.SELL, OptionType.CALL, 650.0, 2.0, date(2026, 8, 21), "SPY260821C00650000"),
        ]
        assert result.proposal == TradeProposal.model_validate(json.loads(_text("proposal_ok.json"))["proposal"])

        assert _kinds(conn) == ["cycle_start", "market_snapshot", "proposal", "cycle_end"]
        assert _event(conn, "cycle_start").payload == {"cycle_id": result.cycle_id, "symbols": ["SPY", "QQQ"]}
        assert _event(conn, "proposal").payload == {
            "cycle_id": result.cycle_id, "proposal_id": result.proposal_id, "symbol": "SPY",
        }
        assert _event(conn, "cycle_end").payload == {
            "cycle_id": result.cycle_id, "outcome": "proposal", "tokens_in": 300, "tokens_out": 150,
            "estimated_cost_usd": result.estimated_cost_usd, "proposal_id": result.proposal_id,
        }
        assert all(event.level is EventLevel.INFO for event in _events(conn))

        trace = get_proposal_trace(conn, result.proposal_id)
        assert [r.stage for r in trace.reasoning] == STAGES
        assert all(r.proposal_id == result.proposal_id for r in trace.reasoning)
        assert [leg.leg_index for leg in trace.legs] == [0, 1]

    def test_market_snapshot_event_is_the_audit_copy_of_the_input(self, conn, snapshot, config):
        result = run_cycle(conn, _fake(), config=config, snapshot=snapshot)
        event = _event(conn, MARKET_SNAPSHOT_EVENT)
        assert event.payload is not None and event.payload["cycle_id"] == result.cycle_id
        assert [s["symbol"] for s in event.payload["symbols"]] == ["SPY", "QQQ"]
        assert snapshot_from_event(event) == snapshot
        assert result.cycle_id in event.message and "market OPEN" in event.message

    def test_result_carries_brief_candidates_usage_and_cost(self, conn, snapshot, config):
        result = run_cycle(conn, _fake(), config=config, snapshot=snapshot)
        assert result.brief is not None
        assert [s.symbol for s in result.brief.symbols] == ["SPY", "QQQ"]
        assert [describe_candidate(c) for c in result.candidates] == [
            "SPY call_debit_spread 640/650 exp 2026-08-21",
            "QQQ equity",
        ]
        assert [u.stage for u in result.usage] == STAGES
        # requested (what the estimate prices) and answered (what the rows record)
        assert [u.model for u in result.usage] == [config.brain.stage(s.value).model for s in STAGES]
        assert [u.response_model for u in result.usage] == ["fake-model"] * 3
        assert (result.total_tokens_in, result.total_tokens_out) == (300, 150)
        assert result.estimated_cost_usd == pytest.approx(estimate_cost(result.usage, config.brain.prices_per_mtok))
        assert result.estimated_cost_usd is not None and result.estimated_cost_usd > 0

    def test_rows_record_the_answering_model_and_the_cost_prices_the_configured_one(self, conn, snapshot, config):
        llm = _fake(model="claude-answered-20260901")
        result = run_cycle(conn, llm, config=config, snapshot=snapshot)
        rows = get_cycle_reasoning(conn, result.cycle_id)
        assert {row.model_name for row in rows} == {"claude-answered-20260901"}
        assert "claude-answered-20260901" not in config.brain.prices_per_mtok
        # priced by the configured models, so the estimate exists though the answering model has no price
        assert result.estimated_cost_usd == pytest.approx(estimate_cost(result.usage, config.brain.prices_per_mtok))
        assert result.estimated_cost_usd is not None and result.estimated_cost_usd > 0
        proposal = get_proposal(conn, result.proposal_id)
        assert proposal is not None and proposal.model_name == config.brain.stages.proposal.model

    def test_cost_is_none_when_a_model_has_no_price(self, conn, snapshot, config):
        unpriced = config.model_copy(update={"brain": config.brain.model_copy(update={"prices_per_mtok": {}})})
        result = run_cycle(conn, _fake(), config=unpriced, snapshot=snapshot)
        assert result.outcome == "proposal" and result.estimated_cost_usd is None
        assert _event(conn, "cycle_end").payload["estimated_cost_usd"] is None

    def test_no_candidates_is_a_no_trade_with_two_rows_kept(self, conn, snapshot, config):
        llm = FakeLLM.from_fixtures({"scan": ["scan_ok.json"], "thesis": ["thesis_empty.json"]})
        result = run_cycle(conn, llm, config=config, snapshot=snapshot)
        assert result.outcome == "no_trade"
        assert result.proposal is None and result.proposal_id is None
        assert result.candidates == ()
        assert result.no_trade_reason is not None and "No edge" in result.no_trade_reason
        assert len(llm.calls) == 2
        assert get_recent_proposals(conn) == []
        rows = get_cycle_reasoning(conn, result.cycle_id)
        assert [row.stage for row in rows] == STAGES[:2]
        assert {row.cycle_id for row in rows} == {result.cycle_id}
        assert {row.proposal_id for row in rows} == {None}
        assert _kinds(conn) == ["cycle_start", "market_snapshot", "no_trade", "cycle_end"]
        no_trade = _event(conn, "no_trade")
        assert no_trade.payload == {"cycle_id": result.cycle_id, "reason": result.no_trade_reason}
        assert "thesis stage" in no_trade.message
        end = _event(conn, "cycle_end").payload
        assert (end["outcome"], end["tokens_in"], end["proposal_id"]) == ("no_trade", 200, None)

    def test_proposal_stage_no_trade_keeps_three_rows(self, conn, snapshot, config):
        llm = FakeLLM.from_fixtures({**OK, "proposal": ["proposal_no_trade.json"]})
        result = run_cycle(conn, llm, config=config, snapshot=snapshot)
        assert result.outcome == "no_trade"
        assert result.proposal is None and result.proposal_id is None
        assert len(result.candidates) == 2
        assert result.no_trade_reason is not None and "pricing does not justify it" in result.no_trade_reason
        assert [row.stage for row in get_cycle_reasoning(conn, result.cycle_id)] == STAGES
        assert get_recent_proposals(conn) == []
        assert _kinds(conn) == ["cycle_start", "market_snapshot", "no_trade", "cycle_end"]
        no_trade = _event(conn, "no_trade")
        assert no_trade.payload == {"cycle_id": result.cycle_id, "reason": result.no_trade_reason}
        assert "proposal stage" in no_trade.message
        assert _event(conn, "cycle_end").payload["tokens_in"] == 300

    def test_budget_halt_before_any_call(self, conn, snapshot, config):
        tiny = config.model_copy(update={"brain": BrainConfig(per_cycle_token_cap=10, daily_token_budget=10)})
        llm = _fake()
        result = run_cycle(conn, llm, config=tiny, snapshot=snapshot)  # returned, not raised
        assert result.outcome == "halted"
        assert result.halt_reason is not None
        assert result.halt_reason.startswith("brain failed: budget halt: cycle limit would be exceeded")
        assert result.brief is None and result.candidates == () and result.usage == ()
        assert result.proposal is None and result.no_trade_reason is None
        assert result.estimated_cost_usd == 0.0
        assert llm.calls == [] and llm.remaining("scan") == 1
        assert get_cycle_reasoning(conn, result.cycle_id) == []
        assert _kinds(conn) == ["cycle_start", "market_snapshot", "budget_halt", "cycle_end"]
        halt = _event(conn, "budget_halt")
        assert halt.level is EventLevel.WARNING
        assert halt.payload is not None
        assert (halt.payload["cycle_id"], halt.payload["stage"], halt.payload["limit"]) == (result.cycle_id, "scan", "cycle")
        end = _event(conn, "cycle_end").payload
        assert (end["outcome"], end["tokens_in"], end["tokens_out"], end["proposal_id"]) == ("halted", 0, 0, None)

    def test_daily_budget_from_the_store_halts(self, conn, snapshot, config):
        _seed(conn, "cycle-earlier", config.brain.daily_token_budget)  # today's rows already fill the budget
        llm = _fake()
        result = run_cycle(conn, llm, config=config, snapshot=snapshot)
        assert result.outcome == "halted"
        assert "daily limit would be exceeded" in (result.halt_reason or "")
        assert llm.calls == []
        assert _event(conn, "budget_halt").payload["limit"] == "daily"

    def test_halt_mid_cycle_keeps_the_rows_written_so_far(self, conn, snapshot, config):
        # The scan's own tokens nearly fill the per-cycle cap, so the thesis call is refused.
        llm = _fake(tokens_in=config.brain.per_cycle_token_cap - 2000, tokens_out=50)
        result = run_cycle(conn, llm, config=config, snapshot=snapshot)
        assert result.outcome == "halted"
        assert [call.stage for call in llm.calls] == [ReasoningStage.SCAN]
        assert result.brief is not None and result.candidates == ()
        assert [u.stage for u in result.usage] == [ReasoningStage.SCAN]
        assert [row.stage for row in get_cycle_reasoning(conn, result.cycle_id)] == [ReasoningStage.SCAN]
        assert _kinds(conn) == ["cycle_start", "market_snapshot", "budget_halt", "cycle_end"]
        assert _event(conn, "budget_halt").payload["stage"] == "thesis"
        end = _event(conn, "cycle_end").payload
        assert (end["outcome"], end["tokens_in"], end["tokens_out"]) == ("halted", config.brain.per_cycle_token_cap - 2000, 50)

    def test_malformed_proposal_output_is_a_brain_error_with_events(self, conn, snapshot, config):
        llm = FakeLLM.from_fixtures({**OK, "proposal": ["malformed.json"] * (config.brain.max_retries + 1)})
        with pytest.raises(BrainError) as info:
            run_cycle(conn, llm, config=config, snapshot=snapshot)
        assert not isinstance(info.value, BudgetExceeded)
        assert info.value.what == "proposal output invalid after 4 attempts"
        assert len(llm.calls) == 2 + config.brain.max_retries + 1
        cycle_id = llm.calls[0].cycle_id
        assert info.value.key == cycle_id

        assert _kinds(conn) == ["cycle_start", "market_snapshot", "brain_error", "cycle_end"]
        error = _event(conn, "brain_error")
        assert error.level is EventLevel.ERROR
        assert error.payload == {"cycle_id": cycle_id, "stage": "proposal", "error": str(info.value)}
        end = _event(conn, "cycle_end")
        assert end.payload is not None
        # every attempt of the failed stage was billed, so cycle_end counts all six calls
        assert (end.payload["outcome"], end.payload["tokens_in"], end.payload["tokens_out"]) == ("failed", 600, 300)
        assert end.payload["proposal_id"] is None
        rows = get_cycle_reasoning(conn, cycle_id)
        assert [row.stage for row in rows] == STAGES
        failed = rows[-1]  # the failed stage's row: its summed tokens and the last (invalid) text, unlinked
        assert (failed.tokens_in, failed.tokens_out, failed.proposal_id) == (400, 200, None)
        assert failed.content == _text("malformed.json")
        assert failed.model_name == "fake-model"  # the last response's model
        assert get_cycle_token_usage(conn, cycle_id) == TokenUsage(tokens_in=600, tokens_out=300, calls=3)
        assert get_recent_proposals(conn) == []

    def test_malformed_scan_output_names_the_scan_stage(self, conn, snapshot, config):
        llm = FakeLLM.from_fixtures({"scan": ["malformed.json"] * (config.brain.max_retries + 1)})
        with pytest.raises(BrainError, match="scan output invalid after 4 attempts"):
            run_cycle(conn, llm, config=config, snapshot=snapshot)
        assert _event(conn, "brain_error").payload["stage"] == "scan"
        (row,) = get_cycle_reasoning(conn, llm.calls[0].cycle_id)
        assert (row.stage, row.tokens_in, row.tokens_out, row.content) == (ReasoningStage.SCAN, 400, 200, _text("malformed.json"))

    def test_failed_stage_spend_reaches_the_daily_budget(self, conn, snapshot, config):
        """Every attempt of a failing stage was billed: the store, and so the guard, must see them."""
        llm = FakeLLM.from_fixtures(
            {**OK, "proposal": ["malformed.json"] * (config.brain.max_retries + 1)}, tokens_in=100, tokens_out=50
        )
        with pytest.raises(BrainError):
            run_cycle(conn, llm, config=config, snapshot=snapshot)
        assert len(llm.calls) == 6
        # every billed token is in the table the guard sums: FakeLLM handed out 6 x (100 in / 50 out)
        assert get_cycle_token_usage(conn, llm.calls[0].cycle_id) == TokenUsage(tokens_in=600, tokens_out=300, calls=3)
        assert get_token_usage(conn) == TokenUsage(tokens_in=600, tokens_out=300, calls=3)
        # The next cycle's guard starts from the true spend: a daily budget one token short of
        # 900 + the scan's projection halts before any call.
        budget = 900 + llm.calls[0].estimated_input_tokens + config.brain.stages.scan.max_tokens - 1
        tight = config.model_copy(
            update={"brain": config.brain.model_copy(update={"per_cycle_token_cap": budget, "daily_token_budget": budget})}
        )
        again = _fake()
        result = run_cycle(conn, again, config=tight, snapshot=snapshot)
        assert result.outcome == "halted" and again.calls == []
        halt = _event(conn, "budget_halt").payload
        assert (halt["limit"], halt["today_tokens"]) == ("daily", 900)

    def test_halt_mid_stage_journals_the_responses_received(self, conn, snapshot, config):
        # The scan's first (invalid) attempt nearly fills the per-cycle cap, so its retry is refused.
        cap = config.brain.per_cycle_token_cap
        llm = FakeLLM.from_fixtures({**OK, "scan": ["malformed.json", "scan_ok.json"]}, tokens_in=cap - 2000, tokens_out=50)
        result = run_cycle(conn, llm, config=config, snapshot=snapshot)
        assert result.outcome == "halted"
        assert [call.stage for call in llm.calls] == [ReasoningStage.SCAN]  # the retry never went out
        assert result.brief is None and result.candidates == ()
        (usage,) = result.usage
        assert (usage.stage, usage.tokens_in, usage.tokens_out, usage.attempts) == (ReasoningStage.SCAN, cap - 2000, 50, 1)
        (row,) = get_cycle_reasoning(conn, result.cycle_id)
        assert (row.stage, row.tokens_in, row.tokens_out, row.model_name) == (ReasoningStage.SCAN, cap - 2000, 50, "fake-model")
        # the halt propagated unchanged, so no stage output is known: the row says so, never the prompt
        assert row.content == f"(no usable output: BudgetExceeded: {result.halt_reason})"
        assert get_cycle_token_usage(conn, result.cycle_id).total == cap - 2000 + 50
        assert _kinds(conn) == ["cycle_start", "market_snapshot", "budget_halt", "cycle_end"]
        halt = _event(conn, "budget_halt").payload
        assert (halt["stage"], halt["limit"], halt["cycle_tokens"]) == ("scan", "cycle", cap - 2000 + 50)
        end = _event(conn, "cycle_end").payload
        assert (end["outcome"], end["tokens_in"], end["tokens_out"]) == ("halted", cap - 2000, 50)

    def test_unexpected_error_still_logs_brain_error_and_cycle_end(self, conn, snapshot, config):
        llm = FakeLLM({"scan": [RuntimeError("transport exploded")]})
        with pytest.raises(RuntimeError, match="transport exploded"):
            run_cycle(conn, llm, config=config, snapshot=snapshot)
        assert _kinds(conn) == ["cycle_start", "market_snapshot", "brain_error", "cycle_end"]
        error = _event(conn, "brain_error")
        assert error.level is EventLevel.ERROR
        assert error.payload == {"cycle_id": llm.calls[0].cycle_id, "stage": "scan", "error": "RuntimeError: transport exploded"}
        end = _event(conn, "cycle_end").payload
        assert (end["outcome"], end["tokens_in"], end["tokens_out"], end["proposal_id"]) == ("failed", 0, 0, None)
        assert get_cycle_reasoning(conn, llm.calls[0].cycle_id) == []  # nothing was billed

    def test_unexpected_error_after_a_billed_attempt_journals_it(self, conn, snapshot, config):
        llm = FakeLLM({"scan": [_text("malformed.json"), RuntimeError("transport exploded")]}, tokens_in=70, tokens_out=30)
        with pytest.raises(RuntimeError, match="transport exploded"):
            run_cycle(conn, llm, config=config, snapshot=snapshot)
        (row,) = get_cycle_reasoning(conn, llm.calls[0].cycle_id)
        assert (row.stage, row.tokens_in, row.tokens_out, row.model_name) == (ReasoningStage.SCAN, 70, 30, "fake-model")
        assert row.content == "(no usable output: RuntimeError: transport exploded)"
        end = _event(conn, "cycle_end").payload
        assert (end["outcome"], end["tokens_in"], end["tokens_out"]) == ("failed", 70, 30)

    @pytest.mark.parametrize(
        "interrupt, described",
        [(KeyboardInterrupt(), "KeyboardInterrupt"), (SystemExit(3), "SystemExit: 3")],
        ids=["ctrl-c", "system-exit"],
    )
    def test_an_interrupt_mid_stage_journals_the_billed_attempts(self, conn, snapshot, config, interrupt, described):
        """A Ctrl-C while the thesis retries: the scan and the thesis's first (invalid) attempt were
        billed, so both reach the table the guard sums, and cycle_end is logged before it goes on."""
        llm = FakeLLM({"scan": [_text("scan_ok.json")], "thesis": [_text("malformed.json"), interrupt]})
        with pytest.raises(type(interrupt)):
            run_cycle(conn, llm, config=config, snapshot=snapshot)
        cycle_id = llm.calls[0].cycle_id
        assert [call.stage for call in llm.calls] == [ReasoningStage.SCAN, ReasoningStage.THESIS, ReasoningStage.THESIS]
        scan, thesis = get_cycle_reasoning(conn, cycle_id)
        assert (scan.stage, scan.tokens_in, scan.tokens_out) == (ReasoningStage.SCAN, 100, 50)
        assert (thesis.stage, thesis.tokens_in, thesis.tokens_out, thesis.model_name) == (ReasoningStage.THESIS, 100, 50, "fake-model")
        assert thesis.content == f"(no usable output: {described})"
        # two responses were billed (the interrupted call returned none): all of it is journaled
        assert get_cycle_token_usage(conn, cycle_id) == TokenUsage(tokens_in=200, tokens_out=100, calls=2)
        assert _kinds(conn) == ["cycle_start", "market_snapshot", "brain_error", "cycle_end"]
        assert _event(conn, "brain_error").payload == {"cycle_id": cycle_id, "stage": "thesis", "error": described}
        end = _event(conn, "cycle_end").payload
        assert (end["outcome"], end["tokens_in"], end["tokens_out"], end["proposal_id"]) == ("failed", 200, 100, None)

    def test_truncated_response_is_journaled(self, conn, snapshot, config):
        """An LLMResponseError (stop_reason max_tokens) was billed: its tokens reach the table and cycle_end."""
        out = config.brain.stages.scan.max_tokens  # a truncated answer bills the whole cap
        client = _TruncatingClient(tokens_in=300, tokens_out=out)
        llm = AnthropicLLM(client=client, max_retries=config.brain.max_retries, sleep=lambda seconds: None)
        with pytest.raises(LLMResponseError, match="raise brain.stages.scan.max_tokens in config.yaml") as info:
            run_cycle(conn, llm, config=config, snapshot=snapshot)
        assert len(client.calls) == 1  # never retried: a truncated answer is a config problem
        (cycle_id,) = {e.payload["cycle_id"] for e in _events(conn) if e.payload}
        (row,) = get_cycle_reasoning(conn, cycle_id)
        assert (row.stage, row.tokens_in, row.tokens_out) == (ReasoningStage.SCAN, 300, out)
        assert row.model_name == "claude-haiku-answered"
        assert row.content == '{"symbols": ['  # the billed partial output, kept for the audit
        assert get_cycle_token_usage(conn, cycle_id) == TokenUsage(tokens_in=300, tokens_out=out, calls=1)
        assert _kinds(conn) == ["cycle_start", "market_snapshot", "brain_error", "cycle_end"]
        assert _event(conn, "brain_error").payload["error"] == str(info.value)
        end = _event(conn, "cycle_end").payload
        assert (end["outcome"], end["tokens_in"], end["tokens_out"]) == ("failed", 300, out)
        assert end["estimated_cost_usd"] == pytest.approx((300 * 1.0 + out * 5.0) / 1_000_000)  # haiku placeholders

    def test_blank_truncated_response_falls_back_to_the_note(self, conn, snapshot, config, monkeypatch):
        """A billed answer with no text (thinking alone hit max_tokens) still gets an accounting row."""
        out = config.brain.stages.scan.max_tokens
        client = _TruncatingClient(tokens_in=300, tokens_out=out)
        blank = SimpleNamespace(type="text", text="  ")
        original = client.create
        monkeypatch.setattr(client, "create", lambda **kw: SimpleNamespace(**{**vars(original(**kw)), "content": [blank]}))
        llm = AnthropicLLM(client=client, max_retries=config.brain.max_retries, sleep=lambda seconds: None)
        with pytest.raises(LLMResponseError) as info:
            run_cycle(conn, llm, config=config, snapshot=snapshot)
        (cycle_id,) = {e.payload["cycle_id"] for e in _events(conn) if e.payload}
        (row,) = get_cycle_reasoning(conn, cycle_id)
        assert row.content == f"(no usable output: LLMResponseError: {info.value})"
        assert (row.tokens_in, row.tokens_out) == (300, out)

    def test_store_error_mid_cycle_still_logs_the_events(self, conn, snapshot, config, monkeypatch):
        written = []

        def flaky(conn_, reasoning):
            written.append(reasoning.stage)
            if reasoning.stage is ReasoningStage.THESIS:
                raise StoreError("add reasoning", reasoning.id, Exception("database is locked"))
            return add_reasoning(conn_, reasoning)

        monkeypatch.setattr("aegis.brain.cycle.add_reasoning", flaky)
        llm = _fake()
        with pytest.raises(StoreError, match="add reasoning"):
            run_cycle(conn, llm, config=config, snapshot=snapshot)
        # the thesis row, then its accounting row: that write fails too, and is swallowed
        assert written == [ReasoningStage.SCAN, ReasoningStage.THESIS, ReasoningStage.THESIS]
        assert _kinds(conn) == ["cycle_start", "market_snapshot", "brain_error", "cycle_end"]
        error = _event(conn, "brain_error").payload
        assert error["stage"] == "thesis"
        assert error["error"].startswith("StoreError: store operation failed: add reasoning")
        end = _event(conn, "cycle_end").payload
        assert (end["outcome"], end["tokens_in"]) == ("failed", 200)  # both stages were billed; one row landed
        assert [row.stage for row in get_cycle_reasoning(conn, llm.calls[0].cycle_id)] == [ReasoningStage.SCAN]

    def test_store_error_once_then_the_accounting_row_lands(self, conn, snapshot, config, monkeypatch):
        """The thesis row fails to write once: the accounting row still journals its tokens."""
        failures = []

        def flaky_once(conn_, reasoning):
            if reasoning.stage is ReasoningStage.THESIS and not failures:
                failures.append(reasoning.id)
                raise StoreError("add reasoning", reasoning.id, Exception("database is locked"))
            return add_reasoning(conn_, reasoning)

        monkeypatch.setattr("aegis.brain.cycle.add_reasoning", flaky_once)
        llm = _fake()
        with pytest.raises(StoreError, match="add reasoning") as info:
            run_cycle(conn, llm, config=config, snapshot=snapshot)
        cycle_id = llm.calls[0].cycle_id
        scan, thesis = get_cycle_reasoning(conn, cycle_id)
        assert (thesis.stage, thesis.tokens_in, thesis.tokens_out, thesis.model_name) == (ReasoningStage.THESIS, 100, 50, "fake-model")
        assert thesis.content == f"(no usable output: StoreError: {info.value})"
        assert get_cycle_token_usage(conn, cycle_id) == TokenUsage(tokens_in=200, tokens_out=100, calls=2)
        end = _event(conn, "cycle_end").payload
        assert (end["outcome"], end["tokens_in"], end["tokens_out"]) == ("failed", 200, 100)  # not double-counted

    @pytest.mark.parametrize("error", [RuntimeError("transport exploded"), KeyboardInterrupt()], ids=["error", "ctrl-c"])
    def test_a_store_that_cannot_log_does_not_mask_the_original_error(self, conn, snapshot, config, monkeypatch, error):
        kinds = []

        def dying(conn_, event):
            kinds.append(event.kind)
            if len(kinds) > 2:  # the store dies after cycle_start and market_snapshot
                raise StoreError("log event", event.kind, Exception("disk full"))
            return log_event(conn_, event)

        monkeypatch.setattr("aegis.brain.cycle.log_event", dying)
        with pytest.raises(type(error)) as info:
            run_cycle(conn, FakeLLM({"scan": [error]}), config=config, snapshot=snapshot)
        assert info.value is error
        assert kinds == ["cycle_start", "market_snapshot", "brain_error", "cycle_end"]  # both were attempted

    def test_the_proposal_its_legs_and_the_link_land_together_or_not_at_all(self, conn, snapshot, config):
        """The link of the cycle's reasoning rows runs in the proposal's own transaction: when
        it fails, the proposal and its legs roll back with it — never a stored proposal whose
        reasoning is unlinked while cycle_end says the cycle produced nothing."""
        conn.execute(
            "CREATE TRIGGER refuse_link BEFORE UPDATE OF proposal_id ON reasoning"
            " BEGIN SELECT RAISE(ABORT, 'link refused'); END"
        )
        llm = _fake()
        with pytest.raises(StoreError, match="insert proposal .*link refused"):
            run_cycle(conn, llm, config=config, snapshot=snapshot)
        cycle_id = llm.calls[0].cycle_id
        assert get_recent_proposals(conn) == []
        assert conn.execute("SELECT COUNT(*) FROM proposal_legs").fetchone()[0] == 0
        rows = get_cycle_reasoning(conn, cycle_id)
        assert [row.stage for row in rows] == STAGES and {row.proposal_id for row in rows} == {None}
        assert _kinds(conn) == ["cycle_start", "market_snapshot", "brain_error", "cycle_end"]
        end = _event(conn, "cycle_end").payload
        assert (end["outcome"], end["proposal_id"]) == ("failed", None)

    def test_a_failure_after_the_proposal_landed_still_names_it(self, conn, snapshot, config, monkeypatch):
        def no_proposal_event(conn_, event):
            if event.kind == "proposal":
                raise StoreError("log event", event.kind, Exception("disk full"))
            return log_event(conn_, event)

        monkeypatch.setattr("aegis.brain.cycle.log_event", no_proposal_event)
        with pytest.raises(StoreError, match="log event"):
            run_cycle(conn, _fake(), config=config, snapshot=snapshot)
        (proposal,) = get_recent_proposals(conn)
        assert {row.proposal_id for row in get_cycle_reasoning(conn, proposal.cycle_id)} == {proposal.id}
        end = _event(conn, "cycle_end").payload
        assert (end["outcome"], end["proposal_id"]) == ("failed", proposal.id)

    def test_output_the_store_could_not_hold_is_fed_back_not_a_crash(self, conn, snapshot, config):
        """A NaN limit price used to pass validation and crash the store write with the retry
        unused; it is now fed back and the cycle proposes from the corrected output."""
        nan = _text("proposal_ok.json").replace('"limit_price": 4.55', '"limit_price": NaN')
        llm = FakeLLM({**{s: [_text(n) for n in names] for s, names in OK.items()},
                       "proposal": [nan, _text("proposal_ok.json")]})
        result = run_cycle(conn, llm, config=config, snapshot=snapshot)
        assert result.outcome == "proposal"
        assert [call.stage for call in llm.calls] == [*STAGES, ReasoningStage.PROPOSAL]
        assert "NaN is not a JSON number" in llm.calls[-1].messages[-1].content
        stored = get_proposal(conn, result.proposal_id)
        assert stored is not None and stored.limit_price == 4.55

    def test_a_lone_surrogate_in_a_headline_cannot_stop_the_brain(self, conn, snapshot, config):
        """The real SDK client on a mock transport (a dummy key; nothing leaves the process): a
        truncated emoji escape in a feed's headline and summary used to fail every scan call
        while the SDK encoded the request. It is now rendered as U+FFFD and the cycle runs."""
        truncated = json.loads('"rallies \\ud83d on capex"')
        spy, qqq = snapshot.symbols
        item = HeadlineItem(headline=f"Chipmaker {truncated}", summary=f"Summary {truncated}", source="feed")
        poisoned = snapshot.model_copy(
            update={"symbols": (spy.model_copy(update={"headlines": (*spy.headlines, item)}), qqq)}
        )
        answers = [_text(name) for name in ("scan_ok.json", "thesis_ok.json", "proposal_ok.json")]
        sent = []

        def handler(request):
            body = json.loads(request.content)
            sent.append(body)
            return httpx2.Response(200, json={
                "id": f"msg_{len(sent)}", "type": "message", "role": "assistant", "model": body["model"],
                "content": [{"type": "text", "text": answers[len(sent) - 1]}], "stop_reason": "end_turn",
                "stop_sequence": None, "usage": {"input_tokens": 10, "output_tokens": 5},
            })

        with httpx2.Client(transport=httpx2.MockTransport(handler)) as http:
            client = anthropic.Anthropic(api_key="test-key-not-real", max_retries=0, http_client=http)
            llm = AnthropicLLM(client=client, sleep=lambda seconds: None)
            result = run_cycle(conn, llm, config=config, snapshot=poisoned)
        assert result.outcome == "proposal" and len(sent) == 3
        scan_prompt = sent[0]["messages"][0]["content"]
        assert "Chipmaker rallies \ufffd on capex (feed)" in scan_prompt
        assert "summary: Summary rallies \ufffd on capex" in scan_prompt

    def test_blank_invalidation_never_becomes_a_proposal(self, conn, snapshot, config):
        llm = FakeLLM.from_fixtures(
            {**OK, "proposal": ["proposal_blank_invalidation.json"] * (config.brain.max_retries + 1)}
        )
        with pytest.raises(BrainError, match="proposal output invalid after 4 attempts"):
            run_cycle(conn, llm, config=config, snapshot=snapshot)
        assert get_recent_proposals(conn) == []
        assert all("invalidation must not be blank" in call.messages[-1].content for call in llm.calls[3:])
        assert _kinds(conn) == ["cycle_start", "market_snapshot", "brain_error", "cycle_end"]

    def test_closed_market_brief_carries_the_staleness_warning(self, conn, closed_snapshot, config):
        llm = FakeLLM.from_fixtures({"scan": ["scan_no_warnings.json"], "thesis": ["thesis_empty.json"]})
        assert ScanBrief.model_validate_json(_text("scan_no_warnings.json")).staleness_warnings == ()
        result = run_cycle(conn, llm, config=config, snapshot=closed_snapshot)
        assert result.outcome == "no_trade"
        assert result.brief is not None
        assert result.brief.staleness_warnings == closed_snapshot.data_age.warnings
        assert result.brief.staleness_warnings[0].startswith("market is CLOSED")
        # the row keeps the model's text as validated; the merge is on the result
        (scan_row, _) = get_cycle_reasoning(conn, result.cycle_id)
        assert scan_row.content == _text("scan_no_warnings.json")
        assert "market CLOSED" in _event(conn, MARKET_SNAPSHOT_EVENT).message

    def test_first_resolvable_candidate_is_offered(self, conn, snapshot, config):
        blind = _with_contract(snapshot, "SPY260821C00650000", mid=None)  # the spread cannot be priced
        llm = FakeLLM({**{s: [_text(n) for n in names] for s, names in OK.items()}, "proposal": [EQUITY_TRADE]})
        result = run_cycle(conn, llm, config=config, snapshot=blind)
        assert result.outcome == "proposal"
        assert result.proposal is not None
        assert (result.proposal.symbol, result.proposal.instrument, result.proposal.legs) == ("QQQ", Instrument.EQUITY, ())
        assert "equity QQQ" in llm.calls[2].messages[0].content
        proposal = get_proposal(conn, result.proposal_id)
        assert proposal is not None and proposal.symbol == "QQQ" and proposal.quantity == 5.0
        assert get_proposal_legs(conn, proposal.id) == []
        assert _event(conn, "proposal").payload["symbol"] == "QQQ"

    def test_no_resolvable_candidate_is_a_no_trade_listing_the_reasons(self, conn, snapshot, config):
        blind = _with_contract(snapshot, "SPY260821C00640000", mid=None)
        spy, qqq = blind.symbols
        blind = blind.model_copy(update={"symbols": (spy, qqq.model_copy(update={"spot": None}))})
        llm = _fake()
        result = run_cycle(conn, llm, config=config, snapshot=blind)
        assert result.outcome == "no_trade"
        assert len(llm.calls) == 2 and llm.remaining("proposal") == 1
        assert len(result.candidates) == 2
        assert result.no_trade_reason == (
            "SPY call_debit_spread 640/650 exp 2026-08-21: SPY260821C00640000 has no mid price; "
            "QQQ equity: no spot price"
        )
        no_trade = _event(conn, "no_trade")
        assert no_trade.payload == {"cycle_id": result.cycle_id, "reason": result.no_trade_reason}
        assert "could be resolved" in no_trade.message
        assert [row.stage for row in get_cycle_reasoning(conn, result.cycle_id)] == STAGES[:2]

    def test_equity_without_a_two_sided_quote_is_skipped(self, conn, snapshot, config):
        """Outside hours IEX keeps the last trade but drops the bid/ask: the equity idea
        cannot have its limit price bounded, so it is skipped like an option with no mid."""
        blind = _with_contract(snapshot, "SPY260821C00650000", mid=None)
        spy, qqq = blind.symbols
        blind = blind.model_copy(update={"symbols": (spy, qqq.model_copy(update={"bid": None, "ask": None}))})
        llm = _fake()
        result = run_cycle(conn, llm, config=config, snapshot=blind)
        assert result.outcome == "no_trade" and len(llm.calls) == 2
        assert result.no_trade_reason == (
            "SPY call_debit_spread 640/650 exp 2026-08-21: SPY260821C00650000 has no mid price; "
            "QQQ equity: no bid/ask quote (bid n/a, ask n/a) — a limit price could not be bounded"
        )
        assert get_recent_proposals(conn) == []

    def test_recent_proposals_are_shown_to_the_thesis_stage(self, conn, snapshot, config):
        first = run_cycle(conn, _fake(), config=config, snapshot=snapshot)
        llm = _fake()
        run_cycle(conn, llm, config=config, snapshot=snapshot)
        assert snapshot.account is not None
        expected = build_thesis_prompt(
            merge_staleness_warnings(ScanBrief.model_validate_json(_text("scan_ok.json")), snapshot),
            snapshot,
            snapshot.account.positions,
            config.risk_limits,
            get_recent_proposals(conn, config.brain.snapshot.recent_proposals_limit)[1:],  # only the first cycle's
        )
        assert llm.calls[1].messages[0].content == expected
        assert first.proposal_id in expected
        # a limit of 0 shows none
        none_shown = config.model_copy(
            update={"brain": config.brain.model_copy(update={"snapshot": config.brain.snapshot.model_copy(update={"recent_proposals_limit": 0})})}
        )
        llm = _fake()
        run_cycle(conn, llm, config=none_shown, snapshot=snapshot)
        assert first.proposal_id not in llm.calls[1].messages[0].content

    def test_snapshot_is_built_when_not_given(self, conn, snapshot, config, monkeypatch):
        seen = []

        def fake_build(symbols=None, *, config=None, now=None):
            seen.append((symbols, config, now))
            return snapshot

        monkeypatch.setattr("aegis.brain.cycle.build_market_snapshot", fake_build)
        when = datetime(2026, 7, 30, 19, 45, tzinfo=timezone.utc)
        result = run_cycle(conn, _fake(), config=config, now=when)
        assert result.outcome == "proposal"
        assert seen == [(None, config, when)]
        # cycle_start names what the cycle set out to scan: the config watchlist
        assert _event(conn, "cycle_start").payload["symbols"] == config.watchlist

    def test_default_config_is_the_repo_config(self, conn, snapshot, config, monkeypatch):
        seen = []
        monkeypatch.setattr("aegis.brain.cycle.get_config", lambda: (seen.append(True), config)[1])
        result = run_cycle(conn, _fake(), snapshot=snapshot)
        assert seen == [True] and result.outcome == "proposal"

    def test_snapshot_build_failure_is_a_failed_cycle(self, conn, config, monkeypatch):
        def boom(symbols=None, *, config=None, now=None):
            raise BrainError("build market snapshot", cause=ValueError("no symbols to snapshot"))

        monkeypatch.setattr("aegis.brain.cycle.build_market_snapshot", boom)
        llm = _fake()
        with pytest.raises(BrainError, match="build market snapshot"):
            run_cycle(conn, llm, config=config)
        assert llm.calls == []
        assert _kinds(conn) == ["cycle_start", "brain_error", "cycle_end"]
        error = _event(conn, "brain_error").payload
        assert error is not None and error["stage"] is None and "no symbols to snapshot" in error["error"]
        assert _event(conn, "cycle_end").payload["outcome"] == "failed"

    def test_every_event_of_a_cycle_carries_its_id(self, conn, snapshot, config):
        result = run_cycle(conn, _fake(), config=config, snapshot=snapshot)
        assert {e.payload["cycle_id"] for e in _events(conn) if e.payload} == {result.cycle_id}


# --- the pure helpers ---------------------------------------------------------


class TestHelpers:
    def test_reasoning_row(self):
        usage = StageUsage(stage=ReasoningStage.THESIS, model="m", tokens_in=7, tokens_out=3, latency_ms=2.5, attempts=2)
        row = reasoning_row("cycle-1", usage, "{}")
        assert (row.cycle_id, row.proposal_id, row.stage, row.content) == ("cycle-1", None, ReasoningStage.THESIS, "{}")
        assert (row.tokens_in, row.tokens_out, row.model_name, row.latency_ms) == (7, 3, "m", 2.5)

    def test_proposal_records(self, snapshot, config):
        from aegis.brain.proposal import resolve_offer

        thesis = ThesisOutput.model_validate_json(_text("thesis_ok.json"))
        offer = resolve_offer(thesis.candidates[0], snapshot, contract_multiplier=100)
        result = run_proposal(FakeLLM.from_fixtures({"proposal": ["proposal_ok.json"]}), offer, snapshot, config.brain)
        proposal, legs = proposal_records("cycle-1", result)
        assert proposal.cycle_id == "cycle-1" and proposal.quantity == 2.0
        assert proposal.raw_model_output == _text("proposal_ok.json")
        assert proposal.model_name == config.brain.stages.proposal.model
        assert proposal.prompt_version == result.prompt_version
        assert [(leg.proposal_id, leg.leg_index, leg.symbol, leg.quantity) for leg in legs] == [
            (proposal.id, 0, "SPY260821C00640000", 2.0),
            (proposal.id, 1, "SPY260821C00650000", 2.0),
        ]
        declined = run_proposal(FakeLLM.from_fixtures({"proposal": ["proposal_no_trade.json"]}), offer, snapshot, config.brain)
        with pytest.raises(ValueError, match="did not trade"):
            proposal_records("cycle-1", declined)

    def test_first_offer(self, snapshot):
        thesis = ThesisOutput.model_validate_json(_text("thesis_ok.json"))
        offer, skipped = first_offer(thesis.candidates, snapshot, contract_multiplier=100)
        assert offer is not None and offer.candidate is thesis.candidates[0] and skipped == ()
        blind = _with_contract(snapshot, "SPY260821C00650000", mid=None)
        offer, skipped = first_offer(thesis.candidates, blind, contract_multiplier=100)
        assert offer is not None and offer.candidate is thesis.candidates[1]
        assert skipped == ("SPY call_debit_spread 640/650 exp 2026-08-21: SPY260821C00650000 has no mid price",)
        assert first_offer((), snapshot, contract_multiplier=100) == (None, ())

    def test_market_snapshot_event_round_trips(self, snapshot, closed_snapshot):
        event = market_snapshot_event("cycle-1", snapshot)
        assert event.kind == MARKET_SNAPSHOT_EVENT and event.level is EventLevel.INFO
        assert event.payload is not None and event.payload["cycle_id"] == "cycle-1"
        assert snapshot_from_event(event) == snapshot
        assert snapshot_from_event(market_snapshot_event("c", closed_snapshot)) == closed_snapshot
        with pytest.raises(ValueError, match="not a market snapshot"):
            snapshot_from_event(Event(level=EventLevel.INFO, kind="cycle_start", message="x", payload={}))
        broken = event.model_copy(update={"payload": {"cycle_id": "c", "symbols": []}})
        with pytest.raises(ValueError):
            snapshot_from_event(broken)


# --- the CLI ------------------------------------------------------------------


class TestCliOnce:
    def test_prints_the_report_and_exits_0(self, db_path, snapshot, capsys):
        llm = _fake()
        assert cli_brain.main(["once", "--db", str(db_path)], llm=llm, snapshot_builder=_builder(snapshot)) == 0
        captured = capsys.readouterr()
        assert captured.err == ""
        out = captured.out
        lines = out.splitlines()
        assert lines[0].startswith("cycle ")
        assert "  market    OPEN (next close 2026-07-30 20:00:00 UTC)" in lines
        assert "  symbols   SPY, QQQ" in lines
        assert "  outcome   proposal" in lines
        assert "brief" in lines and "  SPY" in lines and "  QQQ" in lines
        assert "    IV: ATM implied vol 18.45% (640 call, vendor)" in out
        assert "  staleness warnings (2)" in lines
        assert "    - free-plan options quotes run ~15 minutes behind (indicative feed)" in lines
        assert "candidates (2)" in lines
        assert "  [0] bullish SPY call_debit_spread 640/650 exp 2026-08-21   confidence 0.58" in lines
        assert "  [1] bullish QQQ equity   confidence 0.50" in lines
        assert "      invalidation: A daily close below 630 before 2026-08-14." in lines
        assert "  SPY (option)   buy 2 limit @ 4.55   confidence 0.58" in lines
        assert "  legs (2)" in lines
        assert "    [0] buy 2 call 640.0 exp 2026-08-21  SPY260821C00640000" in lines
        assert "    [1] sell 2 call 650.0 exp 2026-08-21  SPY260821C00650000" in lines
        assert "tokens" in lines
        assert "  scan      in 100 / out 50   attempts 1   latency 1 ms   claude-haiku-4-5-20251001" in lines
        assert "  total     in 300 / out 150 = 450" in lines
        assert "ESTIMATE from config brain.prices_per_mtok placeholders — not a bill" in out
        (proposal_line,) = [line for line in lines if line.startswith("proposal ")]
        proposal_id = proposal_line.split()[1]
        assert lines[-1] == f"next: python -m aegis.cli.trace {proposal_id}"
        assert out.index("brief") < out.index("candidates (2)") < out.index("proposal ") < out.index("tokens")

        # the trace CLI then shows the linked lineage
        assert cli_trace.main([proposal_id, "--db", str(db_path)]) == 0
        trace_out = capsys.readouterr().out
        assert "reasoning (3)" in trace_out and "legs (2)" in trace_out

    def test_no_trade_prints_the_reason_and_exits_0(self, db_path, snapshot, capsys):
        llm = FakeLLM.from_fixtures({"scan": ["scan_ok.json"], "thesis": ["thesis_empty.json"]})
        assert cli_brain.main(["once", "--db", str(db_path)], llm=llm, snapshot_builder=_builder(snapshot)) == 0
        out = capsys.readouterr().out
        assert "  outcome   no_trade" in out.splitlines()
        assert "candidates (0)" in out
        assert "  NO_TRADE: No edge at these levels" in out
        assert "next:" not in out and "ESTIMATE" in out

    def test_halted_cycle_exits_2(self, db_path, conn, snapshot, config, capsys):
        _seed(conn, "cycle-earlier", config.brain.daily_token_budget)
        llm = _fake()
        assert cli_brain.main(["once", "--db", str(db_path)], llm=llm, snapshot_builder=_builder(snapshot)) == 2
        captured = capsys.readouterr()
        assert captured.err == ""
        lines = captured.out.splitlines()
        assert "  outcome   halted" in lines
        assert "  (none — the cycle halted before the scan stage)" in lines
        assert any(line.startswith("  HALTED: brain failed: budget halt: daily limit would be exceeded") for line in lines)
        assert "  (no calls made)" in lines and "ESTIMATE" in captured.out
        assert llm.calls == []

    def test_failure_is_one_line_on_stderr_and_exit_1(self, db_path, snapshot, config, capsys):
        llm = FakeLLM.from_fixtures({**OK, "proposal": ["malformed.json"] * (config.brain.max_retries + 1)})
        assert cli_brain.main(["once", "--db", str(db_path)], llm=llm, snapshot_builder=_builder(snapshot)) == 1
        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err.startswith("brain once failed: brain failed: proposal output invalid after 4 attempts")
        assert captured.err.count("\n") == 1 and "Traceback" not in captured.err

    def test_symbols_flag_reaches_the_builder(self, db_path, snapshot, capsys):
        seen: list = []
        assert cli_brain.main(["once", "--db", str(db_path), "--symbols", "spy, qqq"], llm=_fake(), snapshot_builder=_builder(snapshot, seen)) == 0
        assert cli_brain.main(["once", "--db", str(db_path)], llm=_fake(), snapshot_builder=_builder(snapshot, seen)) == 0
        assert [call[0] for call in seen] == [["SPY", "QQQ"], None]
        assert all(call[1] is get_config() for call in seen)
        capsys.readouterr()
        assert cli_brain.main(["once", "--db", str(db_path), "--symbols", " , "], llm=_fake(), snapshot_builder=_builder(snapshot)) == 1
        assert capsys.readouterr().err == (
            "python -m aegis.cli.brain once: error: --symbols needs at least one ticker, got ' , ' (see --help)\n"
        )

    @pytest.mark.parametrize(
        "argv, message",
        [
            (["once", "--nope"], "python -m aegis.cli.brain: error: unrecognized arguments: --nope"),
            (["stage", "review"], "python -m aegis.cli.brain stage: error: argument stage: invalid choice: 'review'"),
            ([], "python -m aegis.cli.brain: error: the following arguments are required: command"),
        ],
        ids=["unknown-flag", "bad-choice", "no-command"],
    )
    def test_a_bad_command_line_is_one_line_and_exit_1_never_2(self, db_path, snapshot, capsys, argv, message):
        """Exit 2 means the budget guard halted the cycle, so argparse's own exit 2 would make a
        mistyped invocation indistinguishable from a halt to a scheduler."""
        llm = _fake()
        assert cli_brain.main(argv, llm=llm, snapshot_builder=_builder(snapshot)) == 1
        captured = capsys.readouterr()
        assert captured.out == "" and captured.err.startswith(message)
        assert captured.err.count("\n") == 1 and captured.err.endswith("(see --help)\n")
        assert llm.calls == [] and not db_path.exists()
        with pytest.raises(SystemExit) as info:  # --help still prints and exits 0
            cli_brain.main(["--help"])
        assert info.value.code == 0 and "once" in capsys.readouterr().out

    def test_unexpected_errors_never_dump_a_traceback(self, db_path, snapshot, capsys):
        def broken(symbols=None, *, config=None, now=None):
            raise RuntimeError("boom")

        assert cli_brain.main(["once", "--db", str(db_path)], llm=_fake(), snapshot_builder=broken) == 1
        captured = capsys.readouterr()
        assert captured.err == "brain once failed (unexpected): RuntimeError: boom\n"

    def test_model_text_cannot_restyle_the_terminal_or_forge_report_lines(self, db_path, snapshot, capsys):
        """Every model-authored field prints on one line with control and format characters
        removed: an ESC could conceal the rest of the report, and a newline could forge a
        NO_TRADE, a tokens block or a next: line."""
        brief = json.loads(_text("scan_ok.json"))
        brief["symbols"][0]["summary"] = "SPY drifts.\n  proposal\n    NO_TRADE: nothing to do\x1b[8m"
        brief["notable_observations"] = ["\x1b]0;owned\x07 title"]
        thesis = json.loads(_text("thesis_ok.json"))
        thesis["candidates"][0]["rationale"] = "Up.\r\ntokens\n  total     in 0 / out 0 = 0‮"
        proposal = json.loads(_text("proposal_ok.json"))
        proposal["proposal"]["thesis"] = "Buy it.\n\nnext: python -m aegis.cli.trace fake-id\x1b[2J"
        llm = FakeLLM({"scan": [json.dumps(brief)], "thesis": [json.dumps(thesis)], "proposal": [json.dumps(proposal)]})
        assert cli_brain.main(["once", "--db", str(db_path)], llm=llm, snapshot_builder=_builder(snapshot)) == 0
        out = capsys.readouterr().out
        assert not any(ch in out for ch in ("\x1b", "\r", "\x07", "‮"))
        lines = out.splitlines()
        assert [line for line in lines if "NO_TRADE" in line] == ["    SPY drifts. proposal NO_TRADE: nothing to do [8m"]
        assert lines.count("tokens") == 1
        assert "      rationale: Up. tokens total in 0 / out 0 = 0" in lines
        assert "    Buy it. next: python -m aegis.cli.trace fake-id [2J" in lines
        assert [line for line in lines if line.startswith("next: ")] == [lines[-1]]

    def test_an_ascii_stdout_neither_fails_nor_loses_the_report(self, db_path, snapshot, monkeypatch):
        """A cron or daemon locale (LANG=C) must not turn a committed cycle into an exit 1 that
        loses the ESTIMATE and next: lines: a character stdout lacks prints as an escape."""
        buffer = io.BytesIO()
        stdout = io.TextIOWrapper(buffer, encoding="ascii")
        monkeypatch.setattr(sys, "stdout", stdout)
        assert cli_brain.main(["once", "--db", str(db_path)], llm=_fake(), snapshot_builder=_builder(snapshot)) == 0
        stdout.flush()
        out = buffer.getvalue().decode("ascii")
        assert "(ESTIMATE from config brain.prices_per_mtok placeholders \\u2014 not a bill)" in out
        assert "\nnext: python -m aegis.cli.trace " in out

    def test_keyboard_interrupt_is_130(self, db_path, snapshot):
        def interrupted(symbols=None, *, config=None, now=None):
            raise KeyboardInterrupt

        assert cli_brain.main(["once", "--db", str(db_path)], llm=_fake(), snapshot_builder=interrupted) == 130

    def test_keyboard_interrupt_mid_stage_is_130_with_the_spend_journaled(self, db_path, snapshot):
        llm = FakeLLM({"scan": [_text("scan_ok.json")], "thesis": [_text("malformed.json"), KeyboardInterrupt()]})
        assert cli_brain.main(["once", "--db", str(db_path)], llm=llm, snapshot_builder=_builder(snapshot)) == 130
        conn = open_store(db_path)
        try:
            cycle_id = llm.calls[0].cycle_id
            assert get_cycle_token_usage(conn, cycle_id) == TokenUsage(tokens_in=200, tokens_out=100, calls=2)
            assert _kinds(conn) == ["cycle_start", "market_snapshot", "brain_error", "cycle_end"]
            assert _event(conn, "cycle_end").payload["outcome"] == "failed"
        finally:
            conn.close()


class TestCliStage:
    def test_scan_writes_a_row_and_the_snapshot_event(self, db_path, snapshot, capsys):
        llm = FakeLLM.from_fixtures({"scan": ["scan_ok.json"]})
        assert cli_brain.main(["stage", "scan", "--db", str(db_path)], llm=llm, snapshot_builder=_builder(snapshot)) == 0
        captured = capsys.readouterr()
        assert captured.err == ""
        out = captured.out
        assert out.splitlines()[0].endswith("stage scan   prior stages run live")
        assert "snapshot: SPY, QQQ, taken 2026-07-30 19:45:00 UTC, market OPEN" in out
        assert "output (scan-v2+system-v1):\n" + _text("scan_ok.json").rstrip("\n") + "\n" in out
        assert "  scan      in 100 / out 50" in out and "ESTIMATE" in out
        conn = open_store(db_path)
        try:
            (cycle_id,) = {call.cycle_id for call in llm.calls}
            rows = get_cycle_reasoning(conn, cycle_id)
            assert [row.stage for row in rows] == [ReasoningStage.SCAN]
            assert rows[0].content == _text("scan_ok.json")
            assert rows[0].model_name == "fake-model"  # the model that ANSWERED, not the configured one
            assert _kinds(conn) == [MARKET_SNAPSHOT_EVENT]
            assert _event(conn, MARKET_SNAPSHOT_EVENT).payload["cycle_id"] == cycle_id
        finally:
            conn.close()

    def test_thesis_with_cycle_reuses_the_prior_rows(self, db_path, snapshot, closed_snapshot, config, capsys):
        assert cli_brain.main(["once", "--db", str(db_path)], llm=_fake(), snapshot_builder=_builder(snapshot)) == 0
        capsys.readouterr()
        conn = open_store(db_path)
        try:
            (proposal,) = get_recent_proposals(conn)
            prior = proposal.cycle_id
        finally:
            conn.close()
        # a later cycle on a different (closed-market) snapshot: --cycle must not pick up its event
        later = FakeLLM.from_fixtures({"scan": ["scan_ok.json"], "thesis": ["thesis_empty.json"]})
        assert cli_brain.main(["once", "--db", str(db_path)], llm=later, snapshot_builder=_builder(closed_snapshot)) == 0
        capsys.readouterr()

        llm = FakeLLM.from_fixtures({"thesis": ["thesis_empty.json"]})
        assert cli_brain.main(["stage", "thesis", "--db", str(db_path), "--cycle", prior], llm=llm) == 0
        out = capsys.readouterr().out
        assert f"stage thesis   prior outputs from cycle {prior}" in out
        assert "No edge at these levels" in out and "  thesis    in 100 / out 50" in out
        assert "  scan " not in out  # nothing else ran
        (call,) = llm.calls
        assert call.stage is ReasoningStage.THESIS and call.cycle_id != prior
        assert snapshot.account is not None
        brief = merge_staleness_warnings(ScanBrief.model_validate_json(_text("scan_ok.json")), snapshot)
        conn = open_store(db_path)
        try:
            recent = get_recent_proposals(conn, config.brain.snapshot.recent_proposals_limit)
            assert call.messages[0].content == build_thesis_prompt(brief, snapshot, snapshot.account.positions, config.risk_limits, recent)
            assert "market CLOSED" not in call.messages[0].content  # the prior cycle's OPEN snapshot
            rows = get_cycle_reasoning(conn, call.cycle_id)
            assert [(row.stage, row.model_name) for row in rows] == [(ReasoningStage.THESIS, "fake-model")]
            assert [row.stage for row in get_cycle_reasoning(conn, prior)] == STAGES  # untouched
            # the fresh cycle logs its own copy of the input it was shown, naming where it came from
            (own,) = [e for e in _events(conn) if e.kind == MARKET_SNAPSHOT_EVENT and e.payload["cycle_id"] == call.cycle_id]
            assert snapshot_from_event(own) == snapshot and f"reloaded from cycle {prior}" in own.message
        finally:
            conn.close()

    def test_a_stage_run_with_cycle_can_feed_the_next_stage(self, db_path, snapshot, capsys):
        """stage scan -> stage thesis --cycle <scan's> -> stage proposal --cycle <thesis's>: each
        fresh cycle carries its own market_snapshot event, so the debugging chain holds."""
        scan = FakeLLM.from_fixtures({"scan": ["scan_ok.json"]})
        assert cli_brain.main(["stage", "scan", "--db", str(db_path)], llm=scan, snapshot_builder=_builder(snapshot)) == 0
        thesis = FakeLLM.from_fixtures({"thesis": ["thesis_ok.json"]})
        assert cli_brain.main(["stage", "thesis", "--db", str(db_path), "--cycle", scan.calls[0].cycle_id], llm=thesis) == 0
        proposal = FakeLLM.from_fixtures({"proposal": ["proposal_ok.json"]})
        assert cli_brain.main(["stage", "proposal", "--db", str(db_path), "--cycle", thesis.calls[0].cycle_id], llm=proposal) == 0
        out = capsys.readouterr().out
        assert "offer: SPY call_debit_spread 640/650 exp 2026-08-21" in out and '"outcome": "trade"' in out
        conn = open_store(db_path)
        try:
            cycles = [llm.calls[0].cycle_id for llm in (scan, thesis, proposal)]
            snapshots = [e.payload["cycle_id"] for e in _events(conn) if e.kind == MARKET_SNAPSHOT_EVENT]
            assert snapshots == cycles
            assert [[row.stage for row in get_cycle_reasoning(conn, c)] for c in cycles] == [[stage] for stage in STAGES]
        finally:
            conn.close()

    def test_proposal_without_cycle_runs_the_prior_stages_first(self, db_path, snapshot, capsys):
        llm = FakeLLM.from_fixtures({**OK, "proposal": ["proposal_ok.json"]})
        assert cli_brain.main(["stage", "proposal", "--db", str(db_path)], llm=llm, snapshot_builder=_builder(snapshot)) == 0
        out = capsys.readouterr().out
        assert "offer: SPY call_debit_spread 640/650 exp 2026-08-21" in out
        assert "output (proposal-v1+system-v1):" in out and '"outcome": "trade"' in out
        assert [call.stage for call in llm.calls] == STAGES
        (cycle_id,) = {call.cycle_id for call in llm.calls}
        conn = open_store(db_path)
        try:
            assert [row.stage for row in get_cycle_reasoning(conn, cycle_id)] == STAGES
            assert get_recent_proposals(conn) == []  # a stage run never stores a proposal
        finally:
            conn.close()

    def test_proposal_with_cycle_that_had_no_candidates(self, db_path, snapshot, capsys):
        llm = FakeLLM.from_fixtures({"scan": ["scan_ok.json"], "thesis": ["thesis_empty.json"]})
        assert cli_brain.main(["once", "--db", str(db_path)], llm=llm, snapshot_builder=_builder(snapshot)) == 0
        capsys.readouterr()
        prior = llm.calls[0].cycle_id
        llm = FakeLLM.from_fixtures({"proposal": ["proposal_ok.json"]})
        assert cli_brain.main(["stage", "proposal", "--db", str(db_path), "--cycle", prior], llm=llm) == 0
        out = capsys.readouterr().out
        assert "NO_TRADE: the thesis stage found no compelling idea: No edge" in out
        assert llm.calls == [] and "  (no calls made)" in out

    def test_proposal_with_cycle_skips_unresolvable_candidates(self, db_path, snapshot, capsys):
        blind = _with_contract(snapshot, "SPY260821C00650000", mid=None)
        llm = FakeLLM({**{s: [_text(n) for n in names] for s, names in OK.items()}, "proposal": [EQUITY_TRADE]})
        assert cli_brain.main(["once", "--db", str(db_path)], llm=llm, snapshot_builder=_builder(blind)) == 0
        capsys.readouterr()
        prior = llm.calls[0].cycle_id
        llm = FakeLLM({"proposal": [EQUITY_TRADE]})
        assert cli_brain.main(["stage", "proposal", "--db", str(db_path), "--cycle", prior], llm=llm) == 0
        out = capsys.readouterr().out
        assert "skipped: SPY call_debit_spread 640/650 exp 2026-08-21: SPY260821C00650000 has no mid price" in out
        assert "offer: QQQ equity" in out
        assert len(llm.calls) == 1

    def test_unknown_cycle_or_missing_prior_row_exits_1(self, db_path, conn, snapshot, config, capsys):
        assert cli_brain.main(["stage", "thesis", "--db", str(db_path), "--cycle", "nope"], llm=_fake()) == 1
        err = capsys.readouterr().err
        assert err.startswith("brain stage failed: brain failed: load market snapshot for nope")
        # a cycle that halted before the scan has a snapshot but no scan row
        _seed(conn, "cycle-earlier", config.brain.daily_token_budget)
        assert cli_brain.main(["once", "--db", str(db_path)], llm=_fake(), snapshot_builder=_builder(snapshot)) == 2
        capsys.readouterr()
        halted = _event(conn, "budget_halt").payload["cycle_id"]
        assert cli_brain.main(["stage", "thesis", "--db", str(db_path), "--cycle", halted], llm=_fake()) == 1
        err = capsys.readouterr().err
        assert "load prior stage output" in err and "no scan reasoning row" in err

    def test_halted_stage_exits_2(self, db_path, conn, snapshot, config, capsys):
        _seed(conn, "cycle-earlier", config.brain.daily_token_budget)
        llm = _fake()
        assert cli_brain.main(["stage", "scan", "--db", str(db_path)], llm=llm, snapshot_builder=_builder(snapshot)) == 2
        out = capsys.readouterr().out
        assert "HALTED: brain failed: budget halt: daily limit would be exceeded" in out
        assert llm.calls == []

    @pytest.mark.parametrize("failure", ["malformed", "halted-mid-scan"])
    def test_thesis_with_a_cycle_whose_scan_failed_is_a_brain_error(self, db_path, snapshot, config, capsys, failure):
        """The failed scan's row holds its last invalid text (or a no-usable-output line): a clean
        BrainError naming the cycle, not an '(unexpected)' OutputInvalid."""
        if failure == "malformed":
            llm = FakeLLM.from_fixtures({"scan": ["malformed.json"] * (config.brain.max_retries + 1)})
            expected_code = 1
        else:
            cap = config.brain.per_cycle_token_cap
            llm = FakeLLM.from_fixtures({"scan": ["malformed.json", "scan_ok.json"]}, tokens_in=cap - 2000, tokens_out=50)
            expected_code = 2
        assert cli_brain.main(["once", "--db", str(db_path)], llm=llm, snapshot_builder=_builder(snapshot)) == expected_code
        capsys.readouterr()
        prior = llm.calls[0].cycle_id
        again = _fake()
        assert cli_brain.main(["stage", "thesis", "--db", str(db_path), "--cycle", prior], llm=again) == 1
        err = capsys.readouterr().err
        assert err == (
            f"brain stage failed: brain failed: stage thesis: cycle {prior} has no valid scan output "
            "(OutputInvalid: output is not valid JSON: Expecting value (position 0))\n"
        )
        assert again.calls == []

    @pytest.mark.parametrize(
        "fixture_name, problem",
        [
            ("thesis_blank_invalidation.json",
             "output did not validate against the schema: candidates.0.invalidation: Value error, "
             "invalidation must not be blank"),
            ("thesis_bad_strike.json",
             "output did not match the market snapshot: candidates.0.strikes: strike 660 is not in the SPY chain"),
        ],
        ids=["schema", "post-validate"],
    )
    def test_proposal_with_a_cycle_whose_thesis_failed_is_a_brain_error(
        self, db_path, snapshot, config, capsys, fixture_name, problem
    ):
        llm = FakeLLM.from_fixtures(
            {"scan": ["scan_ok.json"], "thesis": [fixture_name] * (config.brain.max_retries + 1)}
        )
        assert cli_brain.main(["once", "--db", str(db_path)], llm=llm, snapshot_builder=_builder(snapshot)) == 1
        capsys.readouterr()
        prior = llm.calls[0].cycle_id
        assert cli_brain.main(["stage", "proposal", "--db", str(db_path), "--cycle", prior], llm=_fake()) == 1
        err = capsys.readouterr().err
        assert err.startswith(
            f"brain stage failed: brain failed: stage proposal: cycle {prior} has no valid thesis output "
            f"(OutputInvalid: {problem}"
        )
        assert err.count("\n") == 1 and "(unexpected)" not in err  # a multi-line schema message, on one line

    def test_halted_stage_journals_the_attempt_it_received(self, db_path, snapshot, config, capsys):
        cap = config.brain.per_cycle_token_cap
        llm = FakeLLM.from_fixtures({"scan": ["malformed.json", "scan_ok.json"]}, tokens_in=cap - 2000, tokens_out=50)
        assert cli_brain.main(["stage", "scan", "--db", str(db_path)], llm=llm, snapshot_builder=_builder(snapshot)) == 2
        out = capsys.readouterr().out
        assert "HALTED: brain failed: budget halt: cycle limit would be exceeded" in out
        assert f"  scan      in {cap - 2000:,} / out 50   attempts 1" in out  # counted in the totals too
        conn = open_store(db_path)
        try:
            (row,) = get_cycle_reasoning(conn, llm.calls[0].cycle_id)
            assert (row.stage, row.tokens_in, row.tokens_out, row.model_name) == (ReasoningStage.SCAN, cap - 2000, 50, "fake-model")
            assert row.content.startswith("(no usable output: BudgetExceeded: brain failed: budget halt:")
        finally:
            conn.close()

    def test_failed_stage_journals_its_attempts(self, db_path, snapshot, config, capsys):
        llm = FakeLLM.from_fixtures({"scan": ["malformed.json"] * (config.brain.max_retries + 1)})
        assert cli_brain.main(["stage", "scan", "--db", str(db_path)], llm=llm, snapshot_builder=_builder(snapshot)) == 1
        err = capsys.readouterr().err
        assert err.startswith("brain stage failed: brain failed: scan output invalid after 4 attempts")
        conn = open_store(db_path)
        try:
            cycle_id = llm.calls[0].cycle_id
            (row,) = get_cycle_reasoning(conn, cycle_id)
            assert (row.stage, row.tokens_in, row.tokens_out, row.content) == (ReasoningStage.SCAN, 400, 200, _text("malformed.json"))
            # after the final failure: BrainError AND an event, as the cycle logs it
            assert _kinds(conn) == [MARKET_SNAPSHOT_EVENT, "brain_error"]
            error = _event(conn, "brain_error")
            assert error.level is EventLevel.ERROR
            assert error.payload["cycle_id"] == cycle_id and error.payload["stage"] == "scan"
            assert error.payload["error"].startswith("brain failed: scan output invalid after 4 attempts")
        finally:
            conn.close()

    def test_a_halted_stage_logs_budget_halt_not_brain_error(self, db_path, conn, snapshot, config, capsys):
        _seed(conn, "cycle-earlier", config.brain.daily_token_budget)
        assert cli_brain.main(["stage", "scan", "--db", str(db_path)], llm=_fake(), snapshot_builder=_builder(snapshot)) == 2
        assert _kinds(conn) == [MARKET_SNAPSHOT_EVENT, "budget_halt"]

    def test_interrupted_stage_journals_the_attempt_it_received(self, db_path, snapshot):
        llm = FakeLLM({"scan": [_text("malformed.json"), KeyboardInterrupt()]})
        assert cli_brain.main(["stage", "scan", "--db", str(db_path)], llm=llm, snapshot_builder=_builder(snapshot)) == 130
        conn = open_store(db_path)
        try:
            (row,) = get_cycle_reasoning(conn, llm.calls[0].cycle_id)
            assert (row.stage, row.tokens_in, row.tokens_out, row.model_name) == (ReasoningStage.SCAN, 100, 50, "fake-model")
            assert row.content == "(no usable output: KeyboardInterrupt)"
            assert _event(conn, "brain_error").payload == {
                "cycle_id": llm.calls[0].cycle_id, "stage": "scan", "error": "KeyboardInterrupt",
            }
        finally:
            conn.close()


class TestCliUsage:
    def test_prints_tokens_by_model_and_the_estimate(self, db_path, conn, config, capsys):
        _seed(conn, "c1", 1000, 200, model_name="claude-haiku-4-5-20251001")
        _seed(conn, "c1", 300, 50, model_name="claude-opus-5-5", stage=ReasoningStage.THESIS)
        _seed(conn, "c2", 5000, 0, created_at=datetime.now(timezone.utc) - timedelta(days=2))  # not today
        assert cli_brain.main(["usage", "--db", str(db_path)]) == 0
        captured = capsys.readouterr()
        assert captured.err == ""
        lines = captured.out.splitlines()
        assert lines[0] == f"token usage for {datetime.now(timezone.utc).date().isoformat()} (UTC)"
        assert "  tokens    in 1,300 / out 250 = 1,550 over 2 reasoning row(s)" in lines
        assert "  by model" in lines
        assert any(line.startswith("    claude-haiku-4-5-20251001") and "in 1,000 / out 200   rows 1   est. $0.0020" in line for line in lines)
        assert any(line.startswith("    claude-opus-5-5") and "in 300 / out 50   rows 1   est. $0.0022" in line for line in lines)
        assert "  estimated spend: $0.0042 (ESTIMATE from config brain.prices_per_mtok placeholders — not a bill)" in lines
        budget = config.brain.daily_token_budget
        assert f"  daily budget: 1,550 / {budget:,} tokens ({1550 / budget * 100:.1f}% used)" in lines

    def test_unknown_model_makes_the_spend_unavailable(self, db_path, conn, capsys):
        _seed(conn, "c1", 100, 10, model_name=None)
        assert cli_brain.main(["usage", "--db", str(db_path)]) == 0
        out = capsys.readouterr().out
        assert "    unknown" in out and "est. n/a" in out
        assert "  estimated spend: n/a (no price in brain.prices_per_mtok for unknown)" in out

    def test_a_dated_snapshot_of_a_configured_alias_is_priced(self, db_path, conn, config, capsys):
        """The rows name the model that ANSWERED; a dated snapshot of a configured alias is
        priced at the alias's placeholder rather than making the spend n/a."""
        alias = config.brain.stages.thesis.model
        price = config.brain.prices_per_mtok[alias]
        _seed(conn, "c1", 1000, 100, model_name=f"{alias}-20260301", stage=ReasoningStage.THESIS)
        assert cli_brain.main(["usage", "--db", str(db_path)]) == 0
        lines = capsys.readouterr().out.splitlines()
        cost = f"${(1000 * price.input + 100 * price.output) / 1_000_000:.4f}"
        assert any(line.startswith(f"    {alias}-20260301") and line.endswith(f"est. {cost}") for line in lines)
        assert f"  estimated spend: {cost} ({cli_brain.ESTIMATE_NOTE})" in lines

    def test_once_and_usage_agree_when_the_api_answers_with_snapshot_ids(self, db_path, snapshot, config, capsys):
        llm = _SnapshotIds(_fake(tokens_in=1000, tokens_out=100))
        assert cli_brain.main(["once", "--db", str(db_path)], llm=llm, snapshot_builder=_builder(snapshot)) == 0
        capsys.readouterr()
        conn = open_store(db_path)
        try:
            models = {row.model_name for row in get_cycle_reasoning(conn, llm.inner.calls[0].cycle_id)}
            cost = _event(conn, "cycle_end").payload["estimated_cost_usd"]
        finally:
            conn.close()
        assert len(models) == 3 and any(model not in config.brain.prices_per_mtok for model in models)
        assert cost is not None and cost > 0  # once priced the requested (configured) models
        assert cli_brain.main(["usage", "--db", str(db_path)]) == 0
        lines = capsys.readouterr().out.splitlines()
        assert not any("n/a" in line for line in lines)
        (spend,) = [line for line in lines if line.startswith("  estimated spend: ")]
        assert spend.endswith(f"({cli_brain.ESTIMATE_NOTE})")
        assert float(spend.split("$")[1].split()[0]) == pytest.approx(cost, abs=5e-5)

    def test_empty_day_and_day_flag(self, db_path, conn, capsys):
        _seed(conn, "c1", 700, 30, created_at=datetime(2026, 7, 30, 14, 0, tzinfo=timezone.utc))
        assert cli_brain.main(["usage", "--db", str(db_path)]) == 0
        out = capsys.readouterr().out
        assert "in 0 / out 0 = 0 over 0 reasoning row(s)" in out and "    (none)" in out
        assert "estimated spend: $0.0000 (ESTIMATE" in out
        assert cli_brain.main(["usage", "--db", str(db_path), "--day", "2026-07-30"]) == 0
        out = capsys.readouterr().out
        assert out.splitlines()[0] == "token usage for 2026-07-30 (UTC)"
        assert "in 700 / out 30 = 730 over 1 reasoning row(s)" in out
        assert cli_brain.main(["usage", "--db", str(db_path), "--day", "30/07/2026"]) == 1
        assert capsys.readouterr().err == (
            "python -m aegis.cli.brain usage: error: --day must be YYYY-MM-DD, got '30/07/2026' (see --help)\n"
        )

    def test_rows_are_not_reported_as_api_calls(self, db_path, snapshot, config, capsys):
        """One reasoning row per stage, however many billed attempts its retries took: the
        count is labelled as rows, so six API calls never read as three."""
        llm = FakeLLM.from_fixtures({**OK, "proposal": ["malformed.json"] * 3 + ["proposal_ok.json"]})
        assert cli_brain.main(["once", "--db", str(db_path)], llm=llm, snapshot_builder=_builder(snapshot)) == 0
        assert len(llm.calls) == 6
        capsys.readouterr()
        assert cli_brain.main(["usage", "--db", str(db_path)]) == 0
        out = capsys.readouterr().out
        assert "= 900 over 3 reasoning row(s)" in out and "call" not in out

    def test_an_ascii_stdout_still_prints_the_whole_report(self, db_path, conn, monkeypatch):
        _seed(conn, "c1", 1000, 200, model_name="claude-haiku-4-5-20251001")
        buffer = io.BytesIO()
        stdout = io.TextIOWrapper(buffer, encoding="ascii")
        monkeypatch.setattr(sys, "stdout", stdout)
        assert cli_brain.main(["usage", "--db", str(db_path)]) == 0
        stdout.flush()
        lines = buffer.getvalue().decode("ascii").splitlines()
        assert "  estimated spend: $0.0020 (ESTIMATE from config brain.prices_per_mtok placeholders \\u2014 not a bill)" in lines
        assert lines[-1].startswith("  daily budget: 1,200 / ")

    def test_never_creates_or_migrates_a_database(self, tmp_path, capsys):
        missing = tmp_path / "missing.db"
        assert cli_brain.main(["usage", "--db", str(missing)]) == 1
        captured = capsys.readouterr()
        assert captured.out == "" and "does not exist" in captured.err
        assert not missing.exists()
        assert cli_brain.main(["usage", "--db", ":memory:"]) == 1
        assert "in-memory" in capsys.readouterr().err
