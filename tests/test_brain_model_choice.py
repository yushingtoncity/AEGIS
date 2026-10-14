"""Model selection: each tracked stage calls its family's newest model, and
falls back to the pinned ``model`` whenever the list cannot say. No network:
the Models API list is a canned tuple or a fake SDK client."""

from datetime import datetime, timezone
from types import SimpleNamespace

import anthropic
import httpx2
import pytest

from aegis.brain.cycle import run_cycle
from aegis.brain.errors import BrainError
from aegis.brain.llm import AnthropicLLM, ListedModel
from aegis.brain.model_choice import apply_choices, choose_models, latest_in_family, resolve_models
from aegis.brain.testing import FIXTURES_DIR, FakeLLM
from aegis.brain.models import MarketSnapshot
from aegis.config import BrainConfig, get_config
from aegis.store.db import open_store
from aegis.store.models import EventLevel
from aegis.store.repo import get_recent_events


def get_events(conn):
    return list(reversed(get_recent_events(conn, limit=100)))


def _at(day: int) -> datetime:
    return datetime(2026, 1, day, tzinfo=timezone.utc)


LISTED = (
    ListedModel(id="claude-haiku-4-5-20251001", created_at=_at(1)),
    ListedModel(id="claude-haiku-4-5", created_at=_at(1)),
    ListedModel(id="claude-haiku-5-5", created_at=_at(20)),
    ListedModel(id="claude-opus-5", created_at=_at(2)),
    ListedModel(id="claude-opus-5-5", created_at=_at(15)),
    ListedModel(id="claude-opus-4-8", created_at=_at(25)),  # newer release, older version
    ListedModel(id="claude-fable-5", created_at=_at(3)),
    ListedModel(id="claude-fable-5-1", created_at=_at(16)),
    ListedModel(id="claude-mythos-5-1", created_at=_at(28)),
    ListedModel(id="claude-sonnet-5-5", created_at=_at(14)),
)


def _tracked(**overrides) -> BrainConfig:
    stages = {
        "scan": {"model": "claude-haiku-5-5", "max_tokens": 4000, "effort": "low", "track": "haiku"},
        "thesis": {"model": "claude-opus-5-5", "max_tokens": 4000, "effort": "medium", "track": "opus"},
        "proposal": {"model": "claude-fable-5-1", "max_tokens": 3000, "effort": "low", "track": "fable"},
    }
    for stage, update in overrides.items():
        stages[stage] = {**stages[stage], **update}
    return BrainConfig.model_validate({"stages": stages})


class TestLatestInFamily:
    def test_highest_version_wins_over_a_later_release_of_an_older_version(self):
        assert latest_in_family(LISTED, "opus") == "claude-opus-5-5"

    @pytest.mark.parametrize(
        ("family", "expected"),
        [("haiku", "claude-haiku-5-5"), ("fable", "claude-fable-5-1"), ("sonnet", "claude-sonnet-5-5")],
    )
    def test_each_family(self, family, expected):
        assert latest_in_family(LISTED, family) == expected

    def test_a_new_major_version_is_picked_up(self):
        listed = LISTED + (ListedModel(id="claude-opus-6", created_at=_at(30)),)
        assert latest_in_family(listed, "opus") == "claude-opus-6"

    def test_the_alias_is_preferred_over_its_dated_snapshot(self):
        listed = (
            ListedModel(id="claude-opus-6-20270101", created_at=_at(5)),
            ListedModel(id="claude-opus-6", created_at=_at(5)),
        )
        assert latest_in_family(listed, "opus") == "claude-opus-6"

    def test_a_dated_snapshot_alone_is_still_usable(self):
        listed = (ListedModel(id="claude-haiku-4-5-20251001", created_at=_at(1)),)
        assert latest_in_family(listed, "haiku") == "claude-haiku-4-5-20251001"

    @pytest.mark.parametrize("model_id", ["claude-opus-5-5-preview", "claude-opus-latest", "opus-5-5", "claude-opus-5-555"])
    def test_ids_outside_the_naming_scheme_are_never_picked(self, model_id):
        assert latest_in_family((ListedModel(id=model_id, created_at=_at(9)),), "opus") is None

    def test_none_when_the_family_is_absent(self):
        assert latest_in_family(LISTED[:2], "fable") is None


class TestChooseModels:
    def test_tracked_stages_get_the_latest(self):
        listed = LISTED + (ListedModel(id="claude-fable-6", created_at=_at(31)),)
        choices = {c.stage: c for c in choose_models(_tracked(), listed)}
        assert choices["proposal"].model == "claude-fable-6"
        assert choices["proposal"].source == "latest"
        assert choices["proposal"].pinned == "claude-fable-5-1"
        assert choices["scan"].model == "claude-haiku-5-5"

    def test_an_untracked_stage_stays_pinned_even_when_newer_exists(self):
        brain = _tracked(thesis={"track": None, "model": "claude-opus-5"})
        thesis = next(c for c in choose_models(brain, LISTED) if c.stage == "thesis")
        assert (thesis.model, thesis.source) == ("claude-opus-5", "pinned")

    def test_no_list_falls_back_to_the_pins_with_the_reason(self):
        choices = choose_models(_tracked(), None, unavailable="APIConnectionError: down")
        assert {c.source for c in choices} == {"fallback"}
        assert [c.model for c in choices] == ["claude-haiku-5-5", "claude-opus-5-5", "claude-fable-5-1"]
        assert all(c.note == "APIConnectionError: down" for c in choices)

    def test_a_family_missing_from_the_list_falls_back(self):
        listed = tuple(m for m in LISTED if "fable" not in m.id)
        proposal = next(c for c in choose_models(_tracked(), listed) if c.stage == "proposal")
        assert (proposal.model, proposal.source, proposal.note) == (
            "claude-fable-5-1", "fallback", "no fable model in the list"
        )

    def test_apply_changes_only_the_model(self):
        brain = _tracked()
        listed = LISTED + (ListedModel(id="claude-opus-6", created_at=_at(30)),)
        applied = apply_choices(brain, choose_models(brain, listed))
        assert applied.stages.thesis.model == "claude-opus-6"
        assert applied.stages.thesis.max_tokens == 4000
        assert applied.stages.thesis.effort == "medium"
        assert brain.stages.thesis.model == "claude-opus-5-5"  # the original is untouched
        assert applied.per_cycle_token_cap == brain.per_cycle_token_cap


class _Lister:
    def __init__(self, outcome):
        self.outcome = outcome
        self.calls = 0

    def list_models(self):
        self.calls += 1
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return self.outcome


class TestResolveModels:
    def test_reads_the_list_from_the_client(self):
        lister = _Lister(LISTED)
        assert [c.source for c in resolve_models(lister, _tracked())] == ["latest"] * 3
        assert lister.calls == 1

    def test_no_call_when_nothing_is_tracked(self):
        lister = _Lister(LISTED)
        brain = _tracked(**{s: {"track": None} for s in ("scan", "thesis", "proposal")})
        assert [c.source for c in resolve_models(lister, brain)] == ["pinned"] * 3
        assert lister.calls == 0

    def test_a_client_without_a_list_falls_back(self):
        choices = resolve_models(FakeLLM(), _tracked())
        assert {(c.source, c.note) for c in choices} == {("fallback", "client cannot list models")}

    def test_a_failing_list_falls_back_and_names_the_error(self):
        choices = resolve_models(_Lister(BrainError("model list")), _tracked())
        assert {c.source for c in choices} == {"fallback"}
        assert "BrainError" in choices[0].note


class _Models:
    def __init__(self, outcome):
        self.outcome = outcome

    def list(self):
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return iter(self.outcome)


class TestAnthropicListModels:
    def test_reads_ids_and_release_dates(self):
        client = SimpleNamespace(models=_Models([SimpleNamespace(id="claude-opus-5-5", created_at=_at(15))]))
        listed = AnthropicLLM(client=client).list_models()
        assert listed == (ListedModel(id="claude-opus-5-5", created_at=_at(15)),)

    def test_an_api_error_is_a_brain_error(self):
        request = httpx2.Request("GET", "https://api.anthropic.com/v1/models")
        client = SimpleNamespace(models=_Models(anthropic.APIConnectionError(request=request)))
        with pytest.raises(BrainError, match="model list"):
            AnthropicLLM(client=client).list_models()


class _ListingFake(FakeLLM):
    def list_models(self):
        return LISTED + (ListedModel(id="claude-opus-6", created_at=_at(30)),)


OK = {"scan": ["scan_ok.json"], "thesis": ["thesis_ok.json"], "proposal": ["proposal_ok.json"]}


class TestCycleUsesTheChoice:
    @pytest.fixture
    def snapshot(self):
        return MarketSnapshot.model_validate_json((FIXTURES_DIR / "market_snapshot.json").read_text())

    def test_the_cycle_calls_the_resolved_models_and_journals_them(self, tmp_path, snapshot):
        config = get_config()
        llm = _ListingFake.from_fixtures(OK)
        conn = open_store(tmp_path / "brain.db")
        try:
            run_cycle(conn, llm, config=config, snapshot=snapshot)
            events = [e for e in get_events(conn) if e.kind == "model_selection"]
        finally:
            conn.close()
        assert [request.model for request in llm.calls][:2] == ["claude-haiku-5-5", "claude-opus-6"]
        assert len(events) == 1 and events[0].level is EventLevel.INFO
        stages = {s["stage"]: s for s in events[0].payload["stages"]}
        assert stages["thesis"]["model"] == "claude-opus-6"
        assert stages["thesis"]["pinned"] == "claude-opus-5-5"

    def test_a_fallback_is_a_warning(self, tmp_path, snapshot):
        conn = open_store(tmp_path / "brain.db")
        try:
            run_cycle(conn, FakeLLM.from_fixtures(OK), config=get_config(), snapshot=snapshot)
            event = next(e for e in get_events(conn) if e.kind == "model_selection")
        finally:
            conn.close()
        assert event.level is EventLevel.WARNING


class TestConfig:
    def test_shipped_config_tracks_haiku_opus_and_fable(self):
        stages = get_config().brain.stages
        assert (stages.scan.track, stages.thesis.track, stages.proposal.track) == ("haiku", "opus", "fable")

    def test_an_unknown_family_is_rejected(self):
        with pytest.raises(ValueError, match="track must be one of"):
            _tracked(scan={"track": "gpt"})
