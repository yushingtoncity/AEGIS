"""Which expiration the brain reads a chain from — canned fixtures and fakes, no network.

Each symbol's chain is the nearest listed expiration whose DTE is at least
``risk_limits.min_dte`` and at most ``brain.snapshot.max_dte``. The pure
helpers (``whole_days_to_expiry``, ``in_dte_window``,
``eligible_expiration``) are pinned first, then held against the policy
engine's own ``options_min_dte``. ``build_market_snapshot`` then runs over
the REAL ``aegis.data.market.get_option_chain`` with only its lowest layer
faked — the expiration listing, the contract listing and the options data
client — so the brain's choice and the data layer's fetch are tested
together, and nothing reaches Alpaca.
"""

import math
import random
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest
from policy_factories import exchange_date, long_call, make_context, make_limits

import aegis.brain.snapshot as snapshot_module
import aegis.data.market as market
from aegis.brain.cycle import run_cycle
from aegis.brain.models import ScanBrief, ThesisOutput
from aegis.brain.prompts import build_thesis_prompt
from aegis.brain.schemas import OutputInvalid
from aegis.brain.snapshot import (
    NO_ELIGIBLE_EXPIRATION,
    OPTIONS_FEED_NOTE,
    build_market_snapshot,
    eligible_expiration,
    in_dte_window,
    whole_days_to_expiry,
)
from aegis.brain.testing import FakeLLM
from aegis.brain.thesis import validate_candidates
from aegis.config import AegisConfig, RiskLimits
from aegis.data.cache import default_cache
from aegis.data.errors import DataError, NoEligibleExpiration
from aegis.data.models import AccountState, Bar, MarketClock, NewsItem, Quote
from aegis.policy import rules
from aegis.policy.models import RuleOutcome
from aegis.pricing.time_to_expiry import calendar_days_to_expiry
from aegis.store.db import open_store

UTC = timezone.utc
NEW_YORK = ZoneInfo("America/New_York")
NOW = datetime(2026, 7, 30, 15, 0, tzinfo=UTC)
"""Thursday 2026-07-30, 11:00 in New York: the market is open."""
TODAY = date(2026, 7, 30)
OPEN_CLOCK = MarketClock(
    is_open=True,
    next_open=datetime(2026, 7, 31, 13, 30, tzinfo=UTC),
    next_close=datetime(2026, 7, 30, 20, 0, tzinfo=UTC),
)
STRIKES = (630.0, 635.0, 640.0, 645.0, 650.0)
WINDOW = "7-45 DTE, risk_limits.min_dte, brain.snapshot.max_dte"


def days(n: int) -> date:
    """The expiration ``n`` calendar days after the trading date."""
    return TODAY + timedelta(days=n)


def occ(symbol: str, expiration: date, kind: str, strike: float) -> str:
    return f"{symbol}{expiration:%y%m%d}{kind}{int(strike * 1000):08d}"


def config(**blocks) -> AegisConfig:
    """A two-symbol config; ``blocks`` are top-level config blocks."""
    return AegisConfig.model_validate({"watchlist": ["SPY", "QQQ"], **blocks})


# --- the pure helpers -----------------------------------------------------------


class TestWholeDaysToExpiry:
    def test_it_is_the_pricing_engines_calendar_days_rounded_down(self):
        for n in (0, 1, 7, 8, 45):
            fractional = calendar_days_to_expiry(days(n), NOW)
            assert whole_days_to_expiry(days(n), NOW) == math.floor(fractional)
        # 11:00 in New York: five hours to the close on top of the whole days
        assert calendar_days_to_expiry(days(8), NOW) == pytest.approx(8 + 5 / 24)
        assert whole_days_to_expiry(days(8), NOW) == 8

    def test_after_the_close_a_day_is_one_less(self):
        evening = datetime(2026, 7, 30, 22, 0, tzinfo=UTC)  # 18:00 in New York
        assert whole_days_to_expiry(days(8), evening) == 7
        assert whole_days_to_expiry(days(0), evening) == 0  # expired: clamped at zero
        assert whole_days_to_expiry(days(-3), NOW) == 0

    def test_the_expiry_instant_follows_the_config(self):
        # With a 10:00 expiry, 11:00 is past it: a day less.
        assert whole_days_to_expiry(days(8), NOW, expiry_time="10:00") == 7
        tokyo = whole_days_to_expiry(days(8), NOW, expiry_timezone="Asia/Tokyo")
        assert tokyo == 7  # 16:00 in Tokyo on the 7th is 07:00 UTC: 7 days and 16 hours out

    def test_a_naive_now_is_utc(self):
        assert whole_days_to_expiry(days(8), NOW.replace(tzinfo=None)) == 8

    def test_it_is_never_more_than_the_policy_engines_dte(self):
        """Every two hours of 2026 (both daylight-saving changes included):
        never above the policy's count of calendar days from the trading
        date, and equal to it from the open until an hour before the close."""
        instant = datetime(2026, 1, 1, tzinfo=UTC)
        checked = equal_in_session = 0
        while instant.year == 2026:
            trading_date = instant.astimezone(NEW_YORK).date()
            local = instant.astimezone(NEW_YORK).time()
            in_session = (9, 30) <= (local.hour, local.minute) < (15, 0)
            for n in (0, 1, 6, 7, 8, 30, 45, 46):
                expiration = trading_date + timedelta(days=n)
                whole = whole_days_to_expiry(expiration, instant)
                assert whole <= n, (instant, n, whole)
                if in_session:
                    assert whole == n, (instant, n, whole)
                    equal_in_session += 1
                checked += 1
            instant += timedelta(hours=2)
        assert checked >= 30_000 and equal_in_session >= 5_000


class TestEligibleExpiration:
    def test_the_nearest_expiration_at_or_above_the_floor_is_picked(self):
        listed = [days(n) for n in (0, 1, 3, 8, 15)]
        assert eligible_expiration(listed, now=NOW, min_dte=7, max_dte=45) == days(8)

    def test_nothing_between_the_floor_and_the_cap_is_none(self):
        listed = [days(n) for n in (0, 1, 3, 46, 60)]
        assert eligible_expiration(listed, now=NOW, min_dte=7, max_dte=45) is None
        assert eligible_expiration([days(6), days(46)], now=NOW, min_dte=7, max_dte=45) is None
        assert eligible_expiration([], now=NOW, min_dte=7, max_dte=45) is None

    def test_both_ends_of_the_window_are_included(self):
        assert eligible_expiration([days(7)], now=NOW, min_dte=7, max_dte=45) == days(7)
        assert eligible_expiration([days(45)], now=NOW, min_dte=7, max_dte=45) == days(45)
        assert in_dte_window(days(6), now=NOW, min_dte=7, max_dte=45) is False
        assert in_dte_window(days(46), now=NOW, min_dte=7, max_dte=45) is False

    def test_an_expired_contract_is_never_eligible_even_at_a_zero_floor(self):
        evening = datetime(2026, 7, 30, 22, 0, tzinfo=UTC)  # 18:00 in New York
        assert in_dte_window(days(0), now=NOW, min_dte=0, max_dte=45) is True  # 5 h to the close
        assert in_dte_window(days(0), now=evening, min_dte=0, max_dte=45) is False  # it closed
        assert in_dte_window(days(-1), now=NOW, min_dte=0, max_dte=45) is False
        listed = [days(-1), days(0), days(1)]
        assert eligible_expiration(listed, now=evening, min_dte=0, max_dte=45) == days(1)

    def test_the_listing_order_and_repeats_do_not_matter(self):
        listed = [days(15), days(8), days(1), days(8), days(0)]
        assert eligible_expiration(listed, now=NOW, min_dte=7, max_dte=45) == days(8)

    def test_the_floor_and_cap_are_whatever_is_passed(self):
        listed = [days(n) for n in (0, 1, 3, 8, 15)]
        assert eligible_expiration(listed, now=NOW, min_dte=0, max_dte=45) == days(0)
        assert eligible_expiration(listed, now=NOW, min_dte=2, max_dte=45) == days(3)
        assert eligible_expiration(listed, now=NOW, min_dte=9, max_dte=45) == days(15)
        assert eligible_expiration(listed, now=NOW, min_dte=7, max_dte=7) is None
        assert eligible_expiration(listed, now=NOW, min_dte=9, max_dte=14) is None

    def test_the_brains_pick_always_clears_the_policys_options_min_dte(self):
        """The two share one floor; the brain's count never exceeds the
        policy's, so whatever it picks, ``options_min_dte`` passes."""
        rng = random.Random(20261002)
        picked = 0
        for _ in range(2_000):
            instant = datetime(2026, 1, 1, tzinfo=UTC) + timedelta(minutes=rng.randrange(525_600))
            trading_date = exchange_date(instant)
            listed = [trading_date + timedelta(days=rng.randrange(-2, 70)) for _ in range(6)]
            floor = rng.choice((0, 1, 2, 7, 10, 30))
            choice = eligible_expiration(listed, now=instant, min_dte=floor, max_dte=45)
            if choice is None:
                continue
            picked += 1
            proposal = long_call(expiration=choice)
            context = make_context(
                now=instant, market_date=trading_date, limits=make_limits(min_dte=floor)
            )
            result = rules.options_min_dte(proposal, context)
            assert result.outcome is RuleOutcome.PASS, (instant, choice, floor, result.detail)
        assert picked >= 1_000


# --- the data layer's seam: get_option_chain(eligible=...) ----------------------


class FakeOptionClient:
    """The options data client: a chain snapshot per requested expiration,
    built from the canned full-Greeks fixture; no volume bars."""

    def __init__(self, payload: dict):
        self.payload = payload
        self.chain_requests: list[tuple[str, date]] = []

    def get_option_chain(self, request):
        expiration = date.fromisoformat(str(request.expiration_date))
        symbol = request.underlying_symbol
        self.chain_requests.append((symbol, expiration))
        return {
            occ(symbol, expiration, kind, strike): self.payload
            for strike in STRIKES
            for kind in ("C", "P")
        }

    def get_option_bars(self, request):
        return SimpleNamespace(data={})


@pytest.fixture
def layer(monkeypatch, fixture):
    """The data layer below ``get_option_chain`` faked: ``layer.listed`` maps
    a symbol to its active expirations; the chain fetch, the contract
    listing and the underlying quote are canned. The cache is cleared on the
    way in and out."""
    data = fixture("stock_quote.json")
    quote = Quote.from_alpaca("SPY", quote=data["quote"], trade=data["trade"])
    client = FakeOptionClient(fixture("option_snapshot_full.json"))
    state = SimpleNamespace(listed={}, client=client, listings=[])

    def list_expirations(underlying):
        state.listings.append(underlying)
        listed = state.listed.get(underlying)
        if isinstance(listed, Exception):
            raise listed
        return list(listed or [])  # in whatever order the test gave it

    monkeypatch.setattr(market, "list_expirations", list_expirations)
    monkeypatch.setattr(market, "_fetch_contracts", lambda underlying, expiration: [])
    monkeypatch.setattr(market, "option_client", lambda: client)
    monkeypatch.setattr(
        market, "get_spot", lambda symbol: quote.model_copy(update={"symbol": symbol})
    )
    default_cache.clear()
    yield state
    default_cache.clear()


def window(min_dte=7, max_dte=45):
    return lambda expiration: in_dte_window(
        expiration, now=NOW, min_dte=min_dte, max_dte=max_dte
    )


class TestGetOptionChainEligible:
    def test_the_nearest_accepted_expiration_is_the_one_fetched(self, layer):
        layer.listed["SPY"] = [days(n) for n in (0, 1, 3, 8, 15)]
        chain = market.get_option_chain("SPY", eligible=window())
        assert chain.expiration == days(8)
        assert {c.expiration for c in chain.contracts} == {days(8)}
        assert layer.client.chain_requests == [("SPY", days(8))]  # no nearer chain was read

    def test_none_accepted_raises_and_fetches_no_chain(self, layer):
        listed = [days(n) for n in (0, 1, 3, 46, 60)]
        layer.listed["SPY"] = listed
        with pytest.raises(NoEligibleExpiration) as caught:
            market.get_option_chain("SPY", eligible=window())
        assert isinstance(caught.value, DataError)
        assert caught.value.listed == tuple(listed)
        assert str(caught.value) == "failed to fetch option chain (no eligible expiration) for SPY"
        assert layer.client.chain_requests == []

    def test_the_listing_order_and_repeats_do_not_matter(self, layer):
        layer.listed["SPY"] = [days(15), days(8), days(1), days(8), days(0)]
        assert market.get_option_chain("SPY", eligible=window()).expiration == days(8)

    def test_the_live_pick_is_eligible_expiration_over_the_listing(self, layer):
        """The data layer applies the brain's test nearest-first: what it
        fetches is what ``eligible_expiration`` picks from the same listing."""
        rng = random.Random(20261003)
        checked = 0
        for _ in range(300):
            default_cache.clear()
            listed = [days(rng.randrange(-2, 70)) for _ in range(rng.randrange(0, 7))]
            floor, cap = rng.choice((0, 1, 7, 10)), rng.choice((7, 20, 45))
            layer.listed["SPY"] = listed
            expected = eligible_expiration(listed, now=NOW, min_dte=floor, max_dte=cap)
            try:
                got = market.get_option_chain("SPY", eligible=window(floor, cap)).expiration
            except NoEligibleExpiration:
                got = None
            assert got == expected, (listed, floor, cap)
            checked += got is not None
        assert checked >= 100

    def test_without_eligible_the_nearest_is_fetched_as_before(self, layer):
        layer.listed["SPY"] = [days(n) for n in (0, 1, 3, 8, 15)]
        assert market.get_option_chain("SPY").expiration == days(0)

    def test_an_explicit_expiration_is_not_second_guessed(self, layer):
        layer.listed["SPY"] = [days(n) for n in (0, 8)]
        assert market.get_option_chain("SPY", days(0), eligible=window()).expiration == days(0)
        assert layer.listings == []  # the listing is not even read

    def test_the_chain_is_cached_by_the_expiration_chosen(self, layer):
        layer.listed["SPY"] = [days(n) for n in (1, 8, 15)]
        first = market.get_option_chain("SPY", eligible=window())
        assert market.get_option_chain("SPY", days(8)) is first
        later = market.get_option_chain("SPY", eligible=window(min_dte=9))
        assert later.expiration == days(15)
        assert layer.client.chain_requests == [("SPY", days(8)), ("SPY", days(15))]

    def test_a_listing_failure_is_a_data_error(self, layer):
        layer.listed["SPY"] = DataError("option contracts", "SPY")
        with pytest.raises(DataError, match="option contracts"):
            market.get_option_chain("SPY", eligible=window())


# --- the snapshot, end to end over the faked layer ------------------------------


@pytest.fixture
def feed(monkeypatch, fixture, layer):
    """The snapshot module's own fetchers canned, its chain fetcher the real one."""
    data = fixture("stock_quote.json")
    account = fixture("account.json")
    canned = {
        "get_market_clock": lambda: OPEN_CLOCK,
        "get_account_state": lambda: AccountState.from_alpaca(
            account["account"], account["positions"]
        ),
        "get_spot": lambda symbol: Quote.from_alpaca(
            symbol, quote=data["quote"], trade=data["trade"]
        ),
        "get_bars": lambda symbol, *a, **k: [
            Bar.from_alpaca(symbol, bar) for bar in fixture("bars.json")["bars"]
        ],
        "get_news": lambda symbol, *a, **k: [
            NewsItem.from_alpaca(item) for item in fixture("news.json")["news"]
        ],
    }
    for name, fake in canned.items():
        monkeypatch.setattr(snapshot_module, name, fake)
    assert snapshot_module.get_option_chain is market.get_option_chain
    return layer


def no_eligible(listed: int) -> str:
    return (
        f"{NO_ELIGIBLE_EXPIRATION} ({WINDOW}): none of the {listed} listed expiration(s) is "
        "in the window; option chain omitted"
    )


class TestSnapshotExpiration:
    def test_the_chain_comes_from_the_nearest_eligible_expiration(self, feed):
        feed.listed = {"SPY": [days(n) for n in (0, 1, 3, 8, 15)], "QQQ": [days(n) for n in (2, 9)]}
        snapshot = build_market_snapshot(config=config(), now=NOW)
        spy, qqq = snapshot.symbols
        assert spy.chain.expiration == days(8) and qqq.chain.expiration == days(9)
        assert spy.chain.days_to_expiry == pytest.approx(8 + 5 / 24)
        assert {c.expiration for c in spy.chain.contracts} == {days(8)}
        assert spy.errors == () and qqq.errors == ()
        assert snapshot.data_age.warnings == (OPTIONS_FEED_NOTE,)
        assert feed.client.chain_requests == [("SPY", days(8)), ("QQQ", days(9))]

    def test_the_floor_is_risk_limits_min_dte(self, feed):
        feed.listed = {"SPY": [days(n) for n in (0, 1, 3, 8, 15)]}
        for floor, expected in ((0, days(0)), (2, days(3)), (7, days(8)), (9, days(15))):
            default_cache.clear()
            cfg = config(watchlist=["SPY"], risk_limits={"min_dte": floor})
            assert cfg.risk_limits.min_dte == floor
            spy = build_market_snapshot(config=cfg, now=NOW).symbols[0]
            assert spy.chain.expiration == expected, floor

    def test_the_shipped_floor_is_the_policy_engines(self, feed):
        assert RiskLimits().min_dte == 7  # what options_min_dte enforces by default
        feed.listed = {"SPY": [days(n) for n in (6, 7)]}
        spy = build_market_snapshot(["SPY"], config=config(), now=NOW).symbols[0]
        assert spy.chain.expiration == days(7)

    def test_the_cap_is_brain_snapshot_max_dte(self, feed):
        feed.listed = {"SPY": [days(n) for n in (1, 20, 30)]}
        capped = config(watchlist=["SPY"], brain={"snapshot": {"max_dte": 25}})
        assert build_market_snapshot(config=capped, now=NOW).symbols[0].chain.expiration == days(20)
        default_cache.clear()
        tight = config(watchlist=["SPY"], brain={"snapshot": {"max_dte": 15}})
        spy = build_market_snapshot(config=tight, now=NOW).symbols[0]
        assert spy.chain is None
        assert spy.errors == (no_eligible(3).replace("7-45", "7-15"),)

    def test_no_eligible_expiration_is_a_warning_and_the_snapshot_goes_on(self, feed):
        feed.listed = {
            "SPY": [days(n) for n in (0, 1, 3, 46, 60)], "QQQ": [days(n) for n in (1, 8)]
        }
        snapshot = build_market_snapshot(config=config(), now=NOW)
        spy, qqq = snapshot.symbols
        assert spy.chain is None and spy.errors == (no_eligible(5),)
        assert spy.spot is not None and spy.headlines  # the rest of the symbol is there
        assert qqq.chain is not None and qqq.chain.expiration == days(8)
        warnings = snapshot.data_age.warnings
        assert f"SPY: {no_eligible(5)}" in warnings
        assert "SPY: option chain missing" not in warnings  # the line above says why
        assert OPTIONS_FEED_NOTE in warnings  # QQQ has a chain
        assert feed.client.chain_requests == [("QQQ", days(8))]  # no nearer SPY chain was read

    def test_no_chain_anywhere_drops_the_feed_note(self, feed):
        feed.listed = {"SPY": [days(1)], "QQQ": []}
        snapshot = build_market_snapshot(config=config(), now=NOW)
        assert [s.chain for s in snapshot.symbols] == [None, None]
        assert OPTIONS_FEED_NOTE not in snapshot.data_age.warnings
        assert snapshot.symbols[1].errors == (no_eligible(0),)

    def test_a_chain_outside_the_window_is_refused_never_used(self, feed, monkeypatch):
        """Whatever the fetcher returns, a nearer (or later) chain does not
        stand in for an eligible one."""
        real = market.get_option_chain
        for served in (days(1), days(60)):
            default_cache.clear()
            asked = []

            def wrong(symbol, expiration=None, *, eligible=None, served=served):
                asked.append(eligible)
                return real(symbol, served)

            monkeypatch.setattr(snapshot_module, "get_option_chain", wrong)
            feed.listed = {"SPY": [days(1), days(8), days(60)]}
            spy = build_market_snapshot(["SPY"], config=config(), now=NOW).symbols[0]
            assert spy.chain is None
            whole = whole_days_to_expiry(served, NOW)
            assert spy.errors == (
                f"{NO_ELIGIBLE_EXPIRATION} ({WINDOW}): the chain fetched expires "
                f"{served.isoformat()}, at {whole} DTE; option chain omitted",
            )
            # the window was offered: it accepts 7 and 45 days out, nothing nearer or later
            (eligible,) = asked
            assert [eligible(days(n)) for n in (6, 7, 45, 46)] == [False, True, True, False]

    def test_a_contract_of_another_expiration_never_reaches_the_brain(self, feed, monkeypatch):
        """An eligible chain that carries a contract listed under another
        date: that contract is left out, so no candidate can resolve to it."""
        real = market.get_option_chain

        def mixed(symbol, expiration=None, *, eligible=None):
            chain = real(symbol, expiration, eligible=eligible)
            stray = real(symbol, days(1)).contracts[:2]  # a 1 DTE call and put
            return chain.model_copy(update={"contracts": [*chain.contracts, *stray]})

        monkeypatch.setattr(snapshot_module, "get_option_chain", mixed)
        feed.listed = {"SPY": [days(1), days(8)]}
        spy = build_market_snapshot(["SPY"], config=config(), now=NOW).symbols[0]
        assert spy.chain is not None and spy.chain.expiration == days(8)
        assert {c.expiration for c in spy.chain.contracts} == {days(8)}
        assert not any(days(1).strftime("%y%m%d") in c.symbol for c in spy.chain.contracts)
        assert spy.errors == (
            f"2 contract(s) in the {days(8).isoformat()} chain expire on another date or none: "
            "left out",
        )

    def test_other_fetch_failures_are_unchanged(self, feed):
        feed.listed = {"SPY": DataError("option contracts", "SPY")}
        spy = build_market_snapshot(["SPY"], config=config(), now=NOW).symbols[0]
        assert spy.chain is None
        assert spy.errors == ("failed to fetch option contracts for SPY",)

    def test_the_thesis_stage_sees_and_may_use_only_the_eligible_expiration(self, feed):
        feed.listed = {"SPY": [days(n) for n in (1, 8)], "QQQ": [days(1)]}
        snapshot = build_market_snapshot(config=config(), now=NOW)
        prompt = build_thesis_prompt(_brief(snapshot), snapshot, (), RiskLimits(), ())
        assert f"SPY: expiration {days(8).isoformat()}" in prompt
        assert days(1).isoformat() not in prompt
        assert "QQQ: no option chain in this snapshot — equity candidates only" in prompt

        def candidate(expiration: date) -> ThesisOutput:
            return ThesisOutput.model_validate(
                {
                    "candidates": [
                        {
                            "symbol": "SPY",
                            "direction": "bullish",
                            "instrument": "option",
                            "structure": "call_debit_spread",
                            "expiration": expiration.isoformat(),
                            "strikes": [640, 650],
                            "rationale": "SYNTHETIC TEST DATA.",
                            "confidence": 0.6,
                            "key_risk": "SYNTHETIC TEST DATA.",
                            "invalidation": "SYNTHETIC TEST DATA. A close below 630.",
                        }
                    ]
                }
            )

        validate_candidates(candidate(days(8)), snapshot)  # the eligible chain: accepted
        with pytest.raises(OutputInvalid, match="is not available for SPY"):
            validate_candidates(candidate(days(1)), snapshot)  # the nearer one: refused

    def test_a_cycle_runs_on_with_a_symbol_that_has_no_eligible_expiration(
        self, feed, tmp_path
    ):
        feed.listed = {"SPY": [days(1)], "QQQ": [days(8)]}
        conn = open_store(tmp_path / "brain.db")
        try:
            llm = FakeLLM.from_fixtures(
                {"scan": ["scan_ok.json"], "thesis": ["thesis_empty.json"]}
            )
            result = run_cycle(conn, llm, config=config(), now=NOW)  # the default builder
        finally:
            conn.close()
        assert result.outcome == "no_trade"
        assert f"SPY: {no_eligible(1)}" in result.brief.staleness_warnings


def _brief(snapshot):
    """A minimal scan brief for the thesis prompt: one line per symbol."""
    return ScanBrief.model_validate(
        {
            "symbols": [
                {"symbol": s.symbol, "summary": "SYNTHETIC TEST DATA."} for s in snapshot.symbols
            ],
            "staleness_warnings": list(snapshot.data_age.warnings),
        }
    )

