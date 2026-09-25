"""Market snapshot assembly from canned fixtures — no live API calls.

The pure helpers are driven straight from the data-layer fixtures (the same
ones tests/test_models.py parses); ``build_market_snapshot`` runs with the
module's fetchers monkeypatched, which is the seam the live path exposes on
purpose — nothing here reaches Alpaca.
"""

import json
import os
import subprocess
import sys
from datetime import date, datetime, timedelta, timezone

import pytest

import aegis.brain.snapshot as snapshot_module
import aegis.data.account
import aegis.data.market
import aegis.data.news
import aegis.pricing.enrich
from aegis.brain.errors import BrainError
from aegis.brain.models import ChainSummary, MarketSnapshot, SymbolSnapshot
from aegis.brain.snapshot import (
    CLOCK_UNKNOWN_WARNING,
    MARKET_CLOSED_WARNING,
    OPTIONS_FEED_NOTE,
    account_summary_from,
    build_market_snapshot,
    chain_summary_from,
    data_age_from,
    symbol_snapshot_from,
)
from aegis.config import REPO_ROOT, AegisConfig, ConfigError
from aegis.data.errors import DataError
from aegis.data.models import (
    AccountState,
    Bar,
    ChainSnapshot,
    MarketClock,
    NewsItem,
    OptionSnapshot,
    OptionType,
    Quote,
)
from aegis.pricing.enrich import enrich_chain
from aegis.pricing.errors import PricingError
from aegis.pricing.models import Greeks

# Fixture quotes are as of 2026-07-30 19:45-20:00 UTC with SPY around 638.9.
NOW = datetime(2026, 7, 30, 20, 0, tzinfo=timezone.utc)
SPOT = 638.9
RATE = 0.04
STALE_AFTER = 20 * 60  # the option fixtures are ~15 minutes old at NOW: fresh at 20
EXPIRATION = date(2026, 8, 21)
STRIKES = [620.0, 625.0, 630.0, 635.0, 640.0, 645.0, 650.0, 655.0, 660.0]
OPEN_CLOCK = MarketClock(
    is_open=True,
    next_open=datetime(2026, 7, 31, 13, 30, tzinfo=timezone.utc),
    next_close=datetime(2026, 7, 30, 20, 0, tzinfo=timezone.utc),
)
CLOSED_CLOCK = OPEN_CLOCK.model_copy(update={"is_open": False})


def _occ(strike: float, option_type: OptionType) -> str:
    return f"SPY260821{'C' if option_type is OptionType.CALL else 'P'}{int(strike * 1000):08d}"


@pytest.fixture
def quote(fixture):
    data = fixture("stock_quote.json")
    return Quote.from_alpaca("SPY", quote=data["quote"], trade=data["trade"])


@pytest.fixture
def bars(fixture):
    data = fixture("bars.json")
    return [Bar.from_alpaca(data["symbol"], b) for b in data["bars"]]


@pytest.fixture
def chain(fixture):
    """Nine strikes around 640. Calls carry the full fixture's vendor IV and
    Greeks; puts use the no-greeks fixture (vendor values absent) quoted at
    intrinsic + 4, so the model side has something to solve."""
    full = fixture("option_snapshot_full.json")
    no_greeks = fixture("option_snapshot_no_greeks.json")
    contracts = []
    for strike in STRIKES:
        contracts.append(OptionSnapshot.from_alpaca(_occ(strike, OptionType.CALL), "SPY", full))
        intrinsic = max(strike - SPOT, 0.0)
        payload = {
            **no_greeks,
            "symbol": _occ(strike, OptionType.PUT),
            "latest_quote": {
                **no_greeks["latest_quote"],
                "bid_price": intrinsic + 3.9,
                "ask_price": intrinsic + 4.1,
            },
        }
        contracts.append(OptionSnapshot.from_alpaca(_occ(strike, OptionType.PUT), "SPY", payload))
    return ChainSnapshot(underlying="SPY", expiration=EXPIRATION, spot=SPOT, contracts=contracts)


@pytest.fixture
def enriched(chain):
    return enrich_chain(chain, RATE, now=NOW)


@pytest.fixture
def news(fixture):
    return [NewsItem.from_alpaca(n) for n in fixture("news.json")["news"]]


@pytest.fixture
def account(fixture):
    data = fixture("account.json")
    return AccountState.from_alpaca(data["account"], data["positions"])


def _summary(chain, enriched, **overrides) -> ChainSummary:
    kwargs = dict(now=NOW, stale_after_seconds=STALE_AFTER, market_open=True, strikes_each_side=2)
    return chain_summary_from(chain, enriched, **{**kwargs, **overrides})


def _symbol(quote=None, bars=(), chain=None, enriched=(), news=(), **overrides) -> SymbolSnapshot:
    kwargs = dict(now=NOW, stale_after_seconds=STALE_AFTER, market_open=True, strikes_each_side=2)
    return symbol_snapshot_from(
        "SPY", quote, bars, chain, enriched, news, **{**kwargs, **overrides}
    )


class TestChainSummary:
    def test_picks_atm_plus_minus_n_strikes(self, chain, enriched):
        summary = _summary(chain, enriched)
        assert summary.underlying == "SPY" and summary.expiration == EXPIRATION
        assert summary.spot == SPOT and summary.atm_strike == 640.0
        assert [c.strike for c in summary.contracts] == [
            630.0, 630.0, 635.0, 635.0, 640.0, 640.0, 645.0, 645.0, 650.0, 650.0,
        ]
        # call before put at each strike, and each row is a real contract
        assert [c.option_type for c in summary.contracts[:2]] == [OptionType.CALL, OptionType.PUT]
        assert summary.contracts[4].symbol == "SPY260821C00640000"
        assert all(c.expiration == EXPIRATION for c in summary.contracts)
        assert len(_summary(chain, enriched, strikes_each_side=0).contracts) == 2
        assert len(_summary(chain, enriched, strikes_each_side=10).contracts) == 18

    def test_days_to_expiry_is_calendar_days_to_the_close(self, chain, enriched):
        # 16:00 ET on 2026-08-21 is 20:00 UTC: exactly 22 days after NOW.
        assert _summary(chain, enriched).days_to_expiry == pytest.approx(22.0)
        early = _summary(chain, enriched, expiry_time="13:00")
        assert early.days_to_expiry == pytest.approx(22.0 - 3 / 24)

    def test_greeks_and_iv_sources(self, chain, enriched):
        summary = _summary(chain, enriched)
        calls = [c for c in summary.contracts if c.option_type is OptionType.CALL]
        puts = [c for c in summary.contracts if c.option_type is OptionType.PUT]
        for call in calls:
            assert (call.iv_source, call.greeks_source) == ("vendor", "vendor")
            assert call.implied_vol == 0.1845
            assert call.greeks == Greeks(
                delta=0.5321, gamma=0.0123, theta=-0.0891, vega=0.7421, rho=0.21
            )
        for put in puts:
            assert (put.iv_source, put.greeks_source) == ("model", "model")
            assert put.implied_vol is not None and 0.0 < put.implied_vol < 1.0
            assert put.greeks is not None and put.greeks.delta < 0
        # The summary shows exactly what the enriched option resolved to.
        by_symbol = {e.snapshot.symbol: e for e in enriched}
        for contract in summary.contracts:
            assert contract.implied_vol == by_symbol[contract.symbol].implied_vol
            assert contract.greeks == by_symbol[contract.symbol].greeks

    def test_contract_quote_fields_and_age(self, chain, enriched):
        call = _summary(chain, enriched).contracts[4]  # the 640 call
        assert (call.bid, call.ask, call.mid, call.last) == (12.05, 12.35, pytest.approx(12.2), 12.2)
        assert call.volume is None and call.open_interest is None
        assert call.quote_age_seconds == 901.0  # 19:44:59 -> 20:00:00
        put = _summary(chain, enriched).contracts[5]
        assert put.quote_age_seconds == 900.0
        assert put.bid == pytest.approx(5.0) and put.ask == pytest.approx(5.2)

    def test_fresh_when_open_and_recent(self, chain, enriched):
        summary = _summary(chain, enriched)
        assert summary.stale is False
        assert summary.quote_age_seconds == 900.0  # the newest venue timestamp

    def test_stale_when_market_closed(self, chain, enriched):
        assert _summary(chain, enriched, market_open=False).stale is True

    def test_stale_when_clock_unknown(self, chain, enriched):
        assert _summary(chain, enriched, market_open=None).stale is True

    def test_stale_when_quotes_are_old(self, chain, enriched):
        later = NOW + timedelta(minutes=30)
        summary = _summary(chain, enriched, now=later)
        assert summary.stale is True
        assert summary.quote_age_seconds == 2700.0
        assert summary.contracts[4].quote_age_seconds == 2701.0
        # exactly at the threshold is still fresh; one second past is not
        assert _summary(chain, enriched, stale_after_seconds=900).stale is False
        assert _summary(chain, enriched, stale_after_seconds=899).stale is True

    def test_stale_when_no_quote_timestamps(self, chain, enriched):
        no_times = chain.model_copy(
            update={"contracts": [c.model_copy(update={"quote_time": None}) for c in chain.contracts]}
        )
        summary = _summary(no_times, enriched)
        assert summary.stale is True
        assert summary.quote_age_seconds is None
        assert all(c.quote_age_seconds is None for c in summary.contracts)

    def test_no_spot_and_no_enrichment_shows_vendor_values(self, chain):
        no_spot = chain.model_copy(update={"spot": None})
        summary = _summary(no_spot, [])
        assert summary.spot is None and summary.atm_strike is None
        # without a spot the lowest strikes are shown (ChainSnapshot.atm_strikes)
        assert [c.strike for c in summary.contracts][:2] == [620.0, 620.0]
        call, put = summary.contracts[0], summary.contracts[1]
        assert (call.iv_source, call.greeks_source) == ("vendor", "vendor")
        assert call.greeks is not None and call.greeks.delta == 0.5321
        assert (put.implied_vol, put.iv_source, put.greeks, put.greeks_source) == (None, None, None, None)

    def test_enriched_spot_fills_in_for_a_chain_without_one(self, chain, enriched):
        summary = _summary(chain.model_copy(update={"spot": None}), enriched)
        assert summary.spot == SPOT and summary.atm_strike == 640.0

    def test_contract_without_expiration_takes_the_chains(self, chain):
        contracts = [c.model_copy(update={"expiration": None}) for c in chain.contracts]
        bare = chain.model_copy(update={"contracts": contracts})
        summary = _summary(bare, enrich_chain(bare, RATE, now=NOW))
        assert all(c.expiration == EXPIRATION for c in summary.contracts)

    def test_unparseable_contracts_are_skipped(self, chain, enriched, fixture):
        weird = OptionSnapshot.from_alpaca("WEIRD", "SPY", fixture("option_snapshot_full.json"))
        with_weird = chain.model_copy(update={"contracts": [*chain.contracts, weird]})
        assert [c.symbol for c in _summary(with_weird, enriched).contracts] == [
            c.symbol for c in _summary(chain, enriched).contracts
        ]


class TestSymbolSnapshot:
    def test_quote_fields_from_fixture(self, quote):
        snap = _symbol(quote=quote)
        assert snap.symbol == "SPY"
        assert (snap.spot, snap.bid, snap.ask) == (636.42, 636.4, 636.44)
        assert snap.spot_time == quote.last_time
        assert snap.spot_age_seconds == 2.0
        assert snap.stale is False
        assert snap.chain is None and snap.headlines == () and snap.errors == ()

    def test_after_hours_quote_is_stale_when_closed(self, fixture):
        data = fixture("stock_quote_after_hours.json")
        after = Quote.from_alpaca("SPY", quote=data["quote"], trade=data["trade"])
        snap = _symbol(quote=after, market_open=False, now=NOW + timedelta(hours=4))
        assert snap.spot == 636.9 and snap.bid is None and snap.ask is None
        assert snap.stale is True

    def test_quote_staleness_rules(self, quote):
        assert _symbol(quote=quote, market_open=False).stale is True
        assert _symbol(quote=quote, market_open=None).stale is True
        assert _symbol(quote=quote, now=NOW + timedelta(minutes=30)).stale is True
        no_time = quote.model_copy(update={"last_time": None, "quote_time": None})
        snap = _symbol(quote=no_time)
        assert snap.stale is True and snap.spot_age_seconds is None and snap.spot == 636.42

    def test_missing_quote_is_stale_and_errors_are_kept(self):
        error = str(DataError("spot quote", "SPY"))
        snap = _symbol(quote=None, errors=[error])
        assert snap.spot is None and snap.spot_time is None and snap.spot_age_seconds is None
        assert snap.stale is True
        assert snap.errors == (error,)

    def test_fresh_quote_without_a_usable_price_is_stale(self, quote):
        """A quote stamped 5 s ago but with no trade and no two-sided quote (IEX's 0.0/0.0
        outside hours maps to None) has no spot: stale and warned about, however fresh."""
        priceless = Quote(symbol="SPY", quote_time=NOW - timedelta(seconds=5))
        assert priceless.spot is None and priceless.spot_time == NOW - timedelta(seconds=5)
        snap = _symbol(quote=priceless)
        assert snap.spot is None and snap.spot_age_seconds == 5.0
        assert snap.stale is True
        one_sided = quote.model_copy(update={"last": None, "ask": None})  # a bid alone is no price either
        assert one_sided.spot is None and _symbol(quote=one_sided).stale is True
        age = data_age_from(OPEN_CLOCK, [snap], stale_after_seconds=STALE_AFTER)
        assert age.any_stale is True
        assert age.warnings == ("SPY: spot price missing", "SPY: option chain missing")

    def test_symbol_is_normalised(self, quote):
        snap = symbol_snapshot_from(
            " spy ", quote, [], None, [], [],
            now=NOW, stale_after_seconds=STALE_AFTER, market_open=True, strikes_each_side=2,
        )
        assert snap.symbol == "SPY"

    def test_bars_summary_with_three_bars(self, bars):
        snap = _symbol(bars=bars)
        assert snap.last_close == 636.4
        assert snap.change_5d_pct is None  # needs six bars
        assert snap.high_20d is None and snap.low_20d is None  # needs twenty

    def test_bars_summary_numbers(self, bars):
        template = bars[0]
        many = [
            template.model_copy(
                update={
                    "timestamp": template.timestamp + timedelta(days=i),
                    "close": 600.0 + i,
                    "high": 602.0 + i,
                    "low": 598.0 + i,
                }
            )
            for i in range(25)
        ]
        snap = _symbol(bars=many)
        assert snap.last_close == 624.0
        assert snap.change_5d_pct == pytest.approx((624.0 / 619.0 - 1) * 100)
        assert snap.high_20d == 626.0  # bars 5..24
        assert snap.low_20d == 603.0
        # six bars is enough for the 5-day change, still not for the range
        six = _symbol(bars=many[:6])
        assert six.change_5d_pct == pytest.approx((605.0 / 600.0 - 1) * 100)
        assert six.high_20d is None
        # newest-last is established by timestamp, not by list order
        assert _symbol(bars=list(reversed(many))).last_close == 624.0

    def test_no_bars(self, quote):
        snap = _symbol(quote=quote, bars=[])
        assert (snap.last_close, snap.change_5d_pct, snap.high_20d, snap.low_20d) == (None,) * 4

    def test_headlines_from_news_fixture(self, news):
        snap = _symbol(news=news)
        assert len(snap.headlines) == 2
        first, second = snap.headlines
        assert first.headline.startswith("Fed Holds Rates")
        assert first.summary == "The Federal Reserve left its benchmark rate unchanged."
        assert first.source == "benzinga"
        assert first.published_at == datetime(2026, 7, 29, 18, 5, tzinfo=timezone.utc)
        assert first.age_seconds == (NOW - first.published_at).total_seconds()
        assert first.symbols == ("SPY", "QQQ")
        assert second.summary is None and second.symbols == ("NVDA",)

    def test_chain_is_summarised(self, quote, chain, enriched):
        snap = _symbol(quote=quote, chain=chain, enriched=enriched)
        assert snap.chain is not None
        assert snap.chain == _summary(chain, enriched)
        closed = _symbol(quote=quote, chain=chain, enriched=enriched, market_open=False)
        assert closed.stale is True and closed.chain.stale is True


class TestAccountSummary:
    def test_from_fixture(self, account):
        summary = account_summary_from(account)
        assert (summary.equity, summary.cash, summary.buying_power) == (100000.0, 25000.5, 200000.0)
        assert summary.fetched_at == account.fetched_at
        assert len(summary.positions) == 1
        position = summary.positions[0]
        assert (position.symbol, position.qty, position.side) == ("AAPL", 10.0, "long")
        assert position.avg_entry_price == 210.15
        assert position.market_value == 2150.1
        assert position.unrealized_pl == 48.6
        assert position.current_price == 215.01
        # the account number never crosses into the brain's input
        assert "PA3ABCD" not in repr(summary)

    def test_empty_account(self):
        summary = account_summary_from(AccountState.from_alpaca({"account_number": "X", "status": "ACTIVE"}))
        assert summary.equity is None and summary.positions == ()


class TestDataAge:
    def test_open_and_fresh_carries_only_the_feed_note(self, quote, chain, enriched):
        symbols = [_symbol(quote=quote, chain=chain, enriched=enriched)]
        age = data_age_from(OPEN_CLOCK, symbols, stale_after_seconds=STALE_AFTER)
        assert age.market_open is True
        assert age.next_open == OPEN_CLOCK.next_open and age.next_close == OPEN_CLOCK.next_close
        assert age.stale_after_seconds == STALE_AFTER
        assert age.any_stale is False
        assert age.warnings == (OPTIONS_FEED_NOTE,)

    def test_closed_market_warning(self, quote, chain, enriched):
        symbols = [_symbol(quote=quote, chain=chain, enriched=enriched, market_open=False)]
        age = data_age_from(CLOSED_CLOCK, symbols, stale_after_seconds=STALE_AFTER)
        assert age.market_open is False and age.any_stale is True
        assert age.warnings[0].startswith(MARKET_CLOSED_WARNING)
        assert "next open 2026-07-31 13:30 UTC" in age.warnings[0]
        # the market-wide line covers every symbol: no per-symbol stale lines
        assert age.warnings == (age.warnings[0], OPTIONS_FEED_NOTE)
        no_next = CLOSED_CLOCK.model_copy(update={"next_open": None})
        assert "next open unknown" in data_age_from(no_next, symbols, stale_after_seconds=1).warnings[0]

    def test_clock_unknown_warning(self, quote, chain, enriched):
        symbols = [_symbol(quote=quote, chain=chain, enriched=enriched, market_open=None)]
        age = data_age_from(None, symbols, stale_after_seconds=STALE_AFTER)
        assert age.market_open is None and age.next_open is None and age.next_close is None
        assert age.any_stale is True
        assert age.warnings == (CLOCK_UNKNOWN_WARNING, OPTIONS_FEED_NOTE)

    def test_old_data_while_open_warns_per_symbol(self, quote, chain, enriched):
        later = NOW + timedelta(minutes=30)
        symbols = [_symbol(quote=quote, chain=chain, enriched=enriched, now=later)]
        age = data_age_from(OPEN_CLOCK, symbols, stale_after_seconds=STALE_AFTER)
        assert age.any_stale is True
        assert age.warnings == (
            "SPY: spot quote is STALE (age 30m 02s)",
            "SPY: option chain is STALE (age 45m 00s)",
            OPTIONS_FEED_NOTE,
        )
        no_time = quote.model_copy(update={"last_time": None, "quote_time": None})
        age = data_age_from(OPEN_CLOCK, [_symbol(quote=no_time)], stale_after_seconds=STALE_AFTER)
        assert age.warnings == (
            "SPY: spot quote is STALE (unknown age)",
            "SPY: option chain missing",
        )

    def test_stale_chain_alone_sets_any_stale(self, quote, chain, enriched):
        """The spot is 2 s old (fresh at 60 s); the chain is 15 minutes old (stale)."""
        spy = _symbol(quote=quote, chain=chain, enriched=enriched, stale_after_seconds=60)
        assert spy.stale is False and spy.chain is not None and spy.chain.stale is True
        age = data_age_from(OPEN_CLOCK, [spy], stale_after_seconds=60)
        assert age.any_stale is True
        assert age.warnings == ("SPY: option chain is STALE (age 15m 00s)", OPTIONS_FEED_NOTE)

    def test_stale_spot_alone_sets_any_stale(self, quote, chain, enriched):
        """The mirror case: the chain is fresh, the spot is not."""
        old = quote.model_copy(update={"last_time": NOW - timedelta(minutes=45)})
        spy = _symbol(quote=old, chain=chain, enriched=enriched)
        assert spy.stale is True and spy.chain is not None and spy.chain.stale is False
        age = data_age_from(OPEN_CLOCK, [spy], stale_after_seconds=STALE_AFTER)
        assert age.any_stale is True
        assert age.warnings == ("SPY: spot quote is STALE (age 45m 00s)", OPTIONS_FEED_NOTE)

    def test_missing_data_warnings(self):
        age = data_age_from(OPEN_CLOCK, [_symbol()], stale_after_seconds=STALE_AFTER)
        assert age.warnings == ("SPY: spot price missing", "SPY: option chain missing")
        assert OPTIONS_FEED_NOTE not in age.warnings  # no chain anywhere
        assert age.any_stale is True

    def test_errors_become_warnings_in_order(self, quote, chain, enriched):
        spy = _symbol(quote=quote, chain=chain, enriched=enriched, errors=["failed to fetch news for SPY"])
        qqq = symbol_snapshot_from(
            "QQQ", None, [], None, [], [],
            now=NOW, stale_after_seconds=STALE_AFTER, market_open=True, strikes_each_side=2,
            errors=["failed to fetch spot quote for QQQ"],
        )
        age = data_age_from(
            OPEN_CLOCK, [spy, qqq], stale_after_seconds=STALE_AFTER,
            errors=["account state unavailable: failed to fetch account state"],
        )
        assert age.warnings == (
            "account state unavailable: failed to fetch account state",
            "SPY: failed to fetch news for SPY",
            "QQQ: failed to fetch spot quote for QQQ",
            "QQQ: spot price missing",
            "QQQ: option chain missing",
            OPTIONS_FEED_NOTE,
        )

    def test_warnings_are_deterministic_across_processes(self):
        """The one realistic nondeterminism — hash-seed or set-iteration order — is fixed
        within a process, so the check runs the same assembly in fresh interpreters with
        different ``PYTHONHASHSEED`` values and compares their warnings with this one's."""
        script = (
            "import json\n"
            "from datetime import datetime, timezone\n"
            "from aegis.brain.snapshot import data_age_from, symbol_snapshot_from\n"
            "from aegis.data.models import MarketClock\n"
            "now = datetime(2026, 7, 30, 20, 0, tzinfo=timezone.utc)\n"
            "clock = MarketClock(is_open=True, next_open=now, next_close=now)\n"
            "symbols = [\n"
            "    symbol_snapshot_from(name, None, [], None, [], [], now=now, stale_after_seconds=60,\n"
            "                         market_open=True, strikes_each_side=2,\n"
            "                         errors=[f'{name} feed error {i}' for i in range(3)])\n"
            "    for name in ('SPY', 'QQQ', 'IWM', 'DIA')\n"
            "]\n"
            "age = data_age_from(clock, symbols, stale_after_seconds=60,\n"
            "                    errors=['account state unavailable', 'clock drift'])\n"
            "print(json.dumps(age.warnings))\n"
        )
        runs = []
        for seed in ("1", "2", "3"):
            result = subprocess.run(
                [sys.executable, "-c", script], cwd=REPO_ROOT, capture_output=True, text=True,
                timeout=120, env={**os.environ, "PYTHONHASHSEED": seed},
            )
            assert result.returncode == 0, result.stderr
            runs.append(tuple(json.loads(result.stdout)))
        expected = (
            "account state unavailable",
            "clock drift",
            *(
                line
                for name in ("SPY", "QQQ", "IWM", "DIA")
                for line in (
                    *(f"{name}: {name} feed error {i}" for i in range(3)),
                    f"{name}: spot price missing",
                    f"{name}: option chain missing",
                )
            ),
        )
        assert runs == [expected] * 3


class _Fetcher:
    """A canned stand-in for one data-layer fetcher: returns ``result`` and
    records every call; ``error`` fails every call, ``errors_by_symbol``
    only the named symbols."""

    def __init__(self, result):
        self.result = result
        self.calls = []
        self.error = None
        self.errors_by_symbol = {}

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        symbol = args[0] if args else None
        error = self.errors_by_symbol.get(symbol, self.error)
        if error is not None:
            raise error
        return self.result


@pytest.fixture
def feed(monkeypatch, account, quote, bars, chain, news):
    """Replace the live fetchers on the snapshot module with canned models."""
    fetchers = {
        "get_market_clock": _Fetcher(OPEN_CLOCK),
        "get_account_state": _Fetcher(account),
        "get_spot": _Fetcher(quote),
        "get_bars": _Fetcher(bars),
        "get_option_chain": _Fetcher(chain),
        "get_news": _Fetcher(news),
    }
    for name, fetcher in fetchers.items():
        monkeypatch.setattr(snapshot_module, name, fetcher)
    return fetchers


@pytest.fixture
def config():
    return AegisConfig.model_validate(
        {"watchlist": ["SPY", "QQQ"], "brain": {"snapshot": {"stale_after_minutes": 20}}}
    )


class TestBuildMarketSnapshot:
    def test_fetchers_are_module_attributes_from_the_data_layer(self):
        # The seam the tests below rely on: the live functions themselves,
        # bound as module globals so monkeypatching the module replaces them.
        assert snapshot_module.get_market_clock is aegis.data.market.get_market_clock
        assert snapshot_module.get_spot is aegis.data.market.get_spot
        assert snapshot_module.get_bars is aegis.data.market.get_bars
        assert snapshot_module.get_option_chain is aegis.data.market.get_option_chain
        assert snapshot_module.get_account_state is aegis.data.account.get_account_state
        assert snapshot_module.get_news is aegis.data.news.get_news
        assert snapshot_module.enrich_chain is aegis.pricing.enrich.enrich_chain

    def test_assembles_everything_from_the_fakes(self, feed, config, chain, enriched):
        snapshot = build_market_snapshot(config=config, now=NOW)
        assert isinstance(snapshot, MarketSnapshot)
        assert snapshot.taken_at == NOW
        assert [s.symbol for s in snapshot.symbols] == ["SPY", "QQQ"]  # the watchlist
        assert snapshot.risk_free_rate == config.pricing.risk_free_rate
        assert snapshot.account is not None and snapshot.account.equity == 100000.0
        assert snapshot.data_age.market_open is True and snapshot.data_age.any_stale is False
        assert snapshot.data_age.stale_after_seconds == 20 * 60
        assert snapshot.data_age.warnings == (OPTIONS_FEED_NOTE,)
        spy = snapshot.symbols[0]
        assert spy.spot == 636.42 and spy.last_close == 636.4 and len(spy.headlines) == 2
        assert spy.errors == ()
        assert spy.chain is not None and spy.chain.stale is False
        # the default ATM window is 5 each side: the whole nine-strike chain
        assert len(spy.chain.contracts) == 18
        assert spy.chain.contracts[8].greeks_source == "vendor"
        assert spy.chain.contracts[9].greeks_source == "model"
        assert [len(f.calls) for f in feed.values()] == [1, 1, 2, 2, 2, 2]
        assert feed["get_news"].calls[0] == (("SPY",), {"limit": 5})

    def test_explicit_symbols_are_cleaned_and_deduplicated(self, feed, config):
        snapshot = build_market_snapshot([" nvda", "NVDA", "spy"], config=config, now=NOW)
        assert [s.symbol for s in snapshot.symbols] == ["NVDA", "SPY"]
        assert [c[0] for c in feed["get_spot"].calls] == [("NVDA",), ("SPY",)]

    def test_no_symbols_is_a_brain_error(self, feed, config):
        with pytest.raises(BrainError, match="build market snapshot"):
            build_market_snapshot([" ", ""], config=config, now=NOW)
        assert feed["get_spot"].calls == []

    def test_data_error_per_symbol_keeps_the_symbol(self, feed, config):
        feed["get_spot"].errors_by_symbol["QQQ"] = DataError("spot quote", "QQQ")
        feed["get_news"].errors_by_symbol["QQQ"] = DataError("news", "QQQ")
        snapshot = build_market_snapshot(config=config, now=NOW)
        assert [s.symbol for s in snapshot.symbols] == ["SPY", "QQQ"]
        spy, qqq = snapshot.symbols
        assert spy.errors == () and spy.spot == 636.42
        assert qqq.errors == ("failed to fetch spot quote for QQQ", "failed to fetch news for QQQ")
        assert qqq.spot is None and qqq.stale is True and qqq.headlines == ()
        assert qqq.chain is not None  # the chain fetch still succeeded
        assert qqq.last_close == 636.4
        assert "QQQ: failed to fetch spot quote for QQQ" in snapshot.data_age.warnings
        assert "QQQ: spot price missing" in snapshot.data_age.warnings
        assert snapshot.data_age.any_stale is True

    def test_every_fetcher_can_fail_and_the_symbol_still_appears(self, feed, config):
        for name in ("get_spot", "get_bars", "get_option_chain", "get_news"):
            feed[name].error = DataError(name, "SPY")
        snapshot = build_market_snapshot(["SPY"], config=config, now=NOW)
        spy = snapshot.symbols[0]
        assert spy.symbol == "SPY"
        assert len(spy.errors) == 4 and all(e.startswith("failed to fetch") for e in spy.errors)
        assert spy.spot is None and spy.last_close is None and spy.chain is None
        assert spy.headlines == ()
        assert OPTIONS_FEED_NOTE not in snapshot.data_age.warnings  # no chain anywhere
        assert "SPY: option chain missing" in snapshot.data_age.warnings

    def test_config_error_is_captured_like_a_data_error(self, feed, config):
        feed["get_spot"].error = ConfigError("missing required environment variable ALPACA_API_KEY")
        spy = build_market_snapshot(["SPY"], config=config, now=NOW).symbols[0]
        assert spy.errors == ("missing required environment variable ALPACA_API_KEY",)

    def test_clock_failure_leaves_market_open_unknown(self, feed, config):
        feed["get_market_clock"].error = DataError("market clock")
        snapshot = build_market_snapshot(config=config, now=NOW)
        assert snapshot.data_age.market_open is None
        assert snapshot.data_age.any_stale is True
        assert snapshot.data_age.warnings[:2] == (
            CLOCK_UNKNOWN_WARNING,
            "market clock unavailable: failed to fetch market clock",
        )
        assert all(s.stale and s.chain.stale for s in snapshot.symbols)

    def test_closed_market_marks_everything_stale(self, feed, config):
        feed["get_market_clock"].result = CLOSED_CLOCK
        snapshot = build_market_snapshot(config=config, now=NOW)
        assert snapshot.data_age.market_open is False
        assert snapshot.data_age.next_open == CLOSED_CLOCK.next_open
        assert snapshot.data_age.warnings[0].startswith(MARKET_CLOSED_WARNING)
        assert all(s.stale and s.chain.stale for s in snapshot.symbols)

    def test_account_failure_leaves_account_none(self, feed, config):
        feed["get_account_state"].error = DataError("account state")
        snapshot = build_market_snapshot(config=config, now=NOW)
        assert snapshot.account is None
        assert "account state unavailable: failed to fetch account state" in snapshot.data_age.warnings
        assert snapshot.data_age.market_open is True  # independent of the clock

    def test_enrichment_uses_the_config_pricing_block(self, feed, config, monkeypatch):
        seen = []

        def recording_enrich(chain, rate, **kwargs):
            seen.append((chain.underlying, rate, kwargs))
            return enrich_chain(chain, rate, **kwargs)

        monkeypatch.setattr(snapshot_module, "enrich_chain", recording_enrich)
        pricing = config.pricing.model_copy(
            update={"risk_free_rate": 0.05, "day_count_basis": 360, "expiry_time": "13:00"}
        )
        custom = config.model_copy(update={"pricing": pricing})
        snapshot = build_market_snapshot(["SPY"], config=custom, now=NOW)
        assert seen == [
            (
                "SPY",
                0.05,
                {
                    "spot": SPOT,
                    "now": NOW,
                    "day_count_basis": 360,
                    "expiry_time": "13:00",
                    "expiry_timezone": "America/New_York",
                },
            )
        ]
        assert snapshot.risk_free_rate == 0.05
        # the DTE follows the same expiry instant as the enrichment
        assert snapshot.symbols[0].chain.days_to_expiry == pytest.approx(22.0 - 3 / 24)

    def test_quote_spot_backs_a_chain_without_one(self, feed, config, chain, quote):
        feed["get_option_chain"].result = chain.model_copy(update={"spot": None})
        spy = build_market_snapshot(["SPY"], config=config, now=NOW).symbols[0]
        assert spy.errors == ()
        # 636.42 (the last trade) sits nearer 635 than 640
        assert spy.chain.spot == quote.spot and spy.chain.atm_strike == 635.0
        assert spy.chain.contracts[1].greeks_source == "model"  # enrichment ran

    def test_pricing_error_from_enrichment_is_captured(self, feed, config, chain):
        # no spot anywhere: enrich_chain refuses, the chain is still shown
        feed["get_option_chain"].result = chain.model_copy(update={"spot": None})
        feed["get_spot"].error = DataError("spot quote", "SPY")
        spy = build_market_snapshot(["SPY"], config=config, now=NOW).symbols[0]
        assert len(spy.errors) == 2
        assert spy.errors[1].startswith("cannot compute enrich for SPY")
        assert "no spot price" in spy.errors[1]
        assert spy.chain is not None and spy.chain.spot is None
        assert spy.chain.contracts[0].iv_source == "vendor"
        assert spy.chain.contracts[1].iv_source is None

        def broken(*args, **kwargs):
            raise PricingError("enrich", "SPY")

        feed["get_spot"].error = None
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(snapshot_module, "enrich_chain", broken)
            spy = build_market_snapshot(["SPY"], config=config, now=NOW).symbols[0]
        assert spy.errors == (str(PricingError("enrich", "SPY")),)
        assert spy.chain is not None

    def test_headlines_per_symbol_zero_skips_the_news_fetch(self, feed, config):
        quiet = AegisConfig.model_validate(
            {"watchlist": ["SPY"], "brain": {"snapshot": {"headlines_per_symbol": 0}}}
        )
        spy = build_market_snapshot(config=quiet, now=NOW).symbols[0]
        assert spy.headlines == () and spy.errors == ()
        assert feed["get_news"].calls == []

    def test_snapshot_tunables_come_from_config(self, feed):
        tight = AegisConfig.model_validate(
            {
                "watchlist": ["SPY"],
                "brain": {
                    "snapshot": {
                        "strikes_each_side": 1,
                        "headlines_per_symbol": 1,
                        "stale_after_minutes": 1,
                    }
                },
            }
        )
        snapshot = build_market_snapshot(config=tight, now=NOW)
        spy = snapshot.symbols[0]
        assert [c.strike for c in spy.chain.contracts] == [635.0, 635.0, 640.0, 640.0, 645.0, 645.0]
        assert feed["get_news"].calls[0][1] == {"limit": 1}
        assert snapshot.data_age.stale_after_seconds == 60.0
        assert spy.stale is False  # the trade is 2 s old
        assert spy.chain.stale is True  # the chain is 15 minutes old
        assert "SPY: option chain is STALE (age 15m 00s)" in snapshot.data_age.warnings
        assert snapshot.data_age.market_open is True
        assert snapshot.data_age.any_stale is True  # a stale chain counts though the spot is fresh

    def test_naive_now_is_treated_as_utc(self, feed, config):
        snapshot = build_market_snapshot(["SPY"], config=config, now=NOW.replace(tzinfo=None))
        assert snapshot.taken_at == NOW
        assert snapshot.symbols[0].spot_age_seconds == 2.0

    def test_now_defaults_to_the_clock(self, feed, config):
        before = datetime.now(timezone.utc)
        snapshot = build_market_snapshot(["SPY"], config=config)
        assert before <= snapshot.taken_at <= datetime.now(timezone.utc)
        assert snapshot.symbols[0].stale is True  # the 2026-07 fixtures are old news by now
