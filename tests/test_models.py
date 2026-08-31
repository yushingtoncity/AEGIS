"""Model parsing from canned fixture JSON — no live API calls."""

from datetime import date, datetime, timedelta, timezone

import pytest

from aegis.data.models import (
    AccountState,
    Bar,
    ChainSnapshot,
    NewsItem,
    OptionSnapshot,
    OptionType,
    Quote,
    format_age,
    parse_occ_symbol,
    utcnow,
)


class TestQuote:
    def test_parse(self, fixture):
        data = fixture("stock_quote.json")
        quote = Quote.from_alpaca("spy", quote=data["quote"], trade=data["trade"])
        assert quote.symbol == "SPY"
        assert quote.bid == 636.4
        assert quote.ask == 636.44
        assert quote.mid == pytest.approx(636.42)
        assert quote.last == 636.42
        assert quote.spot == 636.42  # prefers last trade
        assert quote.quote_time == datetime(
            2026, 7, 30, 19, 59, 59, 123456, tzinfo=timezone.utc
        )
        assert quote.spot_time == quote.last_time
        assert quote.fetched_at.tzinfo is not None

    def test_after_hours_zero_quote_maps_to_none(self, fixture):
        data = fixture("stock_quote_after_hours.json")
        quote = Quote.from_alpaca("SPY", quote=data["quote"], trade=data["trade"])
        assert quote.bid is None and quote.ask is None
        assert quote.mid is None
        assert quote.spot == 636.9  # falls back to the last trade

    def test_missing_everything(self):
        quote = Quote.from_alpaca("SPY")
        assert quote.spot is None
        assert quote.spot_time is None


class TestBar:
    def test_parse_list(self, fixture):
        data = fixture("bars.json")
        now = utcnow()
        bars = [Bar.from_alpaca(data["symbol"], b, fetched_at=now) for b in data["bars"]]
        assert len(bars) == 3
        assert bars[0].open == 630.1
        assert bars[0].trade_count == 412345
        assert bars[2].vwap is None and bars[2].trade_count is None
        assert all(b.timestamp.tzinfo is not None for b in bars)
        assert all(b.fetched_at == now for b in bars)


class TestOptionSnapshot:
    def test_full_snapshot(self, fixture):
        data = fixture("option_snapshot_full.json")
        snap = OptionSnapshot.from_alpaca(data["symbol"], "SPY", data)
        assert snap.underlying == "SPY"
        # strike/type/expiration parsed from the OCC symbol
        assert snap.strike == 640.0
        assert snap.option_type is OptionType.CALL
        assert snap.expiration == date(2026, 8, 21)
        assert snap.bid == 12.05 and snap.ask == 12.35
        assert snap.mid == pytest.approx(12.20)
        assert snap.last == 12.2
        assert snap.implied_vol == 0.1845
        assert snap.delta == 0.5321
        assert snap.theta == -0.0891
        assert snap.rho == 0.21

    def test_missing_greeks_never_crash(self, fixture):
        data = fixture("option_snapshot_no_greeks.json")
        snap = OptionSnapshot.from_alpaca(data["symbol"], "SPY", data)
        assert snap.option_type is OptionType.PUT
        assert snap.strike == 500.0
        for field in ("implied_vol", "delta", "gamma", "theta", "vega", "rho", "last"):
            assert getattr(snap, field) is None
        assert snap.bid == 0.0  # a zero option bid is real data, not missing
        assert snap.mid == pytest.approx(0.025)

    def test_contract_metadata_wins_over_occ_parse(self, fixture):
        data = fixture("option_snapshot_full.json")
        snap = OptionSnapshot.from_alpaca(
            data["symbol"],
            "SPY",
            data,
            strike=641.0,
            option_type="call",
            expiration="2026-08-22",
            open_interest="1523",
            volume=87.0,
        )
        assert snap.strike == 641.0
        assert snap.expiration == date(2026, 8, 22)
        assert snap.open_interest == 1523.0  # Alpaca sends strings
        assert snap.volume == 87.0

    def test_empty_payload(self):
        snap = OptionSnapshot.from_alpaca("SPY260821C00640000", "SPY", None)
        assert snap.strike == 640.0
        assert snap.bid is None and snap.mid is None

    def test_unparseable_symbol_degrades_to_none(self):
        snap = OptionSnapshot.from_alpaca("WEIRD", "SPY", {})
        assert snap.strike is None
        assert snap.option_type is None
        assert snap.expiration is None


class TestParseOccSymbol:
    def test_call(self):
        root, exp, opt_type, strike = parse_occ_symbol("SPY260821C00640000")
        assert (root, exp, opt_type, strike) == (
            "SPY", date(2026, 8, 21), OptionType.CALL, 640.0,
        )

    def test_put_with_fractional_strike(self):
        root, exp, opt_type, strike = parse_occ_symbol("AAPL260918P00232500")
        assert root == "AAPL"
        assert opt_type is OptionType.PUT
        assert strike == 232.5

    @pytest.mark.parametrize("bad", ["AAPL", "SPY260821X00640000", "SPY26AB21C00640000", ""])
    def test_invalid_raises(self, bad):
        with pytest.raises(ValueError):
            parse_occ_symbol(bad)


def _contract(strike, opt_type, **kwargs):
    symbol = f"SPY260821{'C' if opt_type is OptionType.CALL else 'P'}{int(strike * 1000):08d}"
    payload = {
        "latest_quote": {
            "bid_price": 1.0,
            "ask_price": 1.2,
            "timestamp": "2026-07-30T19:45:00+00:00",
        }
    }
    return OptionSnapshot.from_alpaca(symbol, "SPY", payload, **kwargs)


class TestChainSnapshot:
    @pytest.fixture
    def chain(self):
        contracts = [
            _contract(s, t)
            for s in (630.0, 635.0, 640.0, 645.0, 650.0)
            for t in (OptionType.CALL, OptionType.PUT)
        ]
        return ChainSnapshot(
            underlying="SPY",
            expiration=date(2026, 8, 21),
            spot=638.9,
            contracts=contracts,
        )

    def test_sides_sorted_by_strike(self, chain):
        assert [c.strike for c in chain.calls] == [630.0, 635.0, 640.0, 645.0, 650.0]
        assert len(chain.puts) == 5
        assert all(c.option_type is OptionType.PUT for c in chain.puts)

    def test_atm_selection(self, chain):
        assert chain.atm_strike() == 640.0
        assert chain.atm_strikes(each_side=1) == [635.0, 640.0, 645.0]
        # window clamps at the edge of available strikes
        assert chain.atm_strikes(each_side=10) == chain.strikes

    def test_contract_at(self, chain):
        found = chain.contract_at(640.0, OptionType.PUT)
        assert found is not None and found.option_type is OptionType.PUT
        assert chain.contract_at(999.0, OptionType.CALL) is None

    def test_latest_quote_time(self, chain):
        assert chain.latest_quote_time == datetime(
            2026, 7, 30, 19, 45, tzinfo=timezone.utc
        )

    def test_no_spot_falls_back_to_lowest_strikes(self, chain):
        no_spot = chain.model_copy(update={"spot": None})
        assert no_spot.atm_strike() is None
        assert no_spot.atm_strikes(each_side=1) == [630.0, 635.0, 640.0]


class TestAccountState:
    def test_parse(self, fixture):
        data = fixture("account.json")
        state = AccountState.from_alpaca(data["account"], data["positions"])
        assert state.masked_account_number == "****2345"
        assert "PA3ABCD" not in state.masked_account_number
        assert state.status == "ACTIVE"
        assert state.equity == 100000.0  # converted from Alpaca's strings
        assert state.buying_power == 200000.0
        assert state.cash == 25000.5
        assert len(state.positions) == 1
        position = state.positions[0]
        assert position.symbol == "AAPL"
        assert position.qty == 10.0
        assert position.side == "long"
        assert position.unrealized_pl == 48.6

    def test_missing_fields(self):
        state = AccountState.from_alpaca({"account_number": "X1", "status": "ACTIVE"})
        assert state.equity is None
        assert state.positions == []
        assert state.masked_account_number == "****X1"


class TestNewsItem:
    def test_parse(self, fixture):
        items = [NewsItem.from_alpaca(n) for n in fixture("news.json")["news"]]
        assert len(items) == 2
        first = items[0]
        assert first.id == 43107442
        assert first.headline.startswith("Fed Holds Rates")
        assert first.symbols == ["SPY", "QQQ"]
        assert first.published_at == datetime(2026, 7, 29, 18, 5, tzinfo=timezone.utc)
        assert items[1].summary is None  # empty string normalized
        assert items[1].url is None


class TestFormatAge:
    @pytest.mark.parametrize(
        ("delta", "expected"),
        [
            (timedelta(seconds=30), "30s"),
            (timedelta(seconds=200), "3m 20s"),
            (timedelta(hours=2, minutes=5), "2h 05m"),
            (timedelta(days=3, hours=4), "3d 04h"),
        ],
    )
    def test_formats(self, delta, expected):
        ts = datetime(2026, 7, 30, 12, 0, tzinfo=timezone.utc)
        assert format_age(ts, now=ts + delta) == expected

    def test_none_and_future(self):
        assert format_age(None) == "unknown age"
        ts = utcnow() + timedelta(minutes=5)  # clock skew never goes negative
        assert format_age(ts) == "0s"
