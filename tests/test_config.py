"""Config loading and validation."""

import pytest

from aegis.config import (
    DEFAULT_CONFIG_PATH,
    AegisConfig,
    ConfigError,
    load_config,
    require_env,
)

MINIMAL = "watchlist: [spy, qqq]\n"


def write(tmp_path, text):
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_shipped_config_is_valid():
    config = load_config(DEFAULT_CONFIG_PATH)
    assert config.watchlist == ["SPY", "QQQ", "AAPL", "NVDA", "MSFT"]
    assert config.cache.ttl_seconds.quotes == 5
    assert config.cache.ttl_seconds.chains == 30
    assert config.cache.ttl_seconds.news == 60
    assert config.risk_limits.max_open_positions == 5


def test_minimal_config_gets_defaults(tmp_path):
    config = load_config(write(tmp_path, MINIMAL))
    assert config.watchlist == ["SPY", "QQQ"]  # uppercased
    assert config.cache.ttl_seconds.quotes == 5
    assert config.bars.timeframe == "1Day"
    assert config.bars.lookback_days == 30
    assert config.risk_limits.no_trade_list == []


def test_watchlist_strips_and_uppercases(tmp_path):
    config = load_config(write(tmp_path, "watchlist: ['  aapl ', nvda]\n"))
    assert config.watchlist == ["AAPL", "NVDA"]


def test_empty_watchlist_rejected(tmp_path):
    with pytest.raises(ConfigError):
        load_config(write(tmp_path, "watchlist: []\n"))


def test_whitespace_only_watchlist_rejected(tmp_path):
    with pytest.raises(ConfigError):
        load_config(write(tmp_path, "watchlist: ['   ']\n"))


@pytest.mark.parametrize(
    "timeframe",
    [
        "1min", "Day", "0Min", "15Sec", "daily",
        # amounts the alpaca-py SDK rejects must fail at config load, not per-fetch
        "60Min", "24Hour", "2Day", "5Week", "4Month",
    ],
)
def test_invalid_timeframe_rejected(tmp_path, timeframe):
    text = MINIMAL + f"bars:\n  timeframe: {timeframe}\n"
    with pytest.raises(ConfigError):
        load_config(write(tmp_path, text))


@pytest.mark.parametrize(
    "timeframe",
    ["1Min", "15Min", "59Min", "1Hour", "23Hour", "1Day", "1Week", "3Month", "12Month"],
)
def test_valid_timeframes_accepted(tmp_path, timeframe):
    text = MINIMAL + f"bars:\n  timeframe: {timeframe}\n"
    assert load_config(write(tmp_path, text)).bars.timeframe == timeframe


def test_nonpositive_ttl_rejected(tmp_path):
    text = MINIMAL + "cache:\n  ttl_seconds:\n    quotes: 0\n"
    with pytest.raises(ConfigError):
        load_config(write(tmp_path, text))


def test_risk_limit_bounds(tmp_path):
    text = MINIMAL + "risk_limits:\n  max_position_pct: 150\n"
    with pytest.raises(ConfigError):
        load_config(write(tmp_path, text))


def test_no_trade_list_uppercased(tmp_path):
    text = MINIMAL + "risk_limits:\n  no_trade_list: [gme, ' amc ']\n"
    config = load_config(write(tmp_path, text))
    assert config.risk_limits.no_trade_list == ["GME", "AMC"]


def test_unknown_key_rejected(tmp_path):
    with pytest.raises(ConfigError):
        load_config(write(tmp_path, MINIMAL + "surprise: true\n"))


def test_missing_file_raises_config_error(tmp_path):
    with pytest.raises(ConfigError, match="cannot read config file"):
        load_config(tmp_path / "nope.yaml")


def test_invalid_yaml_raises_config_error(tmp_path):
    with pytest.raises(ConfigError, match="not valid YAML"):
        load_config(write(tmp_path, "watchlist: [unclosed\n"))


def test_non_mapping_yaml_rejected(tmp_path):
    with pytest.raises(ConfigError, match="must be a YAML mapping"):
        load_config(write(tmp_path, "- just\n- a\n- list\n"))


def test_require_env_present(monkeypatch):
    monkeypatch.setenv("AEGIS_TEST_VAR", "value")
    assert require_env("AEGIS_TEST_VAR") == "value"


@pytest.mark.parametrize("value", [None, "", "   "])
def test_require_env_missing_or_blank(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("AEGIS_TEST_VAR", raising=False)
    else:
        monkeypatch.setenv("AEGIS_TEST_VAR", value)
    with pytest.raises(ConfigError, match="AEGIS_TEST_VAR"):
        require_env("AEGIS_TEST_VAR")


def test_model_rejects_extra_nested_keys():
    with pytest.raises(Exception):
        AegisConfig.model_validate(
            {"watchlist": ["SPY"], "cache": {"ttl_seconds": {"bogus": 1}}}
        )


def test_pricing_block_defaults(tmp_path):
    config = load_config(write(tmp_path, MINIMAL))
    assert config.pricing.risk_free_rate == 0.04
    assert config.pricing.day_count_basis == 365
    assert config.pricing.contract_multiplier == 100
    assert config.pricing.expiry_time == "16:00"
    assert config.pricing.expiry_timezone == "America/New_York"
    assert config.store.db_path == "data/aegis.db"


@pytest.mark.parametrize(
    "text",
    [
        "pricing:\n  day_count_basis: 0\n",
        "pricing:\n  expiry_time: '25:00'\n",
        "pricing:\n  expiry_time: '4pm'\n",
        "pricing:\n  expiry_timezone: Mars/Olympus\n",
        "pricing:\n  contract_multiplier: 0\n",
        "store:\n  db_path: '   '\n",
    ],
)
def test_invalid_pricing_and_store_blocks_rejected(tmp_path, text):
    with pytest.raises(ConfigError):
        load_config(write(tmp_path, MINIMAL + text))


def test_brain_block_defaults(tmp_path):
    config = load_config(write(tmp_path, MINIMAL))
    assert config.brain.cadence_minutes == 30
    assert config.brain.max_retries == 3
    assert config.brain.stages.scan.model == "claude-haiku-4-5-20251001"
    assert config.brain.stages.scan.effort is None
    assert config.brain.stages.thesis.model == "claude-opus-5-5"
    assert config.brain.stages.proposal.model == "claude-fable-5-1"
    assert [config.brain.stage(s).max_tokens for s in ("scan", "thesis", "proposal")] == [3000, 4000, 3000]
    assert config.brain.daily_token_budget >= config.brain.per_cycle_token_cap
    with pytest.raises(KeyError):
        config.brain.stage("bogus")


def test_shipped_brain_block_prices_cover_every_stage_model():
    config = load_config(DEFAULT_CONFIG_PATH)
    for name in ("scan", "thesis", "proposal"):
        assert config.brain.stage(name).model in config.brain.prices_per_mtok


@pytest.mark.parametrize(
    "text",
    [
        "brain:\n  stages:\n    scan:\n      model: ''\n      max_tokens: 10\n",
        "brain:\n  stages:\n    scan:\n      model: m\n      max_tokens: 0\n",
        "brain:\n  stages:\n    scan:\n      model: m\n      max_tokens: 10\n      effort: warp\n",
        "brain:\n  stages:\n    scan:\n      model: m\n      max_tokens: 10\n      temperature: 0.2\n",
        "brain:\n  per_cycle_token_cap: 1000\n  daily_token_budget: 999\n",
        "brain:\n  prices_per_mtok:\n    m: {input: -1, output: 5}\n",
        "brain:\n  cadence_minutes: 0\n",
        "brain:\n  snapshot:\n    max_dte: -1\n",
    ],
)
def test_invalid_brain_blocks_rejected(tmp_path, text):
    with pytest.raises(ConfigError):
        load_config(write(tmp_path, MINIMAL + text))


def test_risk_limits_defaults(tmp_path):
    limits = load_config(write(tmp_path, MINIMAL)).risk_limits
    assert (limits.daily_loss_limit_pct, limits.halt_fallback_hours) == (2.0, 24.0)
    assert (limits.max_daily_trades, limits.max_open_positions) == (10, 5)
    assert (limits.no_trade_list, limits.watchlist_only) == ([], True)
    assert limits.max_position_pct == 5.0
    assert (limits.duplicate_window_minutes, limits.allow_market_orders) == (60, False)
    assert limits.limit_price_tolerance_pct == 5.0
    assert (limits.max_quote_age_seconds.equity, limits.max_quote_age_seconds.option) == (
        120.0,
        1200.0,
    )
    assert (limits.min_dte, limits.max_loss_per_trade, limits.max_contracts) == (7, 1000.0, 10)
    assert (limits.reject_short_sales, limits.min_confidence) == (False, 0.5)
    assert (limits.auto_execute.enabled, limits.auto_execute.max_notional) == (True, 1000.0)


def test_shipped_risk_limits_match_the_code_defaults():
    """config.yaml states every limit explicitly; none silently rides on a default."""
    import yaml

    from aegis.config import RiskLimits

    shipped = load_config(DEFAULT_CONFIG_PATH).risk_limits
    assert shipped == RiskLimits()
    block = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))["risk_limits"]
    assert set(block) == set(RiskLimits.model_fields)
    assert set(block["auto_execute"]) == {"enabled", "max_notional"}
    assert set(block["max_quote_age_seconds"]) == {"equity", "option"}


def test_shipped_snapshot_window_matches_the_code_default():
    """brain.snapshot.max_dte is stated in config.yaml, not left to a default."""
    import yaml

    from aegis.config import SnapshotConfig

    shipped = load_config(DEFAULT_CONFIG_PATH)
    assert shipped.brain.snapshot.max_dte == SnapshotConfig().max_dte == 45
    block = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    assert block["brain"]["snapshot"]["max_dte"] == 45
    assert shipped.risk_limits.min_dte == 7


def test_risk_limits_are_frozen(tmp_path):
    limits = load_config(write(tmp_path, MINIMAL)).risk_limits
    with pytest.raises(Exception):
        limits.max_position_pct = 100.0
    with pytest.raises(Exception):
        limits.auto_execute.enabled = False


@pytest.mark.parametrize(
    "text",
    [
        "risk_limits:\n  daily_loss_limit_pct: 0\n",
        "risk_limits:\n  halt_fallback_hours: 0\n",
        "risk_limits:\n  max_daily_trades: -1\n",
        "risk_limits:\n  duplicate_window_minutes: -5\n",
        "risk_limits:\n  limit_price_tolerance_pct: -1\n",
        "risk_limits:\n  min_dte: -1\n",
        "risk_limits:\n  max_quote_age_seconds:\n    equity: -1\n",
        "risk_limits:\n  max_quote_age_seconds:\n    option: -0.5\n",
        "risk_limits:\n  max_quote_age_seconds:\n    option: .inf\n",
        "risk_limits:\n  max_quote_age_seconds:\n    equity: .nan\n",
        "risk_limits:\n  max_quote_age_seconds:\n    future: 10\n",
        "risk_limits:\n  max_quote_age_seconds: 120\n",
        "risk_limits:\n  max_loss_per_trade: -1\n",
        "risk_limits:\n  max_contracts: -1\n",
        "risk_limits:\n  min_confidence: 1.5\n",
        "risk_limits:\n  auto_execute:\n    max_notional: -1\n",
        "risk_limits:\n  auto_execute:\n    surprise: true\n",
        "risk_limits:\n  allow_market_orders: maybe\n",
        "risk_limits:\n  max_notional: 10\n",
        # a non-finite number would switch the limit off without saying so
        "risk_limits:\n  max_loss_per_trade: .inf\n",
        "risk_limits:\n  limit_price_tolerance_pct: .inf\n",
        "risk_limits:\n  halt_fallback_hours: .inf\n",
        "risk_limits:\n  max_position_pct: .nan\n",
        "risk_limits:\n  min_confidence: .nan\n",
        "risk_limits:\n  auto_execute:\n    max_notional: .inf\n",
        "risk_limits:\n  auto_execute:\n    max_notional: .nan\n",
    ],
)
def test_invalid_risk_limits_rejected(tmp_path, text):
    with pytest.raises(ConfigError):
        load_config(write(tmp_path, MINIMAL + text))


def test_broker_block_defaults(tmp_path):
    broker = load_config(write(tmp_path, MINIMAL)).broker
    assert broker.enabled is False and broker.kind == "paper"
    assert (broker.http_timeout_seconds, broker.max_decision_age_seconds) == (10.0, 300.0)
    assert (broker.approval_ttl_seconds, broker.not_found_grace_seconds) == (900.0, 120.0)


def test_shipped_broker_block_matches_the_code_defaults():
    """config.yaml states every broker value, ships with placing off, and
    holds the D1 values the user approved."""
    import yaml

    from aegis.config import BrokerConfig

    shipped = load_config(DEFAULT_CONFIG_PATH).broker
    assert shipped == BrokerConfig()
    assert shipped.enabled is False
    block = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))["broker"]
    assert set(block) == set(BrokerConfig.model_fields)
    assert (
        block["max_decision_age_seconds"], block["approval_ttl_seconds"],
        block["not_found_grace_seconds"],
    ) == (300, 900, 120)


@pytest.mark.parametrize(
    "broker",
    [
        "  kind: live\n",
        "  http_timeout_seconds: 0\n",
        "  max_decision_age_seconds: -1\n",
        "  approval_ttl_seconds: .inf\n",
        "  not_found_grace_seconds: -5\n",
        "  enabled: maybe\n",
        "  base_url: https://api.alpaca.markets\n",
    ],
)
def test_broker_block_refuses_what_it_cannot_honour(tmp_path, broker):
    with pytest.raises(ConfigError, match="broker"):
        load_config(write(tmp_path, MINIMAL + "broker:\n" + broker))
