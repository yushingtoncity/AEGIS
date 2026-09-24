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
