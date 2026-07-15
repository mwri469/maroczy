import pandas as pd
import pytest

from maroczy.datafeed.lse import LSEData
from maroczy.datafeed.router import DataSourcePolicy, UnifiedMarketData, _duration_to_days


class _FakeIBKR:
    """Mimics maroczy.broker.data.MarketData's .history() signature/shape."""

    def __init__(self, fail=False):
        self.calls = []
        self.fail = fail

    def history(self, symbol, duration="2 Y", bar_size="1 day", end="", **kwargs):
        self.calls.append({"symbol": symbol, "duration": duration, "bar_size": bar_size, "end": end})
        if self.fail:
            raise RuntimeError("IBKR pacing violation")
        return pd.DataFrame({"open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0], "volume": [100.0]})


class _FakeLSE:
    """Mimics maroczy.datafeed.lse.LSEData's .candles() signature/shape."""

    def __init__(self, fail=False):
        self.calls = []
        self.fail = fail

    def candles(self, symbol, timeframe="1d", start=None, end=None, **kwargs):
        self.calls.append({"symbol": symbol, "timeframe": timeframe, "start": start, "end": end})
        if self.fail:
            raise RuntimeError("LSE 429 rate limited")
        return pd.DataFrame({"open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0], "volume": [100.0]})


def test_duration_parsing():
    assert _duration_to_days("2 Y") == pytest.approx(730)
    assert _duration_to_days("30 D") == pytest.approx(30)
    assert _duration_to_days("6 M") == pytest.approx(180)
    assert _duration_to_days("garbage") is None


def test_explicit_source_wins():
    router = UnifiedMarketData(ibkr=_FakeIBKR(), lse=_FakeLSE())
    assert router.resolve_source(source="ibkr") == "ibkr"
    assert router.resolve_source(source="lse") == "lse"


def test_realtime_requires_ibkr():
    router = UnifiedMarketData(lse=_FakeLSE())
    with pytest.raises(ValueError):
        router.resolve_source(realtime=True)

    router2 = UnifiedMarketData(ibkr=_FakeIBKR(), lse=_FakeLSE())
    assert router2.resolve_source(realtime=True) == "ibkr"


def test_long_intraday_lookback_routes_to_lse():
    router = UnifiedMarketData(ibkr=_FakeIBKR(), lse=_FakeLSE(), policy=DataSourcePolicy(ibkr_intraday_limit_days=10))
    assert router.resolve_source(bar_size="1 min", duration="60 D") == "lse"


def test_short_intraday_lookback_routes_to_ibkr():
    router = UnifiedMarketData(ibkr=_FakeIBKR(), lse=_FakeLSE(), policy=DataSourcePolicy(ibkr_intraday_limit_days=10))
    assert router.resolve_source(bar_size="1 min", duration="5 D") == "ibkr"


def test_daily_history_prefers_lse_by_default():
    router = UnifiedMarketData(ibkr=_FakeIBKR(), lse=_FakeLSE())
    assert router.resolve_source(bar_size="1 day", duration="2 Y") == "lse"


def test_daily_history_uses_ibkr_when_lse_not_configured():
    router = UnifiedMarketData(ibkr=_FakeIBKR())
    assert router.resolve_source(bar_size="1 day", duration="2 Y") == "ibkr"


def test_fallback_on_failure():
    lse = _FakeLSE(fail=True)
    ibkr = _FakeIBKR(fail=False)
    router = UnifiedMarketData(ibkr=ibkr, lse=lse, policy=DataSourcePolicy(prefer_lse_for_history=True, fallback=True))
    out = router.history("AAPL", bar_size="1 day", duration="2 Y")
    assert not out.empty
    assert len(ibkr.calls) == 1  # fell back to IBKR after LSE failed


def test_no_fallback_raises():
    lse = _FakeLSE(fail=True)
    router = UnifiedMarketData(lse=lse, policy=DataSourcePolicy(fallback=False))
    with pytest.raises(RuntimeError):
        router.history("AAPL", bar_size="1 day", duration="2 Y")


def test_requires_at_least_one_source():
    with pytest.raises(ValueError):
        UnifiedMarketData()


def test_lse_data_requires_api_key(monkeypatch):
    # explicitly clear regardless of any real .env loaded into this process/environment
    monkeypatch.delenv("LSE_API_KEY", raising=False)
    monkeypatch.delenv("lse_api_key", raising=False)
    with pytest.raises(ValueError):
        LSEData(api_key=None, cache_path=None)
