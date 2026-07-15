import numpy as np
import pandas as pd

from maroczy.characteristics import CharacteristicEngine, CharacteristicRegistry


def _fake_bars(n=800, seed=0):
    rng = np.random.default_rng(seed)
    ret = rng.standard_normal(n) * 0.01
    close = 100 * np.cumprod(1 + ret)
    high = close * (1 + np.abs(rng.standard_normal(n) * 0.005))
    low = close * (1 - np.abs(rng.standard_normal(n) * 0.005))
    open_ = close * (1 + rng.standard_normal(n) * 0.001)
    volume = rng.integers(1e5, 1e6, size=n).astype(float)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=idx)


def test_registry_lists_implemented():
    registry = CharacteristicRegistry()
    names = registry.names()
    assert len(names) > 20
    assert "ami_126d" in names
    assert "ret_12_1" in names


def test_registry_get_function():
    registry = CharacteristicRegistry()
    func = registry.get("retvol")
    assert callable(func)


def test_engine_compute_core_subset():
    bars = _fake_bars()
    engine = CharacteristicEngine()
    out = engine.compute(bars, names=["ret_12_1", "ami_126d", "retvol", "bidaskhl_21d"])
    assert set(out.columns) == {"ret_12_1", "ami_126d", "retvol", "bidaskhl_21d"}
    assert len(out) == len(bars)


def test_engine_compute_with_market_kwarg():
    bars = _fake_bars()
    mkt_ret = bars["close"].pct_change()
    engine = CharacteristicEngine()
    out = engine.compute(bars, names=["ivol_capm_21d", "beta_60m"], mkt_ret=mkt_ret)
    assert "ivol_capm_21d" in out.columns
    assert "beta_60m" in out.columns


def test_engine_compute_universe():
    panel = {"AAA": _fake_bars(seed=1), "BBB": _fake_bars(seed=2)}
    engine = CharacteristicEngine()
    out = engine.compute_universe(panel, names=["ret_12_1", "retvol"])
    assert isinstance(out.index, pd.MultiIndex)
    assert set(out.index.get_level_values("symbol")) == {"AAA", "BBB"}
