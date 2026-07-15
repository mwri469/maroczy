import numpy as np
import pandas as pd

from maroczy.strategy import combine_signals, long_short_weights, rank_signal, run_backtest, zscore


def test_zscore_cross_sectional():
    df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]})
    z = zscore(df, axis=1)
    assert np.allclose(z.mean(axis=1), 0, atol=1e-8)


def test_rank_signal_bounds():
    df = pd.DataFrame(np.random.default_rng(0).standard_normal((5, 10)))
    r = rank_signal(df)
    assert (r.to_numpy() >= 0).all() and (r.to_numpy() <= 1).all()


def test_combine_signals_weighted():
    df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    combo = combine_signals(df, {"a": 1.0, "b": 1.0})
    assert np.allclose(combo, [2.0, 3.0])


def test_long_short_weights_dollar_neutral():
    signal = pd.Series(np.arange(10), index=[f"S{i}" for i in range(10)])
    w = long_short_weights(signal, n_long=3, n_short=3, dollar_neutral=True)
    assert np.isclose(w.sum(), 0.0)
    assert np.isclose(w[w > 0].sum(), 0.5)
    assert np.isclose(w[w < 0].sum(), -0.5)


def test_run_backtest_basic():
    rng = np.random.default_rng(0)
    dates = pd.date_range("2021-01-01", periods=100, freq="B")
    symbols = ["A", "B", "C"]
    returns = pd.DataFrame(rng.standard_normal((100, 3)) * 0.01, index=dates, columns=symbols)
    weights = pd.DataFrame(1 / 3, index=dates, columns=symbols)
    result = run_backtest(weights, returns, cost_bps=1.0)
    assert "sharpe" in result.stats
    assert len(result.returns) == len(dates)
