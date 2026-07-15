"""Signal construction from characteristics: cross-sectional standardization,
ranking, composite scoring, and long/short weight construction.
"""

from __future__ import annotations

import pandas as pd

__all__ = ["zscore", "rank_signal", "combine_signals", "long_short_weights"]


def zscore(data: pd.Series | pd.DataFrame, axis: int = 0, clip: float | None = 3.0) -> pd.Series | pd.DataFrame:
    """Cross-sectional (``axis=1``, i.e. across columns/symbols for a given row) or
    time-series (``axis=0``, default) z-score, optionally winsorized at ``+/- clip``.
    """
    mean = data.mean(axis=axis)
    std = data.std(axis=axis)
    if axis == 0:
        z = (data - mean) / std
    else:
        z = data.sub(mean, axis=0).div(std, axis=0)
    if clip is not None:
        z = z.clip(-clip, clip)
    return z


def rank_signal(data: pd.DataFrame, axis: int = 1, ascending: bool = True) -> pd.DataFrame:
    """Cross-sectional percentile rank in ``[0, 1]`` (``axis=1``: rank across symbols per row)."""
    return data.rank(axis=axis, pct=True, ascending=ascending)


def combine_signals(characteristics: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    """Weighted composite of (already standardized/ranked) characteristic columns.

    Parameters
    ----------
    characteristics: DataFrame with one column per characteristic.
    weights: mapping ``characteristic_name -> weight``. Missing columns are ignored
        with a proportional re-weighting of the remaining ones.
    """
    available = {k: v for k, v in weights.items() if k in characteristics.columns}
    if not available:
        raise ValueError("None of the requested characteristics are present in the DataFrame.")
    total_w = sum(abs(w) for w in available.values())
    composite = sum(characteristics[name] * (w / total_w) for name, w in available.items())
    return composite


def long_short_weights(
    signal: pd.Series,
    n_long: int | None = None,
    n_short: int | None = None,
    dollar_neutral: bool = True,
) -> pd.Series:
    """Turn a single cross-section of scores into a long/short weight vector.

    Parameters
    ----------
    signal: Series indexed by symbol (one cross-section / one date).
    n_long, n_short: number of names to go long/short; defaults to the top/bottom
        30% of the universe if not given.
    dollar_neutral: if True, long and short legs each sum to +0.5/-0.5 (total
        gross exposure = 1); otherwise each leg is equally weighted to +1/-1.
    """
    s = signal.dropna().sort_values(ascending=False)
    n = len(s)
    n_long = n_long if n_long is not None else max(int(n * 0.3), 1)
    n_short = n_short if n_short is not None else max(int(n * 0.3), 1)

    longs = s.index[:n_long]
    shorts = s.index[-n_short:] if n_short > 0 else []

    weights = pd.Series(0.0, index=signal.index)
    leg_weight = 0.5 if dollar_neutral else 1.0
    if len(longs):
        weights.loc[longs] = leg_weight / len(longs)
    if len(shorts):
        weights.loc[shorts] = -leg_weight / len(shorts)
    return weights
