"""Monthly-frequency price characteristics (CSV ``class == "crspm"``).

Period lengths are expressed in trading days (``~21`` per month, ``~252``
per year) so these work directly on daily bars from
:meth:`maroczy.broker.data.MarketData.history` without requiring a separate
monthly resample.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from maroczy.characteristics.functions import characteristic

_DAYS_PER_MONTH = 21


def _mom(df: pd.DataFrame, start_months: int, end_months: int = 1) -> pd.Series:
    """Cumulative return from ``t - start_months`` to ``t - end_months`` (skipping the most recent month)."""
    px = df["close"]
    start = start_months * _DAYS_PER_MONTH
    end = end_months * _DAYS_PER_MONTH
    return px.shift(end) / px.shift(start) - 1


@characteristic("ret_12_1")
def ret_12_1(df: pd.DataFrame) -> pd.Series:
    """Jegadeesh & Titman (1993): 12-month momentum, skip-month convention."""
    return _mom(df, 12)


@characteristic("ret_9_1")
def ret_9_1(df: pd.DataFrame) -> pd.Series:
    """Jegadeesh & Titman (1993): 9-month momentum."""
    return _mom(df, 9)


@characteristic("ret_6_1")
def ret_6_1(df: pd.DataFrame) -> pd.Series:
    """Jegadeesh & Titman (1993): 6-month momentum."""
    return _mom(df, 6)


@characteristic("ret_3_1")
def ret_3_1(df: pd.DataFrame) -> pd.Series:
    """Jegadeesh & Titman (1993): 3-month momentum."""
    return _mom(df, 3)


@characteristic("ret_1_0")
def ret_1_0(df: pd.DataFrame) -> pd.Series:
    """Jegadeesh (1990): short-term (1-month) reversal."""
    return df["close"].pct_change(_DAYS_PER_MONTH)


@characteristic("ret_36_12")
def ret_36_12(df: pd.DataFrame) -> pd.Series:
    """De Bondt & Thaler (1985): long-term reversal, 12-36 months back."""
    return _mom(df, 36, end_months=12)


@characteristic("ret_60_12")
def ret_60_12(df: pd.DataFrame) -> pd.Series:
    """De Bondt & Thaler (1985): long-term reversal, 12-60 months back."""
    return _mom(df, 60, end_months=12)


@characteristic("chmom")
def chmom(df: pd.DataFrame) -> pd.Series:
    """Gettleman & Marks (2006): change in 6-month momentum."""
    m6 = _mom(df, 6)
    return m6 - m6.shift(6 * _DAYS_PER_MONTH)


@characteristic("seas_1_1an")
def seas_1_1an(df: pd.DataFrame) -> pd.Series:
    """Heston & Sadka (2008): 1-year-lagged same-month return (annual seasonality)."""
    return df["close"].pct_change(_DAYS_PER_MONTH).shift(12 * _DAYS_PER_MONTH)


@characteristic("market_equity")
def market_equity(df: pd.DataFrame, shares_out: pd.Series | float) -> pd.Series:
    """Banz (1981): market capitalization = price * shares outstanding."""
    return df["close"] * shares_out


@characteristic("beta_dimson_21d")
def beta_dimson_21d(df: pd.DataFrame, mkt_ret: pd.Series, window: int = 21, lags: int = 1) -> pd.Series:
    """Dimson (1979): beta adjusting for non-synchronous trading via lead/lag market returns.

    Sums the slope coefficients of a regression of stock returns on
    ``lags`` leads, the contemporaneous, and ``lags`` lags of the market
    return (implemented via simple OLS on the stacked regressors).
    """
    ret = df["close"].pct_change()
    regs = {f"mkt_lag{k}": mkt_ret.shift(k) for k in range(-lags, lags + 1)}
    frame = pd.concat({"ret": ret, **regs}, axis=1).dropna()
    if frame.empty:
        return pd.Series(dtype=float)
    y = frame["ret"].to_numpy()
    X = np.column_stack([np.ones(len(frame))] + [frame[c].to_numpy() for c in regs])
    out = np.full(len(frame), np.nan)
    p = X.shape[1]
    for i in range(window - 1, len(frame)):
        Xi = X[i - window + 1 : i + 1]
        yi = y[i - window + 1 : i + 1]
        try:
            theta, *_ = np.linalg.lstsq(Xi, yi, rcond=None)
        except np.linalg.LinAlgError:
            continue
        out[i] = theta[1:].sum()
    return pd.Series(out, index=frame.index)
