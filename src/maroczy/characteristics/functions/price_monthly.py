"""Monthly-frequency price characteristics.

Period lengths are expressed in trading days (``~21`` per month, ``~252``
per year) so these work directly on daily bars without requiring a separate
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
    for i in range(window - 1, len(frame)):
        Xi = X[i - window + 1 : i + 1]
        yi = y[i - window + 1 : i + 1]
        try:
            theta, *_ = np.linalg.lstsq(Xi, yi, rcond=None)
        except np.linalg.LinAlgError:
            continue
        out[i] = theta[1:].sum()
    return pd.Series(out, index=frame.index)


# ---------------------------------------------------------------------------
# Additional momentum / seasonality
# ---------------------------------------------------------------------------

@characteristic("ret_12_6")
def ret_12_6(df: pd.DataFrame) -> pd.Series:
    """Novy-Marx (2012): intermediate momentum (months 7-12 back)."""
    return _mom(df, 12, end_months=6)


@characteristic("seas_1_1na")
def seas_1_1na(df: pd.DataFrame) -> pd.Series:
    """Heston & Sadka (2008): year 1-lagged return, non-annual months."""
    ret_m = df["close"].pct_change(_DAYS_PER_MONTH)
    annual = ret_m.shift(12 * _DAYS_PER_MONTH)
    # average of months 2-11 back from 1 year ago
    total = pd.Series(0.0, index=df.index)
    for m in [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]:
        total = total + ret_m.shift(m * _DAYS_PER_MONTH)
    return total / 11


@characteristic("seas_2_5an")
def seas_2_5an(df: pd.DataFrame) -> pd.Series:
    """Heston & Sadka (2008): years 2-5 lagged returns, annual."""
    ret_m = df["close"].pct_change(_DAYS_PER_MONTH)
    total = pd.Series(0.0, index=df.index)
    for yr in range(2, 6):
        total = total + ret_m.shift(yr * 12 * _DAYS_PER_MONTH)
    return total / 4


@characteristic("seas_2_5na")
def seas_2_5na(df: pd.DataFrame) -> pd.Series:
    """Heston & Sadka (2008): years 2-5 lagged returns, non-annual."""
    ret_m = df["close"].pct_change(_DAYS_PER_MONTH)
    total = pd.Series(0.0, index=df.index)
    count = 0
    for yr in range(2, 6):
        for offset in range(1, 12):
            total = total + ret_m.shift((yr * 12 + offset) * _DAYS_PER_MONTH)
            count += 1
    return total / count


@characteristic("seas_6_10an")
def seas_6_10an(df: pd.DataFrame) -> pd.Series:
    """Heston & Sadka (2008): years 6-10 lagged returns, annual."""
    ret_m = df["close"].pct_change(_DAYS_PER_MONTH)
    total = pd.Series(0.0, index=df.index)
    for yr in range(6, 11):
        total = total + ret_m.shift(yr * 12 * _DAYS_PER_MONTH)
    return total / 5


@characteristic("seas_6_10na")
def seas_6_10na(df: pd.DataFrame) -> pd.Series:
    """Heston & Sadka (2008): years 6-10 lagged returns, non-annual."""
    ret_m = df["close"].pct_change(_DAYS_PER_MONTH)
    total = pd.Series(0.0, index=df.index)
    count = 0
    for yr in range(6, 11):
        for offset in range(1, 12):
            total = total + ret_m.shift((yr * 12 + offset) * _DAYS_PER_MONTH)
            count += 1
    return total / count


@characteristic("seas_11_15an")
def seas_11_15an(df: pd.DataFrame) -> pd.Series:
    """Heston & Sadka (2008): years 11-15 lagged returns, annual."""
    ret_m = df["close"].pct_change(_DAYS_PER_MONTH)
    total = pd.Series(0.0, index=df.index)
    for yr in range(11, 16):
        total = total + ret_m.shift(yr * 12 * _DAYS_PER_MONTH)
    return total / 5


@characteristic("seas_11_15na")
def seas_11_15na(df: pd.DataFrame) -> pd.Series:
    """Heston & Sadka (2008): years 11-15 lagged returns, non-annual."""
    ret_m = df["close"].pct_change(_DAYS_PER_MONTH)
    total = pd.Series(0.0, index=df.index)
    count = 0
    for yr in range(11, 16):
        for offset in range(1, 12):
            total = total + ret_m.shift((yr * 12 + offset) * _DAYS_PER_MONTH)
            count += 1
    return total / count


@characteristic("seas_16_20an")
def seas_16_20an(df: pd.DataFrame) -> pd.Series:
    """Heston & Sadka (2008): years 16-20 lagged returns, annual."""
    ret_m = df["close"].pct_change(_DAYS_PER_MONTH)
    total = pd.Series(0.0, index=df.index)
    for yr in range(16, 21):
        total = total + ret_m.shift(yr * 12 * _DAYS_PER_MONTH)
    return total / 5


@characteristic("seas_16_20na")
def seas_16_20na(df: pd.DataFrame) -> pd.Series:
    """Heston & Sadka (2008): years 16-20 lagged returns, non-annual."""
    ret_m = df["close"].pct_change(_DAYS_PER_MONTH)
    total = pd.Series(0.0, index=df.index)
    count = 0
    for yr in range(16, 21):
        for offset in range(1, 12):
            total = total + ret_m.shift((yr * 12 + offset) * _DAYS_PER_MONTH)
            count += 1
    return total / count


# ---------------------------------------------------------------------------
# Residual momentum
# ---------------------------------------------------------------------------

@characteristic("resff3_12_1")
def resff3_12_1(df: pd.DataFrame, mkt_ret: pd.Series, smb: pd.Series = None, hml: pd.Series = None) -> pd.Series:
    """Blitz, Huij & Martens (2011): 12-month residual momentum (FF3)."""
    ret = df["close"].pct_change()
    if smb is None or hml is None:
        # fallback: CAPM residual momentum
        frame = pd.concat([ret, mkt_ret], axis=1).dropna()
        y, x = frame.iloc[:, 0].to_numpy(), frame.iloc[:, 1].to_numpy()
        X = np.column_stack([np.ones(len(frame)), x])
    else:
        frame = pd.concat([ret, mkt_ret, smb, hml], axis=1).dropna()
        y = frame.iloc[:, 0].to_numpy()
        X = np.column_stack([np.ones(len(frame))] + [frame.iloc[:, j].to_numpy() for j in range(1, 4)])
    try:
        theta, *_ = np.linalg.lstsq(X, y, rcond=None)
    except np.linalg.LinAlgError:
        return pd.Series(np.nan, index=df.index)
    resid = pd.Series(y - X @ theta, index=frame.index)
    # cumulative residual return months 2-12 back
    start = 12 * _DAYS_PER_MONTH
    end = 1 * _DAYS_PER_MONTH
    cum_resid = resid.rolling(start - end).sum()
    return cum_resid.shift(end)


@characteristic("resff3_6_1")
def resff3_6_1(df: pd.DataFrame, mkt_ret: pd.Series, smb: pd.Series = None, hml: pd.Series = None) -> pd.Series:
    """Blitz, Huij & Martens (2011): 6-month residual momentum (FF3)."""
    ret = df["close"].pct_change()
    if smb is None or hml is None:
        frame = pd.concat([ret, mkt_ret], axis=1).dropna()
        y, x = frame.iloc[:, 0].to_numpy(), frame.iloc[:, 1].to_numpy()
        X = np.column_stack([np.ones(len(frame)), x])
    else:
        frame = pd.concat([ret, mkt_ret, smb, hml], axis=1).dropna()
        y = frame.iloc[:, 0].to_numpy()
        X = np.column_stack([np.ones(len(frame))] + [frame.iloc[:, j].to_numpy() for j in range(1, 4)])
    try:
        theta, *_ = np.linalg.lstsq(X, y, rcond=None)
    except np.linalg.LinAlgError:
        return pd.Series(np.nan, index=df.index)
    resid = pd.Series(y - X @ theta, index=frame.index)
    start = 6 * _DAYS_PER_MONTH
    end = 1 * _DAYS_PER_MONTH
    cum_resid = resid.rolling(start - end).sum()
    return cum_resid.shift(end)


# ---------------------------------------------------------------------------
# Issuance / equity changes
# ---------------------------------------------------------------------------

@characteristic("eqnpo_12m")
def eqnpo_12m(df: pd.DataFrame, shares_out: pd.Series | float) -> pd.Series:
    """Daniel & Titman (2006): composite equity issuance (12-month log change in split-adj shares)."""
    if isinstance(shares_out, (int, float)):
        return pd.Series(0.0, index=df.index)
    log_shares = np.log(shares_out.replace(0, np.nan))
    return -(log_shares - log_shares.shift(12 * _DAYS_PER_MONTH))


@characteristic("eqnpo_60m")
def eqnpo_60m(df: pd.DataFrame, shares_out: pd.Series | float) -> pd.Series:
    """Daniel & Titman (2006): composite equity issuance (60-month)."""
    if isinstance(shares_out, (int, float)):
        return pd.Series(0.0, index=df.index)
    log_shares = np.log(shares_out.replace(0, np.nan))
    return -(log_shares - log_shares.shift(60 * _DAYS_PER_MONTH))


@characteristic("chcsho_12m")
def chcsho_12m(df: pd.DataFrame, shares_out: pd.Series | float) -> pd.Series:
    """Pontiff & Woodgate (2008): net stock issues (12-month % change in shares)."""
    if isinstance(shares_out, (int, float)):
        return pd.Series(0.0, index=df.index)
    return shares_out.pct_change(12 * _DAYS_PER_MONTH)


# ---------------------------------------------------------------------------
# Turnover / volume (monthly frequency versions)
# ---------------------------------------------------------------------------

@characteristic("dolvol")
def dolvol(df: pd.DataFrame) -> pd.Series:
    """Brennan, Chordia & Subrahmanyam (1998) GHZ: log avg monthly dollar volume."""
    dollar_vol = df["close"] * df["volume"]
    monthly = dollar_vol.rolling(_DAYS_PER_MONTH).mean()
    return np.log(monthly.replace(0, np.nan))


@characteristic("turn")
def turn(df: pd.DataFrame, shares_out: pd.Series | float) -> pd.Series:
    """Datar, Naik & Radcliffe (1998) GHZ: average monthly share turnover."""
    turnover = df["volume"] / shares_out
    return turnover.rolling(3 * _DAYS_PER_MONTH).mean()


# ---------------------------------------------------------------------------
# Dividend / price
# ---------------------------------------------------------------------------

@characteristic("div12m_me")
def div12m_me(df: pd.DataFrame, dividends: pd.Series = None, shares_out: pd.Series | float = 1.0) -> pd.Series:
    """Litzenberger & Ramaswamy (1979): trailing 12-month dividend yield."""
    if dividends is None:
        return pd.Series(np.nan, index=df.index)
    div_12m = dividends.rolling(12 * _DAYS_PER_MONTH).sum()
    price = df["close"]
    dy = div_12m / price.replace(0, np.nan)
    return dy.where(div_12m > 0)


@characteristic("price")
def price(df: pd.DataFrame) -> pd.Series:
    """Miller & Scholes (1982): share price level."""
    return df["close"]


# ---------------------------------------------------------------------------
# Indicators
# ---------------------------------------------------------------------------

@characteristic("ipo")
def ipo(df: pd.DataFrame) -> pd.Series:
    """Loughran & Ritter (1995): IPO indicator (1 if listed < 1 year)."""
    first_date = df.index[0]
    days_since = (df.index - first_date).days
    return pd.Series((days_since < 365).astype(float), index=df.index)


@characteristic("divi")
def divi(df: pd.DataFrame, dividends: pd.Series = None) -> pd.Series:
    """Michaely, Thaler & Womack (1995): dividend initiation indicator."""
    if dividends is None:
        return pd.Series(np.nan, index=df.index)
    has_div = dividends > 0
    had_div_prev_year = has_div.rolling(12 * _DAYS_PER_MONTH).sum().shift(_DAYS_PER_MONTH)
    has_div_now = has_div.rolling(_DAYS_PER_MONTH).sum()
    return ((has_div_now > 0) & (had_div_prev_year == 0)).astype(float)


@characteristic("divo")
def divo(df: pd.DataFrame, dividends: pd.Series = None) -> pd.Series:
    """Michaely, Thaler & Womack (1995): dividend omission indicator."""
    if dividends is None:
        return pd.Series(np.nan, index=df.index)
    has_div = dividends > 0
    had_div_prev_year = has_div.rolling(12 * _DAYS_PER_MONTH).sum().shift(_DAYS_PER_MONTH)
    has_div_now = has_div.rolling(12 * _DAYS_PER_MONTH).sum()
    return ((has_div_now == 0) & (had_div_prev_year > 0)).astype(float)


# ---------------------------------------------------------------------------
# Industry momentum (cross-sectional, operates on a panel)
# ---------------------------------------------------------------------------

@characteristic("indmom")
def indmom(df: pd.DataFrame, industry_ret: pd.Series = None) -> pd.Series:
    """Moskowitz & Grinblatt (1999): industry momentum (6-month industry return).

    Accepts the pre-computed cap-weighted industry return series as ``industry_ret``.
    """
    if industry_ret is None:
        return pd.Series(np.nan, index=df.index)
    return industry_ret.rolling(6 * _DAYS_PER_MONTH).apply(lambda x: (1 + x).prod() - 1, raw=True).shift(_DAYS_PER_MONTH)
