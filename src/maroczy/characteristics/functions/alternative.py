"""Alternative data characteristics.

Signal-ready outputs from insider trades, dividends, COT positioning, and
FX data. Functions accept pre-fetched DataFrames from LSE endpoints.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from maroczy.characteristics.functions import characteristic


# ---------------------------------------------------------------------------
# Insider trading signals
# ---------------------------------------------------------------------------

@characteristic("insider_buy_ratio")
def insider_buy_ratio(insider_df: pd.DataFrame, window: int = 90) -> pd.Series:
    """Insider buy ratio: purchases / (purchases + sales) over trailing window.

    Parameters
    ----------
    insider_df: DataFrame from ``lse.insider_trades()`` with columns including
        ``date`` and ``type`` (containing "Purchase" or "Sale").
    """
    df = insider_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")
    is_buy = df["type"].str.contains("Purchase|Buy", case=False, na=False).astype(float)
    is_sell = df["type"].str.contains("Sale|Sell", case=False, na=False).astype(float)
    buys = is_buy.rolling(f"{window}D").sum()
    total = (is_buy + is_sell).rolling(f"{window}D").sum()
    return (buys / total.replace(0, np.nan)).rename("insider_buy_ratio")


@characteristic("insider_net_value")
def insider_net_value(insider_df: pd.DataFrame, me: pd.Series = None) -> pd.Series:
    """Net $ purchased by insiders / market cap (if provided) over trailing 90 days.

    Parameters
    ----------
    insider_df: DataFrame with ``date``, ``type``, ``value`` columns.
    me: market equity series (optional, for normalization).
    """
    df = insider_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")
    value_col = next((c for c in df.columns if c.lower() in ("value", "amount", "transactionamount")), None)
    if value_col is None:
        return pd.Series(np.nan, index=df.index, name="insider_net_value")
    signed = df[value_col].copy()
    signed[df["type"].str.contains("Sale|Sell", case=False, na=False)] *= -1
    net = signed.rolling("90D").sum()
    if me is not None:
        net = net / me.reindex(net.index, method="ffill").replace(0, np.nan)
    return net.rename("insider_net_value")


@characteristic("insider_cluster")
def insider_cluster(insider_df: pd.DataFrame, window: int = 30) -> pd.Series:
    """Number of distinct insiders buying in trailing window (cluster signal)."""
    df = insider_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    buys = df[df["type"].str.contains("Purchase|Buy", case=False, na=False)]
    name_col = next((c for c in buys.columns if c.lower() in ("name", "insider", "reportingname")), "type")
    buys = buys.sort_values("date").set_index("date")
    # Count unique insiders per rolling window
    counts = buys[name_col].rolling(f"{window}D").apply(lambda x: len(set(x)), raw=False)
    return counts.rename("insider_cluster")


# ---------------------------------------------------------------------------
# Dividend signals
# ---------------------------------------------------------------------------

@characteristic("div_yield_change")
def div_yield_change(div_df: pd.DataFrame, price: pd.Series) -> pd.Series:
    """Change in trailing 12-month dividend yield."""
    df = div_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    amount_col = next((c for c in df.columns if c.lower() in ("amount", "dividend", "dividendamount")), None)
    if amount_col is None:
        return pd.Series(np.nan, index=price.index, name="div_yield_change")
    # Aggregate to daily
    daily_div = df.groupby("date")[amount_col].sum().reindex(price.index, fill_value=0)
    trailing_12m = daily_div.rolling(252).sum()
    dy = trailing_12m / price.replace(0, np.nan)
    return dy.diff(21).rename("div_yield_change")


@characteristic("div_initiation")
def div_initiation(div_df: pd.DataFrame, price: pd.Series) -> pd.Series:
    """Dividend initiation: 1 if firm started paying dividends in trailing year."""
    df = div_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    amount_col = next((c for c in df.columns if c.lower() in ("amount", "dividend", "dividendamount")), None)
    if amount_col is None:
        return pd.Series(0.0, index=price.index, name="div_initiation")
    daily_div = df.groupby("date")[amount_col].sum().reindex(price.index, fill_value=0)
    has_div_now = daily_div.rolling(252).sum() > 0
    had_div_prev = daily_div.shift(252).rolling(252).sum() > 0
    return (has_div_now & ~had_div_prev).astype(float).rename("div_initiation")


@characteristic("div_growth_3y")
def div_growth_3y(div_df: pd.DataFrame, price: pd.Series) -> pd.Series:
    """3-year CAGR in dividends per share."""
    df = div_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    amount_col = next((c for c in df.columns if c.lower() in ("amount", "dividend", "dividendamount")), None)
    if amount_col is None:
        return pd.Series(np.nan, index=price.index, name="div_growth_3y")
    daily_div = df.groupby("date")[amount_col].sum().reindex(price.index, fill_value=0)
    annual_now = daily_div.rolling(252).sum()
    annual_3y_ago = daily_div.shift(756).rolling(252).sum()
    cagr = (annual_now / annual_3y_ago.replace(0, np.nan)) ** (1 / 3) - 1
    return cagr.rename("div_growth_3y")


# ---------------------------------------------------------------------------
# COT / positioning signals
# ---------------------------------------------------------------------------

@characteristic("cot_z")
def cot_z(cot_df: pd.DataFrame, window: int = 52) -> pd.Series:
    """Z-score of net speculative position (from COT data)."""
    from maroczy.features.positioning import cot_z_score
    return cot_z_score(cot_df, window=window)


@characteristic("cot_mom")
def cot_mom(cot_df: pd.DataFrame, periods: int = 4) -> pd.Series:
    """4-week change in speculator ratio."""
    from maroczy.features.positioning import cot_momentum
    return cot_momentum(cot_df, periods=periods)


# ---------------------------------------------------------------------------
# FX / currency signals
# ---------------------------------------------------------------------------

@characteristic("fx_carry")
def fx_carry(domestic_rate: pd.Series, foreign_rate: pd.Series) -> pd.Series:
    """FX carry: domestic - foreign interest rate differential."""
    return (domestic_rate - foreign_rate).rename("fx_carry")


@characteristic("fx_momentum_3m")
def fx_momentum_3m(fx_price: pd.Series) -> pd.Series:
    """3-month FX momentum (return)."""
    return fx_price.pct_change(63).rename("fx_momentum_3m")


@characteristic("fx_ppp_deviation")
def fx_ppp_deviation(real_exchange_rate: pd.Series, window: int = 1260) -> pd.Series:
    """PPP deviation: real exchange rate vs its 5-year mean (mean-reversion signal)."""
    mean = real_exchange_rate.rolling(window).mean()
    std = real_exchange_rate.rolling(window).std()
    return ((real_exchange_rate - mean) / std.replace(0, np.nan)).rename("fx_ppp_deviation")
