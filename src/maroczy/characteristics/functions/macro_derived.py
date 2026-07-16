"""Macro-derived characteristics.

Signal-ready macro overlay outputs for equity positioning. Functions accept
pre-fetched yield/economics DataFrames and return single-series signals.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from maroczy.characteristics.functions import characteristic


@characteristic("yield_curve_slope")
def yield_curve_slope(yields: pd.DataFrame) -> pd.Series:
    """10Y-2Y yield spread.

    Parameters
    ----------
    yields: DataFrame with columns for different tenors (e.g. ``US10Y``, ``US2Y``)
        or a 2-column DataFrame with 10Y and 2Y series.
    """
    if "US10Y" in yields.columns and "US2Y" in yields.columns:
        return (yields["US10Y"] - yields["US2Y"]).rename("yield_curve_slope")
    # Fallback: assume first col is long, second is short
    return (yields.iloc[:, 0] - yields.iloc[:, 1]).rename("yield_curve_slope")


@characteristic("yield_curve_curvature")
def yield_curve_curvature(yields: pd.DataFrame) -> pd.Series:
    """Yield curve butterfly: 2*5Y - 2Y - 10Y."""
    y2 = yields.get("US2Y", yields.iloc[:, 0])
    y5 = yields.get("US5Y", yields.iloc[:, 1] if yields.shape[1] > 2 else yields.iloc[:, 0])
    y10 = yields.get("US10Y", yields.iloc[:, -1])
    return (2 * y5 - y2 - y10).rename("yield_curve_curvature")


@characteristic("real_rate_10y")
def real_rate_10y(nominal_10y: pd.Series, breakeven_10y: pd.Series) -> pd.Series:
    """Real 10Y yield: nominal - breakeven inflation."""
    return (nominal_10y - breakeven_10y).rename("real_rate_10y")


@characteristic("credit_spread")
def credit_spread(corporate_yield: pd.Series, treasury_yield: pd.Series) -> pd.Series:
    """Credit spread: corporate - treasury yield."""
    return (corporate_yield - treasury_yield).rename("credit_spread")


@characteristic("term_premium")
def term_premium(long_yield: pd.Series, short_yield: pd.Series, window: int = 252) -> pd.Series:
    """Term premium proxy: long yield - rolling average of short yield (expectations proxy)."""
    expected_path = short_yield.rolling(window).mean()
    return (long_yield - expected_path).rename("term_premium")


@characteristic("economic_surprise")
def economic_surprise(calendar_df: pd.DataFrame, window: int = 90) -> pd.Series:
    """Rolling economic surprise index (actual - consensus, exponentially weighted)."""
    from maroczy.features.macro import economic_surprise_index
    return economic_surprise_index(calendar_df, window=window)


@characteristic("cpi_momentum")
def cpi_momentum(cpi_series: pd.Series) -> pd.Series:
    """CPI momentum: change in year-over-year CPI (acceleration in inflation)."""
    yoy = cpi_series.pct_change(12) if len(cpi_series) > 12 else cpi_series.pct_change()
    return yoy.diff().rename("cpi_momentum")


@characteristic("fed_funds_gap")
def fed_funds_gap(market_implied_rate: pd.Series, actual_rate: pd.Series) -> pd.Series:
    """Gap between market-implied fed funds rate and actual."""
    return (market_implied_rate - actual_rate).rename("fed_funds_gap")


@characteristic("financial_conditions")
def financial_conditions(credit_spread_s: pd.Series, equity_vol: pd.Series, yield_slope: pd.Series) -> pd.Series:
    """Simplified financial conditions index (z-scored composite)."""
    def _z(s):
        return (s - s.expanding().mean()) / s.expanding().std().replace(0, np.nan)

    # Higher spread = tighter, higher vol = tighter, inverted curve = tighter
    fci = _z(credit_spread_s) + _z(equity_vol) - _z(yield_slope)
    return (fci / 3).rename("financial_conditions")


@characteristic("recession_prob")
def recession_prob(spread_10y_2y: pd.Series) -> pd.Series:
    """12-month recession probability from the 10Y-2Y spread (Estrella & Mishkin probit)."""
    from maroczy.features.macro import recession_probability
    return recession_probability(spread_10y_2y)
