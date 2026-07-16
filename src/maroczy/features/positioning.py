"""Futures positioning and curve structure analysis.

Tools for analyzing CFTC Commitment of Traders data and futures term
structure (basis, roll yield, contango/backwardation) — fed by
:meth:`maroczy.datafeed.lse.LSEData.cot` and :meth:`~LSEData.candles`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# COT positioning
# ---------------------------------------------------------------------------

def cot_speculator_ratio(cot_df: pd.DataFrame) -> pd.Series:
    """Net speculative positioning ratio: spec_long / (spec_long + spec_short).

    Parameters
    ----------
    cot_df: DataFrame from ``lse.cot()`` with columns including
        long/short positions for speculators (non-commercial traders).
        Expected column names: ``noncomm_long``, ``noncomm_short``
        (or variants like ``noncommercial_long``).
    """
    # Flexible column matching
    long_col = _find_col(cot_df, ["noncomm_long", "noncommercial_long", "spec_long", "money_manager_long"])
    short_col = _find_col(cot_df, ["noncomm_short", "noncommercial_short", "spec_short", "money_manager_short"])
    total = cot_df[long_col] + cot_df[short_col]
    return (cot_df[long_col] / total.replace(0, np.nan)).rename("spec_ratio")


def cot_net_position(cot_df: pd.DataFrame) -> pd.Series:
    """Net speculative position (long - short)."""
    long_col = _find_col(cot_df, ["noncomm_long", "noncommercial_long", "spec_long", "money_manager_long"])
    short_col = _find_col(cot_df, ["noncomm_short", "noncommercial_short", "spec_short", "money_manager_short"])
    return (cot_df[long_col] - cot_df[short_col]).rename("net_spec")


def cot_z_score(cot_df: pd.DataFrame, window: int = 52) -> pd.Series:
    """Z-score of net speculative position over trailing window (weeks)."""
    net = cot_net_position(cot_df)
    mu = net.rolling(window).mean()
    sigma = net.rolling(window).std()
    return ((net - mu) / sigma.replace(0, np.nan)).rename("cot_z")


def cot_momentum(cot_df: pd.DataFrame, periods: int = 4) -> pd.Series:
    """Change in speculator ratio over ``periods`` weeks."""
    ratio = cot_speculator_ratio(cot_df)
    return (ratio - ratio.shift(periods)).rename("cot_momentum")


# ---------------------------------------------------------------------------
# Futures term structure
# ---------------------------------------------------------------------------

def futures_basis(front_price: pd.Series, spot_price: pd.Series, annualize: bool = True, days_to_expiry: int = 30) -> pd.Series:
    """Futures basis: (front - spot) / spot, optionally annualized.

    Parameters
    ----------
    front_price: front-month futures price series.
    spot_price: spot/cash price series.
    annualize: if True, scale to annual rate assuming ``days_to_expiry``.
    """
    basis = (front_price - spot_price) / spot_price.replace(0, np.nan)
    if annualize:
        basis = basis * (365.25 / days_to_expiry)
    return basis.rename("basis")


def roll_yield(front_price: pd.Series, back_price: pd.Series) -> pd.Series:
    """Roll yield / calendar spread: (back - front) / front.

    Positive = contango (you earn roll yield by shorting back, buying front).
    Negative = backwardation.
    """
    return ((back_price - front_price) / front_price.replace(0, np.nan)).rename("roll_yield")


def contango_backwardation(front_price: pd.Series, back_price: pd.Series) -> pd.Series:
    """Curve shape indicator: +1 = contango, -1 = backwardation."""
    return np.sign(back_price - front_price).rename("curve_shape")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_col(df: pd.DataFrame, candidates: list[str]) -> str:
    """Find first matching column name from candidates."""
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    raise KeyError(f"None of {candidates} found in columns: {list(df.columns)}")
