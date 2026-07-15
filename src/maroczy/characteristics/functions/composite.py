"""Composite characteristics combining price and fundamental data."""

from __future__ import annotations

import pandas as pd

from maroczy.characteristics.functions import characteristic


@characteristic("age")
def age(price_history: pd.DataFrame) -> pd.Series:
    """Jiang, Lee & Zhang (2005): firm age in years since the first observed price."""
    first_date = price_history.index[0]
    days = (price_history.index - first_date).days
    return pd.Series(days / 365.25, index=price_history.index, name="age")


def _cross_sectional_zscore(panel: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional (row-wise) z-score of each column, winsorized at +/-3."""
    z = (panel - panel.mean()) / panel.std(ddof=0)
    return z.clip(-3, 3)


@characteristic("qmj_prof")
def qmj_prof(profitability_components: pd.DataFrame) -> pd.Series:
    """Assness, Frazzini & Pedersen (2018): profitability sub-score.

    ``profitability_components``: DataFrame indexed by symbol with one
    column per raw profitability ratio (e.g. gpoa, roe, roa, cfoa, gmar,
    accruals); each column is cross-sectionally z-scored and averaged.
    """
    return _cross_sectional_zscore(profitability_components).mean(axis=1)


@characteristic("qmj_growth")
def qmj_growth(growth_components: pd.DataFrame) -> pd.Series:
    """Assness, Frazzini & Pedersen (2018): growth sub-score (5y change in profitability ratios)."""
    return _cross_sectional_zscore(growth_components).mean(axis=1)


@characteristic("qmj_safety")
def qmj_safety(safety_components: pd.DataFrame) -> pd.Series:
    """Assness, Frazzini & Pedersen (2018): safety sub-score.

    Orient inputs so *larger = safer* (e.g. pass ``-beta``, ``-leverage``,
    ``-earnings_volatility``) before calling.
    """
    return _cross_sectional_zscore(safety_components).mean(axis=1)


@characteristic("qmj")
def qmj(profitability: pd.Series, growth: pd.Series, safety: pd.Series) -> pd.Series:
    """Assness, Frazzini & Pedersen (2018): Quality-minus-Junk composite.

    ``quality = z(profitability) + z(growth) + z(safety)``; inputs are
    typically the outputs of :func:`qmj_prof`, :func:`qmj_growth` and
    :func:`qmj_safety` for one cross-section (indexed by symbol).
    """
    frame = pd.concat({"profitability": profitability, "growth": growth, "safety": safety}, axis=1)
    return _cross_sectional_zscore(frame).sum(axis=1)
