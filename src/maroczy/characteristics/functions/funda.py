"""Annual fundamentals characteristics (CSV ``class == "funda"``).

Functions operate on a user-supplied fundamentals DataFrame indexed by
fiscal-period-end date, using standard Compustat-style column names (``at``
= total assets, ``ceq`` = common equity, ``sale`` = sales/revenue, ``ni`` =
net income, ``dltt``/``dlc`` = long/short-term debt, ``ebit``, ``che`` =
cash & equivalents, ``act``/``lct`` = current assets/liabilities, ``lt`` =
total liabilities, ``re`` = retained earnings, ``capx`` = capex). Functions
needing market equity accept ``me: pd.Series`` (e.g. from
:func:`maroczy.characteristics.functions.crspm.market_equity`).

This is a deliberately small, high-value core subset -- extend by adding
more ``@characteristic``-decorated functions following the same pattern.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from maroczy.characteristics.functions import characteristic


def _get(funda: pd.DataFrame, col: str) -> pd.Series:
    if col not in funda.columns:
        raise KeyError(f"Fundamentals frame is missing required column {col!r}.")
    return funda[col]


@characteristic("at_gr1")
def at_gr1(funda: pd.DataFrame) -> pd.Series:
    """Cooper, Gulen & Schill (2008): asset growth."""
    return _get(funda, "at").pct_change()


@characteristic("sale_gr1")
def sale_gr1(funda: pd.DataFrame) -> pd.Series:
    """Lakonishok, Shleifer & Vishny (1994): annual sales growth."""
    return _get(funda, "sale").pct_change()


@characteristic("sale_gr3")
def sale_gr3(funda: pd.DataFrame) -> pd.Series:
    """Lakonishok, Shleifer & Vishny (1994): three-year sales growth."""
    sale = _get(funda, "sale")
    return sale / sale.shift(3) - 1


@characteristic("capx_gr2")
def capx_gr2(funda: pd.DataFrame) -> pd.Series:
    """Anderson & Garcia-Feijoo (2006): two-year capex growth."""
    capx = _get(funda, "capx")
    return capx / capx.shift(2) - 1


@characteristic("capx_gr3")
def capx_gr3(funda: pd.DataFrame) -> pd.Series:
    """Anderson & Garcia-Feijoo (2006): three-year capex growth."""
    capx = _get(funda, "capx")
    return capx / capx.shift(3) - 1


@characteristic("ni_me")
def ni_me(funda: pd.DataFrame, me: pd.Series) -> pd.Series:
    """Basu (1977): earnings to price (``ni / market equity``)."""
    return _get(funda, "ni") / me


@characteristic("debt_me")
def debt_me(funda: pd.DataFrame, me: pd.Series) -> pd.Series:
    """Bhandari (1988): debt to market (``(dltt + dlc) / market equity``)."""
    debt = funda.get("dltt", 0).fillna(0) + funda.get("dlc", 0).fillna(0)
    return debt / me


@characteristic("at_me")
def at_me(funda: pd.DataFrame, me: pd.Series) -> pd.Series:
    """Fama & French (1992): assets to market."""
    return _get(funda, "at") / me


@characteristic("fcf_me")
def fcf_me(funda: pd.DataFrame, me: pd.Series) -> pd.Series:
    """Lakonishok, Shleifer & Vishny (1994): free cash flow to price."""
    ocf = funda.get("oancf")
    capx = funda.get("capx", 0)
    if ocf is None:
        raise KeyError("Fundamentals frame is missing required column 'oancf'.")
    return (ocf - capx.fillna(0)) / me


@characteristic("op_at")
def op_at(funda: pd.DataFrame) -> pd.Series:
    """Ball et al. (2016): operating profitability to assets, ``(sale - cogs - xsga) / at``."""
    sale = _get(funda, "sale")
    cogs = funda.get("cogs", 0).fillna(0)
    xsga = funda.get("xsga", 0).fillna(0)
    return (sale - cogs - xsga) / _get(funda, "at")


@characteristic("roic")
def roic(funda: pd.DataFrame) -> pd.Series:
    """Brown & Rowe (2007): return on invested capital, ``ebit / (debt + equity - cash)``."""
    debt = funda.get("dltt", 0).fillna(0) + funda.get("dlc", 0).fillna(0)
    invested = debt + _get(funda, "ceq") - funda.get("che", 0).fillna(0)
    return _get(funda, "ebit") / invested.replace(0, np.nan)


@characteristic("z_score")
def z_score(funda: pd.DataFrame, me: pd.Series) -> pd.Series:
    """Dichev (1998): Altman Z-score (simplified 5-factor formulation)."""
    at_ = _get(funda, "at")
    wc = funda.get("act", np.nan) - funda.get("lct", np.nan)
    re = funda.get("re", np.nan)
    ebit = _get(funda, "ebit")
    sale = _get(funda, "sale")
    lt = _get(funda, "lt")
    return 1.2 * wc / at_ + 1.4 * re / at_ + 3.3 * ebit / at_ + 0.6 * me / lt + 1.0 * sale / at_


@characteristic("tangibility")
def tangibility(funda: pd.DataFrame) -> pd.Series:
    """Hahn & Lee (2009): asset tangibility, ``(che + 0.715*rect + 0.547*invt + 0.535*ppent) / at``."""
    che = funda.get("che", 0).fillna(0)
    rect = funda.get("rect", 0).fillna(0)
    invt = funda.get("invt", 0).fillna(0)
    ppent = funda.get("ppent", 0).fillna(0)
    return (che + 0.715 * rect + 0.547 * invt + 0.535 * ppent) / _get(funda, "at")


@characteristic("noa_at")
def noa_at(funda: pd.DataFrame) -> pd.Series:
    """Hirshleifer et al. (2004): net operating assets to total assets."""
    oa = funda.get("act", 0).fillna(0) + _get(funda, "at") - funda.get("che", 0).fillna(0) - funda.get("ivao", 0).fillna(0)
    ol = _get(funda, "at") - funda.get("dlc", 0).fillna(0) - funda.get("dltt", 0).fillna(0) - funda.get("mib", 0).fillna(0) - funda.get("pstk", 0).fillna(0) - _get(funda, "ceq")
    return (oa - ol) / _get(funda, "at").shift(1)
