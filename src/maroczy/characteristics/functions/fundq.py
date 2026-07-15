"""Quarterly fundamentals characteristics (CSV ``class == "fundq"``).

Functions operate on a quarterly fundamentals DataFrame indexed by
fiscal-quarter-end date, using Compustat-style column names (``atq`` =
total assets, ``ibq`` = quarterly income before extraordinary items, ``ceqq``
= common equity, ``saleq`` = quarterly sales, ``epspxq`` = quarterly EPS,
``cshprq`` = shares used for EPS).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from maroczy.characteristics.functions import characteristic


def _get(fundq: pd.DataFrame, col: str) -> pd.Series:
    if col not in fundq.columns:
        raise KeyError(f"Quarterly fundamentals frame is missing required column {col!r}.")
    return fundq[col]


@characteristic("niq_at")
def niq_at(fundq: pd.DataFrame) -> pd.Series:
    """Balakrishnan, Bartov & Faurel (2010): quarterly return on assets."""
    return _get(fundq, "ibq") / _get(fundq, "atq")


@characteristic("niq_at_chg1")
def niq_at_chg1(fundq: pd.DataFrame) -> pd.Series:
    """Balakrishnan, Bartov & Faurel (2010): change in quarterly ROA."""
    return niq_at(fundq).diff()


@characteristic("niq_be")
def niq_be(fundq: pd.DataFrame) -> pd.Series:
    """Hou, Xue & Zhang (2015): quarterly return on equity."""
    return _get(fundq, "ibq") / _get(fundq, "ceqq")


@characteristic("niq_be_chg1")
def niq_be_chg1(fundq: pd.DataFrame) -> pd.Series:
    """Balakrishnan, Bartov & Faurel (2010): change in quarterly ROE."""
    return niq_be(fundq).diff()


@characteristic("niq_su")
def niq_su(fundq: pd.DataFrame, window: int = 8) -> pd.Series:
    """Foster, Olsen & Shevlin (1984): standardized unexpected earnings (SUE).

    Seasonal random-walk surprise: ``(eps_t - eps_{t-4}) / std(eps_t - eps_{t-4})``
    over the trailing ``window`` quarters.
    """
    eps = _get(fundq, "epspxq")
    surprise = eps - eps.shift(4)
    return surprise / surprise.rolling(window).std()


@characteristic("saleq_su")
def saleq_su(fundq: pd.DataFrame, window: int = 8) -> pd.Series:
    """Jegadeesh & Livnat (2006): revenue surprise (seasonal random walk, scaled by price)."""
    sale_per_share = _get(fundq, "saleq") / _get(fundq, "cshprq")
    surprise = sale_per_share - sale_per_share.shift(4)
    return surprise / surprise.rolling(window).std()


@characteristic("roaq")
def roaq(fundq: pd.DataFrame) -> pd.Series:
    """Balakrishnan, Bartov & Faurel (2010): quarterly ROA (alias of ``niq_at``)."""
    return niq_at(fundq)


@characteristic("roavol")
def roavol(fundq: pd.DataFrame, window: int = 16) -> pd.Series:
    """Francis et al. (2004): volatility of quarterly ROA."""
    return niq_at(fundq).rolling(window).std()
