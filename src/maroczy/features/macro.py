"""Macro/rates research tools: yield curve analysis, economic surprise, regime detection.

Provides tools for decomposing yield curves, constructing economic surprise
indices, and detecting macro regime shifts — all fed by
:meth:`maroczy.datafeed.lse.LSEData.bond_yields`,
:meth:`~LSEData.economics`, and :meth:`~LSEData.economic_calendar`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize


# ---------------------------------------------------------------------------
# Yield curve
# ---------------------------------------------------------------------------

US_TENORS = {
    "US1M": 1 / 12, "US3M": 0.25, "US6M": 0.5, "US1Y": 1.0,
    "US2Y": 2.0, "US3Y": 3.0, "US5Y": 5.0, "US7Y": 7.0,
    "US10Y": 10.0, "US20Y": 20.0, "US30Y": 30.0,
}


def yield_curve_snapshot(yields_dict: dict[str, float]) -> pd.Series:
    """Build a yield curve Series from a tenor → yield mapping.

    Parameters
    ----------
    yields_dict: e.g. {"US1M": 5.2, "US2Y": 4.8, "US10Y": 4.3, ...}

    Returns
    -------
    Series indexed by maturity (years) → yield (%).
    """
    maturities = []
    values = []
    for tenor, yld in yields_dict.items():
        mat = US_TENORS.get(tenor, None)
        if mat is not None:
            maturities.append(mat)
            values.append(yld)
    s = pd.Series(values, index=maturities, name="yield").sort_index()
    return s


def nelson_siegel_fit(maturities: np.ndarray, yields: np.ndarray) -> dict[str, float]:
    """Fit Nelson-Siegel model: y(tau) = beta0 + beta1*(1-exp(-tau/lam))/(tau/lam) + beta2*((1-exp(-tau/lam))/(tau/lam) - exp(-tau/lam)).

    Returns dict with keys: level (beta0), slope (beta1), curvature (beta2), tau (lambda).
    """
    maturities = np.asarray(maturities, dtype=float)
    yields = np.asarray(yields, dtype=float)

    def _ns(params):
        b0, b1, b2, lam = params
        if lam <= 0:
            return 1e10
        x = maturities / lam
        factor1 = (1 - np.exp(-x)) / x
        factor2 = factor1 - np.exp(-x)
        fitted = b0 + b1 * factor1 + b2 * factor2
        return np.sum((yields - fitted) ** 2)

    res = minimize(_ns, x0=[yields.mean(), -1.0, 0.5, 2.0], method="Nelder-Mead")
    b0, b1, b2, lam = res.x
    return {"level": b0, "slope": b1, "curvature": b2, "tau": lam}


def pca_curve_factors(yield_history: pd.DataFrame, n_factors: int = 3) -> pd.DataFrame:
    """PCA decomposition of yield curve changes.

    Parameters
    ----------
    yield_history: DataFrame indexed by date, columns = tenors/maturities,
        values = yield levels.
    n_factors: number of PCs to extract.

    Returns
    -------
    DataFrame indexed by date with columns PC1, PC2, PC3, ... representing
    level, slope, and curvature factors respectively.
    """
    changes = yield_history.diff().dropna()
    # Standardize
    mu = changes.mean()
    sigma = changes.std().replace(0, 1)
    standardized = (changes - mu) / sigma

    # SVD
    U, s, Vt = np.linalg.svd(standardized.values, full_matrices=False)
    scores = U[:, :n_factors] * s[:n_factors]

    cols = [f"PC{i+1}" for i in range(n_factors)]
    return pd.DataFrame(scores, index=changes.index, columns=cols)


# ---------------------------------------------------------------------------
# Economic surprise
# ---------------------------------------------------------------------------

def economic_surprise_index(calendar_df: pd.DataFrame, window: int = 90) -> pd.Series:
    """Citigroup-style economic surprise index.

    Parameters
    ----------
    calendar_df: DataFrame with columns ``date``, ``actual``, ``consensus``
        (or ``forecast``). Rows are individual economic releases.
    window: rolling window in days for exponential decay.

    Returns
    -------
    Series indexed by date → rolling cumulative surprise.
    """
    df = calendar_df.copy()
    # Normalize column names
    if "forecast" in df.columns and "consensus" not in df.columns:
        df = df.rename(columns={"forecast": "consensus"})

    df["surprise"] = df["actual"] - df["consensus"]
    # Normalize by historical std of surprise for each event
    df["surprise_z"] = df.groupby(df.get("event", "all"))["surprise"].transform(
        lambda x: x / x.std() if x.std() > 0 else 0
    )

    # Aggregate to daily and compute rolling ESI
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        daily = df.groupby("date")["surprise_z"].sum().sort_index()
    else:
        daily = df["surprise_z"]

    return daily.ewm(span=window).mean().rename("economic_surprise_index")


# ---------------------------------------------------------------------------
# Recession probability
# ---------------------------------------------------------------------------

def recession_probability(spread_10y_2y: pd.Series) -> pd.Series:
    """Probit recession probability from the 10Y-2Y yield spread.

    Based on Estrella & Mishkin (1998): P(recession in 12 months) = Φ(-0.6 - 0.8 * spread).
    """
    from scipy.stats import norm as _norm
    return pd.Series(
        _norm.cdf(-0.6 - 0.8 * spread_10y_2y.values),
        index=spread_10y_2y.index,
        name="recession_prob_12m",
    )


# ---------------------------------------------------------------------------
# Regime detection
# ---------------------------------------------------------------------------

def regime_filter(series: pd.Series, n_regimes: int = 2, window: int = 60) -> pd.Series:
    """Simple regime detection via rolling mean/vol thresholds.

    Assigns regime labels (0 = low-vol/expansion, 1 = high-vol/contraction)
    based on whether rolling volatility exceeds its long-run median.
    For a proper HMM, use ``hmmlearn`` externally.
    """
    vol = series.rolling(window).std()
    median_vol = vol.expanding().median()
    regime = (vol > median_vol).astype(int)
    return regime.rename("regime")
