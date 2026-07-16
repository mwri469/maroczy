"""Options-derived characteristics.

Signal-ready outputs computed from an options chain DataFrame. Functions
accept a pre-fetched chain (from ``lse.options_chain()``) and spot price,
returning a single scalar or Series value suitable for cross-sectional ranking.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from maroczy.characteristics.functions import characteristic


def _normalize_chain(chain_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize common column name variants."""
    col_map = {"impliedVolatility": "iv", "implied_volatility": "iv",
               "openInterest": "open_interest", "openinterest": "open_interest"}
    return chain_df.rename(columns={k: v for k, v in col_map.items() if k in chain_df.columns})


@characteristic("iv_atm")
def iv_atm(chain_df: pd.DataFrame, spot: float) -> float:
    """ATM implied volatility (nearest-strike, shortest expiry with DTE > 14)."""
    df = _normalize_chain(chain_df)
    df = df[df["dte"] > 14]
    if df.empty:
        return np.nan
    nearest_dte = df["dte"].min()
    subset = df[df["dte"] == nearest_dte]
    idx = (subset["strike"] - spot).abs().idxmin()
    return float(subset.loc[idx, "iv"])


@characteristic("iv_skew_25d")
def iv_skew_25d(chain_df: pd.DataFrame, spot: float) -> float:
    """25-delta put-call IV skew: IV(25Δ put) - IV(25Δ call)."""
    df = _normalize_chain(chain_df)
    df = df[df["dte"].between(20, 45)]
    if df.empty:
        return np.nan
    dte = df["dte"].mode().iloc[0] if not df["dte"].mode().empty else df["dte"].median()
    subset = df[(df["dte"] - dte).abs() <= 3]

    puts = subset[subset["type"].str.lower().str.startswith("p")].sort_values("strike")
    calls = subset[subset["type"].str.lower().str.startswith("c")].sort_values("strike")

    # 25-delta approximation: ~10% OTM
    put_strike = spot * 0.90
    call_strike = spot * 1.10

    if puts.empty or calls.empty:
        return np.nan

    put_iv = puts.iloc[(puts["strike"] - put_strike).abs().argsort().iloc[0]]["iv"]
    call_iv = calls.iloc[(calls["strike"] - call_strike).abs().argsort().iloc[0]]["iv"]
    return float(put_iv - call_iv)


@characteristic("iv_skew_slope")
def iv_skew_slope(chain_df: pd.DataFrame, spot: float) -> float:
    """Linear slope of IV on moneyness (K/S) for nearest expiry."""
    df = _normalize_chain(chain_df)
    df = df[df["dte"].between(20, 45)]
    if df.empty or len(df) < 4:
        return np.nan
    dte = df["dte"].min()
    subset = df[df["dte"] == dte].copy()
    subset["moneyness"] = subset["strike"] / spot
    subset = subset[(subset["moneyness"] > 0.8) & (subset["moneyness"] < 1.2)]
    if len(subset) < 3:
        return np.nan
    # OLS slope
    x = subset["moneyness"].to_numpy()
    y = subset["iv"].to_numpy()
    slope = np.polyfit(x, y, 1)[0]
    return float(slope)


@characteristic("iv_term_slope")
def iv_term_slope(chain_df: pd.DataFrame, spot: float) -> float:
    """Term structure slope: ATM IV at ~6 months minus ATM IV at ~1 month."""
    df = _normalize_chain(chain_df)
    short_term = df[df["dte"].between(20, 40)]
    long_term = df[df["dte"].between(150, 200)]

    def _atm_iv(subset):
        if subset.empty:
            return np.nan
        idx = (subset["strike"] - spot).abs().idxmin()
        return float(subset.loc[idx, "iv"])

    iv_short = _atm_iv(short_term)
    iv_long = _atm_iv(long_term)
    if np.isnan(iv_short) or np.isnan(iv_long):
        return np.nan
    return iv_long - iv_short


@characteristic("iv_butterfly")
def iv_butterfly(chain_df: pd.DataFrame, spot: float) -> float:
    """Butterfly spread: 0.5*(IV_25d_put + IV_25d_call) - IV_ATM (smile curvature)."""
    df = _normalize_chain(chain_df)
    df = df[df["dte"].between(20, 45)]
    if df.empty:
        return np.nan
    dte = df["dte"].min()
    subset = df[df["dte"] == dte]

    atm_idx = (subset["strike"] - spot).abs().idxmin()
    iv_atm_val = float(subset.loc[atm_idx, "iv"])

    puts = subset[subset["type"].str.lower().str.startswith("p")]
    calls = subset[subset["type"].str.lower().str.startswith("c")]

    put_target = spot * 0.90
    call_target = spot * 1.10

    if puts.empty or calls.empty:
        return np.nan

    put_iv = float(puts.iloc[(puts["strike"] - put_target).abs().argsort().iloc[0]]["iv"])
    call_iv = float(calls.iloc[(calls["strike"] - call_target).abs().argsort().iloc[0]]["iv"])

    return 0.5 * (put_iv + call_iv) - iv_atm_val


@characteristic("put_call_ratio")
def put_call_ratio(chain_df: pd.DataFrame) -> float:
    """Put/call volume ratio."""
    df = _normalize_chain(chain_df)
    puts = df[df["type"].str.lower().str.startswith("p")]
    calls = df[df["type"].str.lower().str.startswith("c")]
    put_vol = puts["volume"].sum() if "volume" in puts.columns else 0
    call_vol = calls["volume"].sum() if "volume" in calls.columns else 0
    return float(put_vol / call_vol) if call_vol > 0 else np.nan


@characteristic("put_call_oi_ratio")
def put_call_oi_ratio(chain_df: pd.DataFrame) -> float:
    """Put/call open interest ratio."""
    df = _normalize_chain(chain_df)
    puts = df[df["type"].str.lower().str.startswith("p")]
    calls = df[df["type"].str.lower().str.startswith("c")]
    put_oi = puts["open_interest"].sum() if "open_interest" in puts.columns else 0
    call_oi = calls["open_interest"].sum() if "open_interest" in calls.columns else 0
    return float(put_oi / call_oi) if call_oi > 0 else np.nan


@characteristic("options_flow_imbalance")
def options_flow_imbalance(flow_df: pd.DataFrame) -> float:
    """Net call-minus-put premium from options flow data."""
    if flow_df is None or flow_df.empty:
        return np.nan
    df = flow_df.copy()
    premium_col = next((c for c in df.columns if "premium" in c.lower()), None)
    if premium_col is None:
        return np.nan
    calls = df[df["type"].str.lower().str.startswith("c")]
    puts = df[df["type"].str.lower().str.startswith("p")]
    return float(calls[premium_col].sum() - puts[premium_col].sum())


@characteristic("variance_risk_premium")
def variance_risk_premium(chain_df: pd.DataFrame, spot: float, realized_vol: float) -> float:
    """Variance risk premium: implied variance - realized variance."""
    iv = iv_atm(chain_df, spot)
    if np.isnan(iv):
        return np.nan
    return iv**2 - realized_vol**2


@characteristic("rnd_skewness")
def rnd_skewness(chain_df: pd.DataFrame, spot: float, r: float = 0.05) -> float:
    """Risk-neutral skewness from Breeden-Litzenberger density."""
    from maroczy.features.options import fit_iv_surface, risk_neutral_density, implied_pdf_moments
    surface = fit_iv_surface(chain_df, spot)
    if len(surface.expiries) == 0:
        return np.nan
    tau = surface.expiries[0]
    density = risk_neutral_density(surface, spot, r, tau)
    moments = implied_pdf_moments(density)
    return moments["skewness"]


@characteristic("rnd_kurtosis")
def rnd_kurtosis(chain_df: pd.DataFrame, spot: float, r: float = 0.05) -> float:
    """Risk-neutral kurtosis from Breeden-Litzenberger density."""
    from maroczy.features.options import fit_iv_surface, risk_neutral_density, implied_pdf_moments
    surface = fit_iv_surface(chain_df, spot)
    if len(surface.expiries) == 0:
        return np.nan
    tau = surface.expiries[0]
    density = risk_neutral_density(surface, spot, r, tau)
    moments = implied_pdf_moments(density)
    return moments["kurtosis"]


@characteristic("max_pain")
def max_pain(chain_df: pd.DataFrame) -> float:
    """Max pain strike: the strike that minimizes total option holder payout."""
    df = _normalize_chain(chain_df)
    if "open_interest" not in df.columns:
        return np.nan
    strikes = sorted(df["strike"].unique())
    if len(strikes) < 3:
        return np.nan

    calls = df[df["type"].str.lower().str.startswith("c")]
    puts = df[df["type"].str.lower().str.startswith("p")]

    min_pain = np.inf
    best_strike = np.nan
    for s in strikes:
        call_pain = calls.apply(
            lambda row: max(0, s - row["strike"]) * row.get("open_interest", 0), axis=1
        ).sum()
        put_pain = puts.apply(
            lambda row: max(0, row["strike"] - s) * row.get("open_interest", 0), axis=1
        ).sum()
        total = call_pain + put_pain
        if total < min_pain:
            min_pain = total
            best_strike = s

    return float(best_strike)
