"""Options surface analysis: IV fitting, risk-neutral densities, term structure.

Provides research tools for extracting structured information from raw
options chain data (as returned by :meth:`maroczy.datafeed.lse.LSEData.options_chain`).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline
from scipy.stats import norm


# ---------------------------------------------------------------------------
# Black-Scholes helpers
# ---------------------------------------------------------------------------

def bs_price(S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call") -> float:
    """Black-Scholes European option price."""
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def bs_delta(S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call") -> float:
    """Black-Scholes delta."""
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    if option_type == "call":
        return norm.cdf(d1)
    return norm.cdf(d1) - 1


# ---------------------------------------------------------------------------
# IV Surface
# ---------------------------------------------------------------------------

@dataclass
class IVSurface:
    """Interpolatable implied volatility surface (moneyness × time-to-expiry)."""

    expiries: np.ndarray
    strikes_by_expiry: dict[float, np.ndarray] = field(repr=False)
    iv_by_expiry: dict[float, np.ndarray] = field(repr=False)
    splines: dict[float, UnivariateSpline] = field(repr=False)
    spot: float = 1.0

    def iv(self, strike: float, tau: float) -> float:
        """Interpolate IV at a given strike and time-to-expiry."""
        # Find bracketing expiries and linearly interpolate
        taus = self.expiries
        if tau <= taus[0]:
            return float(self.splines[taus[0]](strike))
        if tau >= taus[-1]:
            return float(self.splines[taus[-1]](strike))
        idx = np.searchsorted(taus, tau)
        t0, t1 = taus[idx - 1], taus[idx]
        w = (tau - t0) / (t1 - t0)
        iv0 = float(self.splines[t0](strike))
        iv1 = float(self.splines[t1](strike))
        return iv0 * (1 - w) + iv1 * w

    def smile(self, tau: float, strikes: np.ndarray | None = None) -> pd.Series:
        """Return IV smile for a given tau."""
        if strikes is None:
            nearest = self.expiries[np.argmin(np.abs(self.expiries - tau))]
            strikes = self.strikes_by_expiry[nearest]
        ivs = np.array([self.iv(k, tau) for k in strikes])
        return pd.Series(ivs, index=strikes, name=f"IV(tau={tau:.3f})")


def fit_iv_surface(chain_df: pd.DataFrame, spot: float, smoothing: float = 0.5) -> IVSurface:
    """Fit a smooth IV surface from a raw options chain DataFrame.

    Parameters
    ----------
    chain_df: DataFrame with columns including ``strike``, ``dte`` (days to
        expiry), ``impliedVolatility`` (or ``iv``), and ``type`` (call/put).
    spot: current underlying price.
    smoothing: spline smoothing factor (0 = interpolate exactly).

    Returns
    -------
    IVSurface with per-expiry spline fits.
    """
    df = chain_df.copy()
    # Normalize column names
    col_map = {"impliedVolatility": "iv", "implied_volatility": "iv"}
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    # Use OTM options for cleaner surface
    calls = df[df["type"].str.lower().str.startswith("c")]
    puts = df[df["type"].str.lower().str.startswith("p")]
    otm_calls = calls[calls["strike"] >= spot]
    otm_puts = puts[puts["strike"] < spot]
    df_otm = pd.concat([otm_calls, otm_puts])

    # Convert DTE to years
    df_otm = df_otm.copy()
    df_otm["tau"] = df_otm["dte"] / 365.25

    expiries_raw = sorted(df_otm["tau"].unique())
    splines = {}
    strikes_by_expiry = {}
    iv_by_expiry = {}

    for tau in expiries_raw:
        subset = df_otm[df_otm["tau"] == tau].sort_values("strike")
        if len(subset) < 4:
            continue
        k = subset["strike"].to_numpy()
        v = subset["iv"].to_numpy()
        # Remove invalid IVs
        mask = (v > 0.01) & (v < 5.0) & np.isfinite(v)
        k, v = k[mask], v[mask]
        if len(k) < 4:
            continue
        s = len(k) * smoothing
        spline = UnivariateSpline(k, v, s=s, ext=3)
        splines[tau] = spline
        strikes_by_expiry[tau] = k
        iv_by_expiry[tau] = v

    expiries = np.array(sorted(splines.keys()))
    return IVSurface(
        expiries=expiries,
        strikes_by_expiry=strikes_by_expiry,
        iv_by_expiry=iv_by_expiry,
        splines=splines,
        spot=spot,
    )


# ---------------------------------------------------------------------------
# Risk-neutral density
# ---------------------------------------------------------------------------

def risk_neutral_density(
    surface: IVSurface, spot: float, r: float, tau: float, n_strikes: int = 200
) -> pd.Series:
    """Breeden-Litzenberger risk-neutral density via ∂²C/∂K².

    Returns a Series indexed by strike with the probability density values.
    """
    k_min = spot * 0.5
    k_max = spot * 1.5
    strikes = np.linspace(k_min, k_max, n_strikes)
    dk = strikes[1] - strikes[0]

    # Price calls across strikes
    prices = np.array([
        bs_price(spot, k, tau, r, max(surface.iv(k, tau), 0.01), "call")
        for k in strikes
    ])

    # Second derivative via finite differences
    d2c_dk2 = np.gradient(np.gradient(prices, dk), dk)
    density = np.exp(r * tau) * d2c_dk2
    density = np.clip(density, 0, None)

    return pd.Series(density, index=strikes, name="risk_neutral_density")


def implied_pdf_moments(density: pd.Series) -> dict[str, float]:
    """Compute moments of a risk-neutral density.

    Returns dict with keys: mean, variance, skewness, kurtosis.
    """
    strikes = density.index.to_numpy()
    pdf = density.to_numpy()
    # Normalize
    total = np.trapz(pdf, strikes)
    if total <= 0:
        return {"mean": np.nan, "variance": np.nan, "skewness": np.nan, "kurtosis": np.nan}
    pdf = pdf / total

    mean = np.trapz(strikes * pdf, strikes)
    var = np.trapz((strikes - mean) ** 2 * pdf, strikes)
    std = np.sqrt(var) if var > 0 else 1e-10
    skew = np.trapz((strikes - mean) ** 3 * pdf, strikes) / std**3
    kurt = np.trapz((strikes - mean) ** 4 * pdf, strikes) / std**4

    return {"mean": mean, "variance": var, "skewness": skew, "kurtosis": kurt}


# ---------------------------------------------------------------------------
# Term structure & skew extraction
# ---------------------------------------------------------------------------

def term_structure(chain_df: pd.DataFrame, spot: float, moneyness: float = 1.0) -> pd.Series:
    """Extract IV term structure at a given moneyness level.

    Parameters
    ----------
    moneyness: K/S ratio (1.0 = ATM, 0.9 = 10% OTM put, 1.1 = 10% OTM call).

    Returns
    -------
    Series indexed by DTE → IV.
    """
    target_strike = spot * moneyness
    df = chain_df.copy()
    col_map = {"impliedVolatility": "iv", "implied_volatility": "iv"}
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    result = {}
    for dte, group in df.groupby("dte"):
        closest_idx = (group["strike"] - target_strike).abs().idxmin()
        result[dte] = group.loc[closest_idx, "iv"]

    return pd.Series(result, name=f"IV_term(m={moneyness})").sort_index()


def skew_smile(chain_df: pd.DataFrame, dte: int) -> pd.Series:
    """Extract the IV smile for a single expiry.

    Returns Series indexed by strike → IV.
    """
    df = chain_df.copy()
    col_map = {"impliedVolatility": "iv", "implied_volatility": "iv"}
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    subset = df[df["dte"] == dte].sort_values("strike")
    return pd.Series(subset["iv"].values, index=subset["strike"].values, name=f"smile(dte={dte})")


def variance_swap_rate(chain_df: pd.DataFrame, spot: float, r: float, dte: int) -> float:
    """Model-free implied variance via the variance swap replication formula.

    Uses the full strip of OTM options to compute fair variance.
    """
    df = chain_df.copy()
    col_map = {"impliedVolatility": "iv", "implied_volatility": "iv"}
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    subset = df[df["dte"] == dte].sort_values("strike")
    if len(subset) < 4:
        return np.nan

    tau = dte / 365.25
    F = spot * np.exp(r * tau)

    # OTM calls (K > F) and OTM puts (K < F)
    calls = subset[(subset["strike"] > F) & (subset["type"].str.lower().str.startswith("c"))]
    puts = subset[(subset["strike"] < F) & (subset["type"].str.lower().str.startswith("p"))]

    var_sum = 0.0
    for _, row in puts.iterrows():
        K = row["strike"]
        price = bs_price(spot, K, tau, r, row["iv"], "put")
        dk = 1.0  # simplified — in production use actual strike spacing
        var_sum += (2 / tau) * price / K**2 * dk

    for _, row in calls.iterrows():
        K = row["strike"]
        price = bs_price(spot, K, tau, r, row["iv"], "call")
        dk = 1.0
        var_sum += (2 / tau) * price / K**2 * dk

    return var_sum
