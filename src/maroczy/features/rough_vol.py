"""Rough log-normal volatility: Hurst estimation & rough Bergomi / microstructure simulation.

Implements the core tools from:

    Hager, P. P., Horst, U., Wagenhofer, T., & Xu, W. (2026).
    "Microstructural Foundation of Rough Log-Normal Volatility Models."

* :func:`estimate_hurst` -- scaling-law ("Volatility is Rough", Gatheral,
  Jaisson & Rosenbaum 2018) Hurst-exponent estimator from realized
  log-volatility, used throughout the rough-vol literature including this
  paper's Section 1.
* :func:`simulate_fbm` -- exact (Cholesky) simulation of standard fractional
  Brownian motion, ``E[B^H_t B^H_s] = 1/2(|t|^{2H}+|s|^{2H}-|t-s|^{2H})``.
* :func:`simulate_riemann_liouville_fbm` -- Volterra/Riemann-Liouville fBM,
  ``B^H_t = sqrt(2H) * int_0^t (t-s)^{H-1/2} dZ_s``, matching the paper's own
  numerical illustration (Section 2.3, "our limiting volatility corresponds
  to a Riemann-Liouville fBM").
* :func:`simulate_rough_bergomi` -- the rough Bergomi price-volatility model,
  ``dS_t/S_t = sqrt(v_t) dW_t``, ``v_t = xi0 * exp(eta*B^H_t - 0.5 eta^2 t^{2H})``.
* :func:`simulate_microstructure` -- the *Poisson order-flow microstructure
  prelimit model* of Definition 2.3 in the paper: order arrivals via a
  Poisson process with a slowly-decaying kernel ``phi_n(t) = (1/n+t)^{H-1/2}``
  whose scaling limit is exactly the rough Bergomi model above.
* :func:`theoretical_weak_rate` -- the weak-convergence rate of the
  microstructure model to its Bergomi limit (Theorem 2.8 of the paper),
  handy as a Monte-Carlo convergence sanity check.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

__all__ = [
    "estimate_hurst",
    "simulate_fbm",
    "simulate_riemann_liouville_fbm",
    "simulate_rough_bergomi",
    "simulate_microstructure",
    "theoretical_weak_rate",
]


# ---------------------------------------------------------------------------
# Hurst exponent estimation
# ---------------------------------------------------------------------------

def estimate_hurst(
    log_vol: pd.Series | np.ndarray,
    lags: Iterable[int] = range(1, 21),
    q: float = 2.0,
) -> tuple[float, pd.DataFrame]:
    """Scaling-law Hurst estimator: regress ``log E|log-vol_{t+lag} - log-vol_t|^q`` on ``log(lag)``.

    Under a fractional-Gaussian-like log-volatility process the slope of
    this regression equals ``q * H`` (Gatheral, Jaisson & Rosenbaum, 2018).

    Returns
    -------
    (H, diagnostics) where ``diagnostics`` has columns ``lag`` and ``m``
    (the empirical ``q``-th absolute moment) for plotting the scaling law.
    """
    series = pd.Series(log_vol).dropna()
    lags = np.array(list(lags))
    m = np.array([np.nanmean(np.abs(series.diff(int(lag)).dropna()) ** q) for lag in lags])
    valid = (m > 0) & np.isfinite(m)
    if valid.sum() < 2:
        raise ValueError("Not enough valid lags to estimate the Hurst exponent.")
    slope, _intercept = np.polyfit(np.log(lags[valid]), np.log(m[valid]), 1)
    H = float(slope / q)
    diagnostics = pd.DataFrame({"lag": lags, "m": m})
    return H, diagnostics


# ---------------------------------------------------------------------------
# fBM simulation
# ---------------------------------------------------------------------------

def simulate_fbm(
    n_steps: int,
    H: float,
    T: float = 1.0,
    n_paths: int = 1,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Exact simulation of standard fractional Brownian motion via Cholesky decomposition.

    Returns an array of shape ``(n_paths, n_steps + 1)`` (path starts at 0).
    Cost is ``O(n_steps^3)`` (one Cholesky factorization reused across paths)
    so this is best suited to moderate grids (a few thousand steps).
    """
    rng = rng or np.random.default_rng()
    t = np.linspace(0.0, T, n_steps + 1)[1:]
    ti, tj = np.meshgrid(t, t, indexing="ij")
    cov = 0.5 * (ti ** (2 * H) + tj ** (2 * H) - np.abs(ti - tj) ** (2 * H))
    cov += 1e-10 * np.eye(n_steps)
    L = np.linalg.cholesky(cov)
    z = rng.standard_normal((n_steps, n_paths))
    increments = L @ z  # (n_steps, n_paths)
    zeros = np.zeros((1, n_paths))
    path = np.concatenate([zeros, increments], axis=0)
    return path.T  # (n_paths, n_steps + 1)


def _rl_kernel(t: np.ndarray, H: float) -> np.ndarray:
    """Lower-triangular Volterra kernel ``(t_k - t_j)^{H-1/2}`` for ``j < k``, else 0."""
    ti, tj = np.meshgrid(t, t, indexing="ij")
    diff = ti - tj
    mask = diff > 0
    safe_diff = np.where(mask, diff, 1.0)
    return np.where(mask, safe_diff ** (H - 0.5), 0.0)


def simulate_riemann_liouville_fbm(
    n_steps: int,
    H: float,
    T: float = 1.0,
    n_paths: int = 1,
    rng: np.random.Generator | None = None,
    return_driver: bool = False,
):
    """Riemann-Liouville fBM: ``B^H_t = sqrt(2H) int_0^t (t-s)^{H-1/2} dZ_s``.

    Vectorized Riemann-sum discretization of the Volterra representation;
    this is the process used in the paper's own numerical simulations
    (Section 2.3) and is the natural driver for :func:`simulate_rough_bergomi`.

    Parameters
    ----------
    return_driver: also return the underlying Brownian increments ``dZ``
        (needed to build a correlated price Brownian motion ``W``).
    """
    rng = rng or np.random.default_rng()
    dt = T / n_steps
    t = np.linspace(dt, T, n_steps)
    kernel = _rl_kernel(t, H)  # (n_steps, n_steps)
    dZ = rng.standard_normal((n_paths, n_steps)) * np.sqrt(dt)
    B_H = np.sqrt(2 * H) * (dZ @ kernel.T)
    if return_driver:
        return B_H, dZ, t
    return B_H, t


def simulate_rough_bergomi(
    n_steps: int,
    n_paths: int,
    H: float,
    eta: float,
    rho: float,
    xi0: float = 0.04,
    T: float = 1.0,
    rng: np.random.Generator | None = None,
) -> dict:
    """Rough Bergomi price-volatility simulation (Bayer, Friz & Gatheral, 2016 style).

    ``v_t = xi0 * exp(eta * B^H_t - 0.5 * eta^2 * t^{2H})``,
    ``dS_t/S_t = sqrt(v_t) dW_t`` with ``corr(dW, dZ) = rho`` where ``Z``
    drives the Riemann-Liouville fBM ``B^H``.

    Returns
    -------
    dict with keys ``t``, ``S`` (n_paths x n_steps), ``v``, ``B_H``.
    """
    rng = rng or np.random.default_rng()
    dt = T / n_steps
    B_H, dZ, t = simulate_riemann_liouville_fbm(n_steps, H, T=T, n_paths=n_paths, rng=rng, return_driver=True)
    dW_indep = rng.standard_normal((n_paths, n_steps)) * np.sqrt(dt)
    dW = rho * dZ + np.sqrt(max(1 - rho**2, 0.0)) * dW_indep

    v = xi0 * np.exp(eta * B_H - 0.5 * eta**2 * t ** (2 * H))
    log_ret = np.sqrt(v) * dW - 0.5 * v * dt
    log_S = np.cumsum(log_ret, axis=1)
    S = np.exp(log_S)
    return {"t": t, "S": S, "v": v, "B_H": B_H}


# ---------------------------------------------------------------------------
# Poisson microstructure prelimit model (Definition 2.3)
# ---------------------------------------------------------------------------

def simulate_microstructure(
    n: int,
    T: float = 1.0,
    H: float = 0.15,
    sigma_p: float = 1.0,
    sigma_v: float = 0.25,
    rho: float = -0.7,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Poisson order-flow microstructure model whose scaling limit is rough Bergomi.

    Orders arrive as a unit-rate Poisson process (rescaled by ``n``); each
    order shifts the log-price by ``J_k`` (times current volatility) and the
    log-volatility by a slowly-decaying kernel contribution
    ``phi_n(t) = (n^{-1} + t)^{H-1/2}`` (Eq. 2.7 of the paper). As
    ``n -> infinity`` the rescaled process converges weakly to the rough
    Bergomi model of :func:`simulate_rough_bergomi` (Theorem 2.7). Matches
    the paper's own numerical simulation setup (Section 2.3): no volatility
    impact from orders arriving before time zero.

    Parameters
    ----------
    n: scaling/intensity parameter (paper's ``n``); larger ``n`` -> closer
        to the continuum rough Bergomi limit but more arrivals to simulate
        (O(n*T) arrivals, O((nT)^2) compute/memory).
    T: time horizon.
    H: Hurst parameter, ``H in (0, 1/2)``.
    sigma_p, sigma_v: std devs of the price/volatility jump distributions
        (``J_k``, ``xi_k``); jointly Gaussian, centered, no skew, correlated
        via ``rho`` (Assumption 2.1 / Example 2.2 of the paper).
    rng: numpy random Generator.

    Returns
    -------
    DataFrame indexed by arrival (macro) time with columns
    ``log_price`` and ``log_vol``.
    """
    rng = rng or np.random.default_rng()
    lam = n * T
    n_arrivals = rng.poisson(lam)
    if n_arrivals == 0:
        return pd.DataFrame(columns=["log_price", "log_vol"])

    micro_times = np.sort(rng.uniform(0.0, n * T, n_arrivals))
    macro_times = micro_times / n

    cov = np.array(
        [
            [sigma_p**2, rho * sigma_p * sigma_v],
            [rho * sigma_p * sigma_v, sigma_v**2],
        ]
    )
    jumps = rng.multivariate_normal([0.0, 0.0], cov, size=n_arrivals)
    J, xi = jumps[:, 0], jumps[:, 1]

    diff = macro_times[:, None] - macro_times[None, :]
    strict_mask = np.tril(np.ones((n_arrivals, n_arrivals), dtype=bool), k=-1)
    incl_mask = np.tril(np.ones((n_arrivals, n_arrivals), dtype=bool), k=0)
    safe_diff = np.where(incl_mask, diff, 0.0)
    phi = np.where(incl_mask, (1.0 / n + safe_diff) ** (H - 0.5), 0.0)

    log_vol_pre = (phi * strict_mask) @ xi / np.sqrt(n)
    log_vol = (phi * incl_mask) @ xi / np.sqrt(n)
    log_price = np.cumsum(J * np.exp(log_vol_pre)) / np.sqrt(n)

    return pd.DataFrame({"log_price": log_price, "log_vol": log_vol}, index=pd.Index(macro_times, name="t"))


def theoretical_weak_rate(H: float) -> float:
    """Weak-convergence rate of the microstructure model to its Bergomi limit (Theorem 2.8).

    Returns the exponent ``r`` such that the N-th moment weak error decays
    like ``n^{-r}`` (up to a ``log(n)`` factor exactly at ``H = 1/4``).
    """
    if H < 0.25:
        return 1.0 / 3.0 + 4.0 * H / (3.0 - 6.0 * H)
    return 1.0  # exact at H=1/4 up to a log(n) factor; exactly 1 for H in (1/4, 1/2)
