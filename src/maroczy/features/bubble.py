"""Phillips, Shi & Yu (2015) explosive-bubble detection.

Implements the (G)SADF right-tailed unit-root testing framework and the
backward-SADF date-stamping strategy from:

    Phillips, P. C. B., Shi, S., & Yu, J. (2015). "Testing for Multiple
    Bubbles: Historical Episodes of Exuberance and Collapse in the S&P
    500." *International Economic Review*, 56(4), 1043-1078.

Core regression (Eq. 4 in the paper)::

    Delta y_t = alpha + beta * y_{t-1} + sum_i psi_i * Delta y_{t-i} + eps_t

run recursively over sub-samples ``[r1, r2]`` of the (fractional) sample.

* ``SADF``  -- sup of the ADF t-stat over forward-expanding windows with a
  fixed start (``r1 = 0``), Section 2.2.
* ``GSADF`` -- sup of the ADF t-stat over *all* feasible windows (flexible
  start *and* end), Section 2.3, Eq. (5).
* ``BSADF_r2`` -- backward sup ADF at a fixed end point ``r2``, used for
  real-time date-stamping of bubble origination/termination, Section 3.1,
  Eqs. (7)-(8).

Notation note: fractional windows ``r1, r2 in [0, 1]`` are expressed here in
terms of the *differenced* sample of length ``N = T - 1 - lags`` (i.e. after
losing observations to differencing/lagging), which is the convention used
by most practical implementations (e.g. the ``psymonitor`` R package).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

__all__ = [
    "default_r0",
    "adf_stat",
    "sadf",
    "gsadf",
    "bsadf_sequence",
    "critical_values",
    "simulate_critical_values",
    "bsadf_dating",
    "SADFResult",
    "GSADFResult",
]


# ---------------------------------------------------------------------------
# Design matrix & recursive sufficient statistics
# ---------------------------------------------------------------------------

def default_r0(T: int) -> float:
    """Recommended minimum window fraction, ``r0 = 0.01 + 1.8/sqrt(T)`` (Sec. 2.3)."""
    return 0.01 + 1.8 / np.sqrt(T)


def _build_regression_arrays(y: np.ndarray, lags: int) -> tuple[np.ndarray, np.ndarray]:
    """Build design matrix ``X = [1, y_{t-1}, dy_{t-1}, ..., dy_{t-lags}]`` and ``z = dy_t``."""
    s = pd.Series(np.asarray(y, dtype=float))
    dy = s.diff()
    frame = {"y_lag1": s.shift(1), "dy": dy}
    for i in range(1, lags + 1):
        frame[f"dy_lag{i}"] = dy.shift(i)
    df = pd.DataFrame(frame).dropna()
    z = df["dy"].to_numpy()
    cols = ["y_lag1"] + [f"dy_lag{i}" for i in range(1, lags + 1)]
    X = np.column_stack([np.ones(len(df))] + [df[c].to_numpy() for c in cols])
    return X, z


def _cumulative_moments(X: np.ndarray, z: np.ndarray):
    """Prefix sums of X'X, X'z, z'z so any window's OLS sufficient stats are O(1)."""
    N, p = X.shape
    outer = np.einsum("ni,nj->nij", X, X)
    xz = X * z[:, None]
    zz = z**2
    cum_outer = np.concatenate([np.zeros((1, p, p)), np.cumsum(outer, axis=0)], axis=0)
    cum_xz = np.concatenate([np.zeros((1, p)), np.cumsum(xz, axis=0)], axis=0)
    cum_zz = np.concatenate([[0.0], np.cumsum(zz)])
    return cum_outer, cum_xz, cum_zz


def _tstat_window(cum_outer, cum_xz, cum_zz, a: int, b: int, p: int) -> float:
    """ADF t-stat on beta (coefficient of y_{t-1}) for window [a, b)."""
    n = b - a
    dof = n - p
    if dof <= 0:
        return np.nan
    XtX = cum_outer[b] - cum_outer[a]
    Xz = cum_xz[b] - cum_xz[a]
    zz = cum_zz[b] - cum_zz[a]
    if p == 2:
        s11, s12, s22 = XtX[0, 0], XtX[0, 1], XtX[1, 1]
        z1, z2 = Xz[0], Xz[1]
        denom = s11 * s22 - s12**2
        if denom <= 0:
            return np.nan
        beta = (s11 * z2 - s12 * z1) / denom
        alpha = (z1 - beta * s12) / s11
        ssr = zz - alpha * z1 - beta * z2
        if ssr <= 0:
            return np.nan
        sigma2 = ssr / dof
        se_beta = np.sqrt(sigma2 * s11 / denom)
        return beta / se_beta
    try:
        theta = np.linalg.solve(XtX, Xz)
    except np.linalg.LinAlgError:
        return np.nan
    ssr = zz - theta @ Xz
    if ssr <= 0:
        return np.nan
    sigma2 = ssr / dof
    try:
        cov = sigma2 * np.linalg.inv(XtX)
    except np.linalg.LinAlgError:
        return np.nan
    se_beta = np.sqrt(cov[1, 1])
    return theta[1] / se_beta


def _tstat_window_vec(cum_outer, cum_xz, cum_zz, a_arr: np.ndarray, b: int) -> np.ndarray:
    """Vectorized (lags=0, p=2) ADF t-stat for many start points ``a_arr`` sharing end ``b``."""
    n = b - a_arr
    XtX_b = cum_outer[b]
    Xz_b = cum_xz[b]
    zz_b = cum_zz[b]
    s11 = n.astype(float)
    s12 = XtX_b[0, 1] - cum_outer[a_arr][:, 0, 1]
    s22 = XtX_b[1, 1] - cum_outer[a_arr][:, 1, 1]
    z1 = Xz_b[0] - cum_xz[a_arr][:, 0]
    z2 = Xz_b[1] - cum_xz[a_arr][:, 1]
    zz = zz_b - cum_zz[a_arr]
    denom = s11 * s22 - s12**2
    with np.errstate(invalid="ignore", divide="ignore"):
        beta = (s11 * z2 - s12 * z1) / denom
        alpha = (z1 - beta * s12) / s11
        ssr = zz - alpha * z1 - beta * z2
        dof = n - 2
        sigma2 = ssr / dof
        se_beta = np.sqrt(sigma2 * s11 / denom)
        t = beta / se_beta
    t[(denom <= 0) | (dof <= 0) | (ssr <= 0)] = np.nan
    return t


# ---------------------------------------------------------------------------
# Results containers
# ---------------------------------------------------------------------------

@dataclass
class SADFResult:
    stat: float
    r2_index: int
    series: pd.Series  # ADF_0^{r2} for r2 in [min_n, N]


@dataclass
class GSADFResult:
    stat: float
    r1_index: int
    r2_index: int
    bsadf: pd.Series = field(repr=False)  # BSADF_{r2}(r0) for r2 in [min_n, N]


# ---------------------------------------------------------------------------
# Public statistics
# ---------------------------------------------------------------------------

def adf_stat(y: pd.Series | np.ndarray, lags: int = 0) -> float:
    """Full-sample ADF t-stat (``r1=0, r2=T``), i.e. the classical right-tailed ADF test."""
    values = np.asarray(y, dtype=float)
    X, z = _build_regression_arrays(values, lags)
    cum_outer, cum_xz, cum_zz = _cumulative_moments(X, z)
    return _tstat_window(cum_outer, cum_xz, cum_zz, 0, len(z), X.shape[1])


def _index_of(y) -> pd.Index:
    if isinstance(y, pd.Series):
        return y.index
    return pd.RangeIndex(len(y))


def sadf(y: pd.Series | np.ndarray, r0: float | None = None, lags: int = 0) -> SADFResult:
    """PWY (2011) SADF test: sup over forward-expanding windows with r1 fixed at 0."""
    values = np.asarray(y, dtype=float)
    T = len(values)
    r0 = r0 if r0 is not None else default_r0(T)
    X, z = _build_regression_arrays(values, lags)
    p = X.shape[1]
    N = len(z)
    min_n = max(int(np.floor(r0 * T)), p + 2)
    cum_outer, cum_xz, cum_zz = _cumulative_moments(X, z)

    stats = np.full(N + 1, np.nan)
    for b in range(min_n, N + 1):
        stats[b] = _tstat_window(cum_outer, cum_xz, cum_zz, 0, b, p)

    idx = _index_of(y)
    date_index = idx[lags:]  # idx[lags:] has exactly N+1 elements, aligned 1:1 with `stats`
    series = pd.Series(stats[min_n:], index=date_index[min_n:])
    r2_star = int(np.nanargmax(stats))
    return SADFResult(stat=float(np.nanmax(stats)), r2_index=r2_star, series=series)


def bsadf_sequence(y: pd.Series | np.ndarray, r0: float | None = None, lags: int = 0) -> pd.Series:
    """Backward SADF sequence ``BSADF_{r2}(r0)`` for every feasible ``r2``.

    This is also the core building block for the GSADF statistic
    (``GSADF = max`` of this sequence) and for :func:`bsadf_dating`.
    """
    values = np.asarray(y, dtype=float)
    T = len(values)
    r0 = r0 if r0 is not None else default_r0(T)
    X, z = _build_regression_arrays(values, lags)
    p = X.shape[1]
    N = len(z)
    min_n = max(int(np.floor(r0 * T)), p + 2)
    cum_outer, cum_xz, cum_zz = _cumulative_moments(X, z)

    out = np.full(N + 1, np.nan)
    for b in range(min_n, N + 1):
        a_arr = np.arange(0, b - min_n + 1)
        if p == 2:
            t = _tstat_window_vec(cum_outer, cum_xz, cum_zz, a_arr, b)
        else:
            t = np.array([_tstat_window(cum_outer, cum_xz, cum_zz, a, b, p) for a in a_arr])
        out[b] = np.nanmax(t) if t.size and np.any(~np.isnan(t)) else np.nan

    idx = _index_of(y)
    date_index = idx[lags:]  # idx[lags:] has exactly N+1 elements, aligned 1:1 with `out`
    series = pd.Series(out[min_n:], index=date_index[min_n:])
    series.name = "bsadf"
    return series


def gsadf(y: pd.Series | np.ndarray, r0: float | None = None, lags: int = 0) -> GSADFResult:
    """PSY (2015) GSADF test: double-sup ADF t-stat over all feasible windows (Eq. 5)."""
    bsadf = bsadf_sequence(y, r0=r0, lags=lags)
    r2_star = int(np.nanargmax(bsadf.to_numpy()))
    stat = float(bsadf.iloc[r2_star])
    return GSADFResult(stat=stat, r1_index=-1, r2_index=r2_star, bsadf=bsadf)


# ---------------------------------------------------------------------------
# Critical values
# ---------------------------------------------------------------------------

# Table 1 of Phillips, Shi & Yu (2015): asymptotic critical values keyed by r0.
_TABLE1_R0 = np.array([0.190, 0.137, 0.100, 0.074, 0.055])
_TABLE1_ASYMPTOTIC = {
    "sadf": {
        0.90: np.array([1.10, 1.12, 1.20, 1.21, 1.23]),
        0.95: np.array([1.37, 1.41, 1.49, 1.51, 1.51]),
        0.99: np.array([1.88, 2.03, 2.07, 2.06, 2.06]),
    },
    "gsadf": {
        0.90: np.array([1.67, 1.78, 1.97, 1.99, 2.08]),
        0.95: np.array([1.89, 2.01, 2.19, 2.20, 2.30]),
        0.99: np.array([2.37, 2.48, 2.69, 2.62, 2.74]),
    },
}
# Finite-sample critical values keyed by (T, r0) pairs used in Table 1.
_TABLE1_T = np.array([100, 200, 400, 800, 1600])
_TABLE1_FINITE = {
    "sadf": {
        0.90: np.array([0.98, 1.12, 1.19, 1.25, 1.28]),
        0.95: np.array([1.30, 1.40, 1.49, 1.53, 1.57]),
        0.99: np.array([1.99, 1.90, 2.05, 2.03, 2.22]),
    },
    "gsadf": {
        0.90: np.array([1.65, 1.84, 1.92, 2.10, 2.19]),
        0.95: np.array([2.00, 2.08, 2.20, 2.34, 2.41]),
        0.99: np.array([2.57, 2.70, 2.80, 2.79, 2.87]),
    },
}


def critical_values(
    T: int,
    r0: float | None = None,
    quantiles: tuple[float, ...] = (0.90, 0.95, 0.99),
    kind: str = "finite",
) -> dict[str, dict[float, float]]:
    """Look up/interpolate Table 1 critical values from Phillips, Shi & Yu (2015).

    Parameters
    ----------
    T: sample size.
    r0: minimum window fraction; defaults to :func:`default_r0`.
    quantiles: which quantiles to return (subset of/interpolated between 0.90/0.95/0.99).
    kind: ``"finite"`` (finite-sample, interpolated over T) or ``"asymptotic"``
        (interpolated over r0).

    Returns
    -------
    dict with keys ``"sadf"`` and ``"gsadf"``, each mapping quantile -> critical value.

    Notes
    -----
    This is a fast, paper-faithful *lookup* alternative to Monte Carlo
    simulation (:func:`simulate_critical_values`), obtained by linear
    interpolation of the tabulated critical values in Table 1 of the paper.
    """
    r0 = r0 if r0 is not None else default_r0(T)
    table = _TABLE1_FINITE if kind == "finite" else _TABLE1_ASYMPTOTIC
    xgrid = _TABLE1_T if kind == "finite" else _TABLE1_R0
    xval = T if kind == "finite" else r0
    # tables are indexed with decreasing r0 / increasing T; np.interp needs increasing x
    order = np.argsort(xgrid)
    xgrid_sorted = xgrid[order]
    out: dict[str, dict[float, float]] = {}
    for test_name, qdict in table.items():
        out[test_name] = {}
        for q in quantiles:
            available = np.array(sorted(qdict.keys()))
            lo = available[available <= q].max(initial=available.min())
            hi = available[available >= q].min(initial=available.max())
            y_lo = np.interp(xval, xgrid_sorted, qdict[lo][order])
            y_hi = np.interp(xval, xgrid_sorted, qdict[hi][order])
            if hi == lo:
                out[test_name][q] = float(y_lo)
            else:
                w = (q - lo) / (hi - lo)
                out[test_name][q] = float(y_lo + w * (y_hi - y_lo))
    return out


def _simulate_null_path(T: int, d: float = 1.0, eta: float = 1.0, rng: np.random.Generator | None = None) -> np.ndarray:
    """Simulate a random walk with asymptotically negligible drift (Eq. 3, null model)."""
    rng = rng or np.random.default_rng()
    eps = rng.standard_normal(T)
    drift = d * (np.arange(1, T + 1) / T**eta)
    return drift + np.cumsum(eps)


def simulate_critical_values(
    T: int,
    r0: float | None = None,
    lags: int = 0,
    n_sims: int = 500,
    quantiles: tuple[float, ...] = (0.90, 0.95, 0.99),
    seed: int | None = None,
) -> dict:
    """Monte Carlo critical values for SADF/GSADF/BSADF-sequence, tailored to your ``T``.

    Slower but exact for arbitrary ``T``/``lags`` (unlike the Table 1 lookup in
    :func:`critical_values`, which is restricted to the paper's grid).
    """
    rng = np.random.default_rng(seed)
    r0 = r0 if r0 is not None else default_r0(T)
    sadf_stats = np.empty(n_sims)
    gsadf_stats = np.empty(n_sims)
    bsadf_paths = []
    for i in range(n_sims):
        path = _simulate_null_path(T, rng=rng)
        s = sadf(path, r0=r0, lags=lags)
        g = gsadf(path, r0=r0, lags=lags)
        sadf_stats[i] = s.stat
        gsadf_stats[i] = g.stat
        bsadf_paths.append(g.bsadf.to_numpy())

    max_len = max(len(p) for p in bsadf_paths)
    padded = np.full((n_sims, max_len), np.nan)
    for i, p in enumerate(bsadf_paths):
        padded[i, -len(p):] = p

    out: dict = {"sadf": {}, "gsadf": {}, "bsadf_sequence": {}}
    for q in quantiles:
        out["sadf"][q] = float(np.nanquantile(sadf_stats, q))
        out["gsadf"][q] = float(np.nanquantile(gsadf_stats, q))
        out["bsadf_sequence"][q] = np.nanquantile(padded, q, axis=0)
    return out


# ---------------------------------------------------------------------------
# Date-stamping
# ---------------------------------------------------------------------------

def bsadf_dating(
    y: pd.Series,
    r0: float | None = None,
    lags: int = 0,
    critical_value: float | pd.Series | None = None,
    quantile: float = 0.95,
    min_duration: float | None = None,
) -> pd.DataFrame:
    """PSY (2015) date-stamping strategy (Eqs. 7-8): origination/termination via BSADF crossings.

    Parameters
    ----------
    y: price/price-dividend-ratio series (should be indexed by date for readable output).
    r0, lags: see :func:`bsadf_sequence`.
    critical_value: a constant threshold, a `pd.Series` aligned to the bsadf sequence
        (for a time-varying threshold, e.g. from :func:`simulate_critical_values`), or
        ``None`` to use the Table 1 lookup (:func:`critical_values`) at ``quantile``.
    quantile: quantile used when looking up the default critical value.
    min_duration: minimum bubble duration in *index units* (defaults to ``log(T)``
        observations, following the paper's ``delta * log(T)`` rule with ``delta=1``).

    Returns
    -------
    DataFrame with columns ``start``, ``end`` (index labels) for each detected episode.
    """
    T = len(y)
    bsadf = bsadf_sequence(y, r0=r0, lags=lags)
    if critical_value is None:
        cv = critical_values(T, r0=r0)["gsadf"][quantile]
        threshold = pd.Series(cv, index=bsadf.index)
    elif np.isscalar(critical_value):
        threshold = pd.Series(float(critical_value), index=bsadf.index)
    else:
        threshold = critical_value.reindex(bsadf.index).ffill()

    exceed = (bsadf > threshold).to_numpy()
    min_dur = min_duration if min_duration is not None else max(int(np.log(T)), 1)

    episodes = []
    n = len(exceed)
    i = 0
    while i < n:
        if exceed[i]:
            start = i
            j = i
            while j < n and exceed[j]:
                j += 1
            end = j - 1
            if (end - start + 1) >= min_dur:
                episodes.append((bsadf.index[start], bsadf.index[end]))
            i = j
        else:
            i += 1

    return pd.DataFrame(episodes, columns=["start", "end"])
