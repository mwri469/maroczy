"""Covariance / correlation matrix cleaning.

Implements the eigenvalue-cleaning family of estimators discussed in
Bongiorno & Challet (2022), *"Covariance Matrix Cleaning: from Average
Rescaling to Rotationally Invariant Estimators"*-style work, i.e. methods
that replace the noisy eigenvalues of a sample correlation matrix with a
"cleaned" version while keeping the (noisy but rotationally-optimal)
sample eigenvectors:

1. **Marchenko-Pastur clipping** (Laloux-Cizeau-Bouchaud-Potters baseline)
   -- eigenvalues inside the MP bulk are collapsed to their average; the
   ``q = n_features / n_obs`` ratio controls the bulk edge.
2. **Ledoit-Wolf linear shrinkage** -- classic baseline via ``sklearn``.
3. **Cross-validated rotationally-invariant eigenvalue cleaning** (the
   Bongiorno & Challet approach) -- repeatedly split the sample in half,
   estimate eigenvectors on one half and evaluate their *realized*
   out-of-sample variance on the other half; average across splits to
   obtain a noise-robust, rotationally-invariant eigenvalue estimate. This
   requires no bulk/edge assumptions and adapts to any noise structure.

All functions operate on a returns matrix (``T x N``, rows = time) and
return a cleaned covariance (or correlation) matrix as a ``pd.DataFrame``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "corr_to_cov",
    "cov_to_corr",
    "condition_number",
    "effective_rank",
    "marchenko_pastur_clip",
    "ledoit_wolf_shrink",
    "cv_eigenvalue_clean",
    "clean_covariance",
]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def cov_to_corr(cov: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (correlation matrix, vector of std devs)."""
    std = np.sqrt(np.diag(cov))
    outer = np.outer(std, std)
    outer[outer == 0] = 1.0
    corr = cov / outer
    np.fill_diagonal(corr, 1.0)
    return corr, std


def corr_to_cov(corr: np.ndarray, std: np.ndarray) -> np.ndarray:
    return corr * np.outer(std, std)


def condition_number(matrix: np.ndarray | pd.DataFrame) -> float:
    """Ratio of largest to smallest eigenvalue -- diagnostic for cleaning quality."""
    m = matrix.to_numpy() if isinstance(matrix, pd.DataFrame) else matrix
    eigvals = np.linalg.eigvalsh(m)
    eigvals = eigvals[eigvals > 1e-12]
    return float(eigvals.max() / eigvals.min()) if len(eigvals) else np.inf


def effective_rank(matrix: np.ndarray | pd.DataFrame) -> float:
    """Exponential of the entropy of the (normalized) eigenvalue spectrum."""
    m = matrix.to_numpy() if isinstance(matrix, pd.DataFrame) else matrix
    eigvals = np.linalg.eigvalsh(m)
    eigvals = eigvals[eigvals > 1e-12]
    p = eigvals / eigvals.sum()
    entropy = -np.sum(p * np.log(p))
    return float(np.exp(entropy))


# ---------------------------------------------------------------------------
# Marchenko-Pastur clipping
# ---------------------------------------------------------------------------

def _mp_edge(q: float, sigma2: float = 1.0) -> tuple[float, float]:
    lo = sigma2 * (1 - np.sqrt(q)) ** 2
    hi = sigma2 * (1 + np.sqrt(q)) ** 2
    return lo, hi


def marchenko_pastur_clip(
    returns: pd.DataFrame,
    q: float | None = None,
) -> pd.DataFrame:
    """Clip eigenvalues within the Marchenko-Pastur bulk to their common average.

    Parameters
    ----------
    returns: T x N returns DataFrame.
    q: ``N/T`` ratio; inferred from ``returns.shape`` if not given.
    """
    X = returns.to_numpy()
    T, N = X.shape
    q = q if q is not None else N / T
    corr = np.corrcoef(X, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(corr)
    _, hi = _mp_edge(min(q, 0.999))
    bulk_mask = eigvals <= hi
    if bulk_mask.any():
        avg_bulk = eigvals[bulk_mask].mean()
        cleaned = eigvals.copy()
        cleaned[bulk_mask] = avg_bulk
    else:
        cleaned = eigvals.copy()
    # rescale so trace (total variance) is preserved
    cleaned *= eigvals.sum() / cleaned.sum()
    corr_clean = (eigvecs * cleaned) @ eigvecs.T
    np.fill_diagonal(corr_clean, 1.0)
    std = X.std(axis=0, ddof=1)
    cov_clean = corr_to_cov(corr_clean, std)
    return pd.DataFrame(cov_clean, index=returns.columns, columns=returns.columns)


# ---------------------------------------------------------------------------
# Ledoit-Wolf shrinkage baseline
# ---------------------------------------------------------------------------

def ledoit_wolf_shrink(returns: pd.DataFrame) -> pd.DataFrame:
    """Classic Ledoit-Wolf linear shrinkage toward a scaled identity target."""
    from sklearn.covariance import LedoitWolf

    lw = LedoitWolf().fit(returns.to_numpy())
    return pd.DataFrame(lw.covariance_, index=returns.columns, columns=returns.columns)


# ---------------------------------------------------------------------------
# Cross-validated rotationally invariant eigenvalue cleaning (Bongiorno & Challet)
# ---------------------------------------------------------------------------

def cv_eigenvalue_clean(
    returns: pd.DataFrame,
    n_splits: int = 50,
    train_frac: float = 0.5,
    shrink_negative: bool = True,
    random_state: int | None = None,
) -> pd.DataFrame:
    """Cross-validated, rotationally invariant eigenvalue cleaning.

    For each of ``n_splits`` random train/test splits of the time index:

    1. Estimate the sample correlation matrix and its eigenvectors ``u_i``
       on the *train* half.
    2. Evaluate the *realized*, out-of-sample variance ``u_i' C_test u_i``
       of each training eigenvector on the *test* half.
    3. Average the resulting out-of-sample eigenvalues across splits and
       across sign-symmetric eigenvector assignment (to remove estimation
       noise in the eigenvectors themselves).

    The final cleaned correlation matrix reuses the *full-sample*
    eigenvectors (best available rotation) combined with the CV-estimated
    eigenvalues -- a rotationally invariant estimator in the sense of
    Bouchaud-Potters/Bun-Bouchaud-Potters, but with an entirely
    non-parametric, assumption-free eigenvalue estimate.

    Parameters
    ----------
    returns: T x N returns DataFrame.
    n_splits: number of random train/test splits to average over.
    train_frac: fraction of observations used for the train half.
    shrink_negative: clip any negative cleaned eigenvalues to a small
        positive floor so the result stays positive semi-definite.
    """
    rng = np.random.default_rng(random_state)
    X = returns.to_numpy()
    T, N = X.shape
    n_train = max(int(T * train_frac), N + 2)

    full_corr = np.corrcoef(X, rowvar=False)
    full_eigvals, full_eigvecs = np.linalg.eigh(full_corr)

    cv_eigval_sum = np.zeros(N)
    cv_eigval_count = np.zeros(N)

    idx = np.arange(T)
    for _ in range(n_splits):
        rng.shuffle(idx)
        train_idx, test_idx = idx[:n_train], idx[n_train:]
        if len(test_idx) < N + 2:
            continue
        X_train, X_test = X[train_idx], X[test_idx]
        train_corr = np.corrcoef(X_train, rowvar=False)
        test_std = X_test.std(axis=0, ddof=1)
        test_std[test_std == 0] = 1.0
        X_test_std = (X_test - X_test.mean(axis=0)) / test_std
        test_cov = (X_test_std.T @ X_test_std) / (len(test_idx) - 1)

        _, train_eigvecs = np.linalg.eigh(train_corr)
        # out-of-sample realized variance along each train eigenvector
        oos_eigvals = np.einsum("ij,jk,ki->i", train_eigvecs.T, test_cov, train_eigvecs)
        cv_eigval_sum += oos_eigvals
        cv_eigval_count += 1

    if cv_eigval_count.sum() == 0:
        raise ValueError("train_frac too large relative to n_features; no valid splits.")
    cv_eigvals = cv_eigval_sum / np.maximum(cv_eigval_count, 1)
    cv_eigvals = np.sort(cv_eigvals)  # align ordering with full_eigvals (ascending)

    if shrink_negative:
        floor = 1e-6
        cv_eigvals = np.clip(cv_eigvals, floor, None)

    # preserve total variance (trace = N for a correlation matrix)
    cv_eigvals *= N / cv_eigvals.sum()

    corr_clean = (full_eigvecs * cv_eigvals) @ full_eigvecs.T
    np.fill_diagonal(corr_clean, 1.0)
    std = X.std(axis=0, ddof=1)
    cov_clean = corr_to_cov(corr_clean, std)
    return pd.DataFrame(cov_clean, index=returns.columns, columns=returns.columns)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def clean_covariance(returns: pd.DataFrame, method: str = "cv", **kwargs) -> pd.DataFrame:
    """Dispatch to one of the cleaning estimators.

    Parameters
    ----------
    method: one of ``"cv"`` (Bongiorno & Challet cross-validated, default),
        ``"mp"`` (Marchenko-Pastur clipping), ``"lw"`` (Ledoit-Wolf).
    """
    if method == "cv":
        return cv_eigenvalue_clean(returns, **kwargs)
    if method == "mp":
        return marchenko_pastur_clip(returns, **kwargs)
    if method == "lw":
        return ledoit_wolf_shrink(returns)
    raise ValueError(f"Unknown method {method!r}; expected one of 'cv', 'mp', 'lw'.")
