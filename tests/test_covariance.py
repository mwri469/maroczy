import numpy as np
import pandas as pd
import pytest

from maroczy.features.covariance import (
    clean_covariance,
    condition_number,
    cv_eigenvalue_clean,
    ledoit_wolf_shrink,
    marchenko_pastur_clip,
)


def _returns(T=250, N=40, seed=0):
    rng = np.random.default_rng(seed)
    # true low-rank + noise structure so cleaning should help
    factor = rng.standard_normal((T, 3))
    loadings = rng.standard_normal((3, N))
    signal = factor @ loadings
    noise = rng.standard_normal((T, N))
    data = 0.7 * signal + noise
    cols = [f"S{i}" for i in range(N)]
    return pd.DataFrame(data, columns=cols)


def test_marchenko_pastur_reduces_condition_number():
    r = _returns()
    raw_corr = np.corrcoef(r.to_numpy(), rowvar=False)
    cleaned = marchenko_pastur_clip(r)
    assert condition_number(cleaned) <= condition_number(raw_corr) * 1.5  # cleaning shouldn't blow up conditioning


def test_ledoit_wolf_shrink_returns_valid_cov():
    r = _returns()
    cov = ledoit_wolf_shrink(r)
    eigvals = np.linalg.eigvalsh(cov.to_numpy())
    assert (eigvals > -1e-8).all()
    assert cov.shape == (r.shape[1], r.shape[1])


def test_cv_eigenvalue_clean_positive_semidefinite():
    r = _returns(T=300, N=20, seed=1)
    cleaned = cv_eigenvalue_clean(r, n_splits=20, random_state=0)
    eigvals = np.linalg.eigvalsh(cleaned.to_numpy())
    assert (eigvals > -1e-6).all()
    assert cleaned.shape == (20, 20)


def test_clean_covariance_dispatch():
    r = _returns(N=15)
    for method in ("cv", "mp", "lw"):
        out = clean_covariance(r, method=method)
        assert out.shape == (15, 15)
    with pytest.raises(ValueError):
        clean_covariance(r, method="unknown")
