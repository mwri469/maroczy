import numpy as np
import pandas as pd
import pytest

from maroczy.features.rough_vol import (
    estimate_hurst,
    simulate_fbm,
    simulate_microstructure,
    simulate_riemann_liouville_fbm,
    simulate_rough_bergomi,
    theoretical_weak_rate,
)


def test_simulate_fbm_shape_and_start_at_zero():
    path = simulate_fbm(n_steps=200, H=0.3, n_paths=4, rng=np.random.default_rng(0))
    assert path.shape == (4, 201)
    assert np.allclose(path[:, 0], 0.0)


def test_hurst_estimate_close_to_true_value_for_fbm():
    true_H = 0.3
    path = simulate_fbm(n_steps=3000, H=true_H, n_paths=1, rng=np.random.default_rng(1))[0]
    series = pd.Series(path)
    H_hat, diagnostics = estimate_hurst(series, lags=range(1, 30))
    assert 0.15 < H_hat < 0.45  # loose bound; exact recovery needs long samples
    assert not diagnostics.empty


def test_riemann_liouville_fbm_runs():
    B_H, t = simulate_riemann_liouville_fbm(n_steps=100, H=0.1, n_paths=2, rng=np.random.default_rng(2))
    assert B_H.shape == (2, 100)
    assert len(t) == 100


def test_rough_bergomi_positive_variance_and_price():
    out = simulate_rough_bergomi(n_steps=100, n_paths=5, H=0.1, eta=1.5, rho=-0.7, xi0=0.04, rng=np.random.default_rng(3))
    assert (out["v"] > 0).all()
    assert (out["S"] > 0).all()
    assert out["S"].shape == (5, 100)


def test_simulate_microstructure_runs_and_scales():
    df = simulate_microstructure(n=50, T=1.0, H=0.15, rng=np.random.default_rng(4))
    assert "log_price" in df.columns
    assert "log_vol" in df.columns
    assert len(df) > 0


def test_theoretical_weak_rate_matches_paper_formula():
    assert theoretical_weak_rate(0.1) == pytest.approx(1 / 3 + 4 * 0.1 / (3 - 6 * 0.1))
    assert theoretical_weak_rate(0.4) == 1.0
