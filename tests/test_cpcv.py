import numpy as np
import pandas as pd
import pytest

from maroczy.features.cpcv import CombinatorialPurgedCV, probability_of_backtest_overfitting


def test_cpcv_split_counts():
    cv = CombinatorialPurgedCV(n_groups=6, n_test_groups=2, embargo_frac=0.01)
    from math import comb

    assert cv.n_splits == comb(6, 2)
    assert cv.n_paths == comb(5, 1)


def test_cpcv_train_test_disjoint_and_purged():
    n_samples = 600
    cv = CombinatorialPurgedCV(n_groups=6, n_test_groups=2, embargo_frac=0.02)
    for split in cv.split(n_samples):
        assert set(split.train_idx).isdisjoint(set(split.test_idx))
        assert len(split.test_idx) > 0


def test_cpcv_path_reconstruction_covers_all_samples():
    n_samples = 300
    cv = CombinatorialPurgedCV(n_groups=5, n_test_groups=2, embargo_frac=0.0)
    oos = {}
    bounds = cv._group_bounds(n_samples)
    for split in cv.split(n_samples):
        for g in split.test_groups:
            a, b = bounds[g]
            oos[(split.combo_idx, g)] = np.full(b - a, split.combo_idx, dtype=float)
    paths = cv.reconstruct_paths(oos, n_samples)
    assert paths.shape == (cv.n_paths, n_samples)
    # every path should be fully covered (no NaNs) since every group appears in n_paths combos
    assert not np.isnan(paths).any()


def test_pbo_runs_and_bounded():
    rng = np.random.default_rng(0)
    # 8 strategies, one genuinely good, rest noise -> PBO should be low-ish but just check bounds/validity
    T, N = 400, 8
    perf = pd.DataFrame(rng.standard_normal((T, N)) * 0.01)
    perf[0] += 0.001  # give strategy 0 a real edge
    result = probability_of_backtest_overfitting(perf, n_splits=8)
    assert 0.0 <= result["pbo"] <= 1.0
    assert result["n_combinations"] > 0
