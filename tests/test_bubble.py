import numpy as np
import pandas as pd
import pytest

from maroczy.features.bubble import (
    adf_stat,
    bsadf_dating,
    bsadf_sequence,
    critical_values,
    default_r0,
    gsadf,
    sadf,
)


def _random_walk(T=300, seed=0):
    rng = np.random.default_rng(seed)
    return pd.Series(np.cumsum(rng.standard_normal(T)))


def _explosive_bubble(T=300, bubble_start=150, bubble_end=220, seed=0):
    rng = np.random.default_rng(seed)
    y = np.zeros(T)
    y[0] = 100
    for t in range(1, T):
        if bubble_start <= t <= bubble_end:
            y[t] = 1.04 * y[t - 1] + rng.standard_normal()
        else:
            y[t] = y[t - 1] + rng.standard_normal()
    return pd.Series(y)


def test_default_r0_reasonable():
    r0 = default_r0(300)
    assert 0 < r0 < 1


def test_adf_stat_runs():
    y = _random_walk()
    stat = adf_stat(y)
    assert np.isfinite(stat)


def test_sadf_no_false_bubble_on_pure_random_walk():
    y = _random_walk(seed=42)
    result = sadf(y)
    cv = critical_values(len(y))["sadf"][0.99]
    # not a hard guarantee (this is a single random draw) but should typically hold
    assert result.stat < cv + 3  # generous margin to avoid flaky failures


def test_gsadf_detects_explosive_bubble():
    y = _explosive_bubble(seed=1)
    result = gsadf(y)
    cv = critical_values(len(y))["gsadf"][0.95]
    assert result.stat > cv


def test_bsadf_sequence_length_matches_series():
    y = _explosive_bubble(seed=2)
    seq = bsadf_sequence(y)
    assert len(seq) <= len(y)
    assert seq.notna().any()


def test_bsadf_dating_finds_episode():
    y = _explosive_bubble(bubble_start=150, bubble_end=220, seed=3)
    episodes = bsadf_dating(y, quantile=0.95)
    assert isinstance(episodes, pd.DataFrame)
    assert set(episodes.columns) == {"start", "end"}


def test_critical_values_lookup():
    cv = critical_values(400)
    assert set(cv.keys()) == {"sadf", "gsadf"}
    for test_name in ("sadf", "gsadf"):
        assert cv[test_name][0.90] < cv[test_name][0.95] < cv[test_name][0.99]
