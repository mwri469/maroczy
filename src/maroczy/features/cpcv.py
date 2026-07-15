"""Combinatorial Purged Cross-Validation (CPCV).

Implements the purged & embargoed combinatorial cross-validation scheme
from Lopez de Prado, *"Advances in Financial Machine Learning"* (2018),
Chapter 12, plus the closely related Probability-of-Backtest-Overfitting
(PBO) diagnostic from Bailey, Borwein, Lopez de Prado & Zhu (2014),
*"The Probability of Backtest Overfitting"*.

Why CPCV over plain K-fold
--------------------------
Financial labels typically span a *window* of time (e.g. a triple-barrier
label realized several bars after the observation date), so naive K-fold
CV leaks information between adjacent train/test folds. CPCV:

1. **Purges** training samples whose label window overlaps the test window.
2. **Embargoes** a further block of training samples immediately following
   the test window (serial correlation leakage).
3. Uses **all** ``C(N, k)`` combinations of ``k`` test groups out of ``N``
   groups (rather than a single sequential pass), which allows many
   independent backtest **paths** to be reconstructed and hence supports
   overfitting-probability diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import comb

import numpy as np
import pandas as pd

__all__ = [
    "CombinatorialPurgedCV",
    "CPCVSplit",
    "probability_of_backtest_overfitting",
]


@dataclass
class CPCVSplit:
    train_idx: np.ndarray
    test_idx: np.ndarray
    test_groups: tuple[int, ...]
    combo_idx: int


class CombinatorialPurgedCV:
    """Purged & embargoed combinatorial cross-validator.

    Parameters
    ----------
    n_groups: number of contiguous time groups to split the sample into.
    n_test_groups: number of groups used as the test set in each combination.
    embargo_frac: fraction of total samples embargoed immediately after each
        test group (as in de Prado, Snippet 7.3).
    """

    def __init__(self, n_groups: int, n_test_groups: int, embargo_frac: float = 0.01):
        if n_test_groups >= n_groups:
            raise ValueError("n_test_groups must be < n_groups")
        self.n_groups = n_groups
        self.n_test_groups = n_test_groups
        self.embargo_frac = embargo_frac
        self._combos = list(combinations(range(n_groups), n_test_groups))

    @property
    def n_splits(self) -> int:
        return len(self._combos)

    @property
    def n_paths(self) -> int:
        """Number of reconstructable, fully-independent backtest paths, ``C(N-1, k-1)``."""
        return comb(self.n_groups - 1, self.n_test_groups - 1)

    def _group_bounds(self, n_samples: int) -> list[tuple[int, int]]:
        edges = np.linspace(0, n_samples, self.n_groups + 1).astype(int)
        return list(zip(edges[:-1], edges[1:]))

    def split(
        self,
        n_samples: int,
        pred_times: np.ndarray | pd.Series | None = None,
        eval_times: np.ndarray | pd.Series | None = None,
    ):
        """Yield a :class:`CPCVSplit` for every ``C(n_groups, n_test_groups)`` combination.

        Parameters
        ----------
        n_samples: total number of observations.
        pred_times: for each sample ``i``, the time the *feature/prediction*
            is made (defaults to the sample's own position, i.e. no label
            overlap assumed).
        eval_times: for each sample ``i``, the time the *label/outcome* is
            fully known (e.g. the triple-barrier touch time). Defaults to
            ``pred_times``.
        """
        bounds = self._group_bounds(n_samples)
        embargo = int(n_samples * self.embargo_frac)
        idx_all = np.arange(n_samples)

        pred_times = np.asarray(pred_times) if pred_times is not None else idx_all.astype(float)
        eval_times = np.asarray(eval_times) if eval_times is not None else pred_times

        for combo_idx, combo in enumerate(self._combos):
            test_ranges = [bounds[g] for g in combo]
            test_idx = np.concatenate([idx_all[a:b] for a, b in test_ranges])
            train_mask = np.ones(n_samples, dtype=bool)
            train_mask[test_idx] = False
            for a, b in test_ranges:
                test_start_t = pred_times[a]
                test_end_t = eval_times[min(b, n_samples) - 1]
                overlap = (pred_times <= test_end_t) & (eval_times >= test_start_t)
                train_mask &= ~overlap
                embargo_end = min(b + embargo, n_samples)
                train_mask[b:embargo_end] = False
            train_idx = idx_all[train_mask]
            yield CPCVSplit(train_idx=train_idx, test_idx=test_idx, test_groups=combo, combo_idx=combo_idx)

    def group_slice(self, n_samples: int, group: int) -> tuple[int, int]:
        return self._group_bounds(n_samples)[group]

    def path_assignment(self) -> dict[int, list[int]]:
        """For each group, the ordered list of combo indices in which it's a test group.

        Each group appears in exactly ``n_paths`` combos; the position in
        this list is used as the "path index" during reconstruction.
        """
        group_to_combos: dict[int, list[int]] = {g: [] for g in range(self.n_groups)}
        for combo_idx, combo in enumerate(self._combos):
            for g in combo:
                group_to_combos[g].append(combo_idx)
        return group_to_combos

    def reconstruct_paths(
        self,
        oos_by_split_group: dict[tuple[int, int], np.ndarray],
        n_samples: int,
    ) -> np.ndarray:
        """Reassemble per-(combo, group) OOS outputs into ``n_paths`` full-length series.

        Parameters
        ----------
        oos_by_split_group: mapping ``(combo_idx, group) -> array`` of the
            out-of-sample predictions/returns for that group's bars,
            produced while iterating :meth:`split`.
        n_samples: total number of observations (same as passed to :meth:`split`).

        Returns
        -------
        Array of shape ``(n_paths, n_samples)``; every path covers every
        bar exactly once (assuming ``oos_by_split_group`` was fully populated).
        """
        bounds = self._group_bounds(n_samples)
        group_to_combos = self.path_assignment()
        paths = np.full((self.n_paths, n_samples), np.nan)
        for g in range(self.n_groups):
            a, b = bounds[g]
            for path_idx, combo_idx in enumerate(group_to_combos[g]):
                key = (combo_idx, g)
                if key in oos_by_split_group:
                    paths[path_idx, a:b] = oos_by_split_group[key]
        return paths


def probability_of_backtest_overfitting(
    performance_matrix: pd.DataFrame,
    n_splits: int = 16,
    metric: str = "sharpe",
) -> dict:
    """Combinatorially-Symmetric Cross-Validation (CSCV) probability of overfitting.

    Bailey, Borwein, Lopez de Prado & Zhu (2014). Splits the sample period
    into ``n_splits`` contiguous blocks, evaluates every strategy/parameter
    column of ``performance_matrix`` (returns, one column per trial) on
    every combination of half the blocks as in-sample vs. the complementary
    half as out-of-sample, and measures how often the best in-sample
    strategy ranks below the out-of-sample median.

    Parameters
    ----------
    performance_matrix: ``T x N`` DataFrame of strategy/trial *returns*
        (one column per configuration tested).
    n_splits: number of contiguous blocks (must be even).
    metric: ``"sharpe"`` (mean/std) or ``"mean"``.

    Returns
    -------
    dict with ``pbo`` (probability of backtest overfitting in ``[0, 1]``),
    ``logits`` (array of per-combination logit-lambda values) and
    ``n_combinations``.
    """
    if n_splits % 2 != 0:
        raise ValueError("n_splits must be even for symmetric IS/OOS combinations.")
    X = performance_matrix.to_numpy()
    T, N = X.shape
    bounds = np.linspace(0, T, n_splits + 1).astype(int)
    blocks = [X[bounds[i]:bounds[i + 1]] for i in range(n_splits)]

    def score(block_rows: np.ndarray) -> np.ndarray:
        mean = block_rows.mean(axis=0)
        if metric == "mean":
            return mean
        std = block_rows.std(axis=0, ddof=1)
        std[std == 0] = np.nan
        return mean / std

    half = n_splits // 2
    logits = []
    for is_blocks in combinations(range(n_splits), half):
        oos_blocks = tuple(b for b in range(n_splits) if b not in is_blocks)
        is_data = np.concatenate([blocks[b] for b in is_blocks], axis=0)
        oos_data = np.concatenate([blocks[b] for b in oos_blocks], axis=0)
        is_perf = score(is_data)
        oos_perf = score(oos_data)
        best_n = np.nanargmax(is_perf)
        rank = (oos_perf < oos_perf[best_n]).sum() + 1  # rank of best-IS strategy OOS (1=worst)
        w = rank / (N + 1)
        w = min(max(w, 1e-6), 1 - 1e-6)
        logit = np.log(w / (1 - w))
        logits.append(logit)

    logits = np.array(logits)
    pbo = float(np.mean(logits <= 0))
    return {"pbo": pbo, "logits": logits, "n_combinations": len(logits)}
