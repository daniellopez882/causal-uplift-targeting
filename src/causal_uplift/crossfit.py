"""Fold assignment for cross-fitting."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np


def stratified_folds(strata: np.ndarray, n_folds: int, seed: int) -> np.ndarray:
    """Return a fold id per row so every stratum is spread evenly over the folds.

    Each stratum is shuffled and dealt round-robin from a random starting fold, so the
    remainders of small strata do not all land in fold 0.
    """
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2")
    rng = np.random.default_rng(seed)
    fold = np.empty(strata.shape[0], dtype=np.int64)
    for value in np.unique(strata):
        rows = np.flatnonzero(strata == value)
        rng.shuffle(rows)
        start = int(rng.integers(n_folds))
        fold[rows] = (np.arange(rows.size) + start) % n_folds
    return fold


def experiment_strata(arm: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Arm crossed with whether the outcome is positive, so rare conversions are balanced too."""
    return arm.astype(np.int64) * 2 + (y > 0).astype(np.int64)


def splits(fold: np.ndarray) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    for k in np.unique(fold):
        yield np.flatnonzero(fold != k), np.flatnonzero(fold == k)
