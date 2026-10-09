"""Honest divergence-split uplift forest (KL or Euclidean criterion), multi-arm."""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Literal

import numpy as np

from causal_uplift import _tree_kernels as kernels
from causal_uplift.base import is_binary
from causal_uplift.learners import CateLearner

Criterion = Literal["kl", "ed"]
_CRITERIA: dict[str, int] = {"kl": kernels.KL, "ed": kernels.ED}


class Binner:
    """Quantile binning to at most ``max_bins`` codes per feature; NaN falls in the top bin."""

    def __init__(self, max_bins: int = 32) -> None:
        if not 2 <= max_bins <= 255:
            raise ValueError("max_bins must be between 2 and 255")
        self.max_bins = max_bins

    def fit(self, X: np.ndarray) -> Binner:
        qs = np.linspace(0.0, 1.0, self.max_bins + 1)[1:-1]
        self.edges_ = []
        for j in range(X.shape[1]):
            column = X[:, j]
            column = column[~np.isnan(column)]
            edges = np.unique(np.quantile(column, qs)) if column.size else np.empty(0)
            # The top quantile equals the max; as an edge it would leave the last bin empty.
            if edges.size and column.size and edges[-1] >= column.max():
                edges = edges[:-1]
            self.edges_.append(edges)
        self.n_bins_ = np.array([e.size + 1 for e in self.edges_], dtype=np.int64)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        out = np.empty(X.shape, dtype=np.uint8)
        for j, edges in enumerate(self.edges_):
            out[:, j] = np.searchsorted(edges, X[:, j], side="left")
        return out


@dataclass(frozen=True)
class FittedTree:
    feature: np.ndarray
    threshold: np.ndarray
    left: np.ndarray
    right: np.ndarray
    effect: np.ndarray  # (n_nodes, K): estimated arm-minus-control mean at every node


def honest_means(
    count: np.ndarray, total: np.ndarray, left: np.ndarray, right: np.ndarray, min_count: int
) -> np.ndarray:
    """Per-node arm means; an arm with fewer than ``min_count`` rows inherits its parent's mean."""
    n_nodes, n_arms = count.shape
    parent = np.full(n_nodes, -1, dtype=np.int64)
    internal = np.flatnonzero(left >= 0)
    parent[left[internal]] = internal
    parent[right[internal]] = internal
    means = np.zeros((n_nodes, n_arms))
    with np.errstate(divide="ignore", invalid="ignore"):
        own = total / count
    root_ok = count[0] > 0
    means[0] = np.where(root_ok, own[0], 0.0)
    for node in range(1, n_nodes):
        enough = count[node] >= min_count
        means[node] = np.where(enough, own[node], means[parent[node]])
    return means


class UpliftForest(CateLearner):
    """Random forest of uplift trees whose leaf effects are estimated on held-out rows.

    Each tree draws ``subsample`` of the rows without replacement. With ``honest=True`` that
    draw is split per arm into a structure half (chooses splits) and an estimation half
    (fills the leaves), so a leaf's effect is not computed from the rows that placed it.
    ``min_samples_leaf`` is per arm on the structure half.
    """

    name = "forest"

    def __init__(
        self,
        criterion: Criterion = "kl",
        *,
        n_trees: int = 200,
        max_depth: int = 6,
        min_samples_leaf: int = 100,
        max_features: float = 0.5,
        subsample: float = 0.5,
        honest: bool = True,
        min_estimation_count: int = 10,
        max_bins: int = 32,
        min_gain: float = 0.0,
        n_jobs: int = 1,
        seed: int = 0,
    ) -> None:
        if criterion not in _CRITERIA:
            raise ValueError(f"criterion must be one of {tuple(_CRITERIA)}")
        if not 0.0 < subsample <= 1.0 or not 0.0 < max_features <= 1.0:
            raise ValueError("subsample and max_features must be in (0, 1]")
        self.criterion = criterion
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.subsample = subsample
        self.honest = honest
        self.min_estimation_count = min_estimation_count
        self.max_bins = max_bins
        self.min_gain = min_gain
        self.n_jobs = n_jobs
        self.seed = seed

    def _fit(self, X: np.ndarray, arm: np.ndarray, y: np.ndarray) -> None:
        if self.criterion == "kl" and not is_binary(y):
            raise ValueError("the KL criterion needs a binary outcome; use criterion='ed' for continuous outcomes")
        self.binner_ = Binner(self.max_bins).fit(X)
        Xb = self.binner_.transform(X)
        arm = np.ascontiguousarray(arm, dtype=np.int64)
        y = np.ascontiguousarray(y, dtype=np.float64)
        n_candidates = max(1, math.ceil(self.max_features * X.shape[1]))
        seeds = np.random.SeedSequence(self.seed).generate_state(self.n_trees)

        def grow(tree_seed: int) -> FittedTree:
            rng = np.random.default_rng(tree_seed)
            structure, estimation = self._draw_rows(arm, rng)
            feature, threshold, left, right = kernels.grow_tree(
                Xb,
                arm,
                y,
                structure,
                self.binner_.n_bins_,
                self.n_arms_,
                self.max_depth,
                self.min_samples_leaf,
                n_candidates,
                _CRITERIA[self.criterion],
                self.min_gain,
                int(tree_seed % (2**31 - 1)),
            )
            count, total = kernels.path_sums(Xb, arm, y, estimation, feature, threshold, left, right, self.n_arms_)
            means = honest_means(count, total, left, right, self.min_estimation_count)
            return FittedTree(feature, threshold, left, right, means[:, 1:] - means[:, :1])

        if self.n_jobs == 1:
            self.trees_ = [grow(int(s)) for s in seeds]
        else:
            with ThreadPoolExecutor(max_workers=self.n_jobs) as pool:
                self.trees_ = list(pool.map(grow, [int(s) for s in seeds]))

    def _draw_rows(self, arm: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        structure: list[np.ndarray] = []
        estimation: list[np.ndarray] = []
        for a in range(self.n_arms_):
            rows = np.flatnonzero(arm == a)
            take = rng.permutation(rows)[: max(1, round(self.subsample * rows.size))]
            if self.honest:
                half = take.size // 2
                structure.append(take[:half])
                estimation.append(take[half:])
            else:
                structure.append(take)
                estimation.append(take)
        return np.sort(np.concatenate(structure)), np.sort(np.concatenate(estimation))

    def predict(self, X: np.ndarray) -> np.ndarray:
        Xb = self.binner_.transform(X)
        out = np.zeros((X.shape[0], self.n_arms_ - 1))
        for tree in self.trees_:
            out += tree.effect[kernels.apply_tree(Xb, tree.feature, tree.threshold, tree.left, tree.right)]
        return out / len(self.trees_)
