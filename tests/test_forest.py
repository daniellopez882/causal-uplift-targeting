from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from causal_uplift import _tree_kernels as kernels
from causal_uplift.forest import Binner, UpliftForest, honest_means
from causal_uplift.synthetic import make_synthetic

EPS = 1e-6


def _pair(p: float, q: float, criterion: int) -> float:
    if criterion == kernels.KL:
        p, q = min(max(p, EPS), 1 - EPS), min(max(q, EPS), 1 - EPS)
        return p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q))
    return (p - q) ** 2


def _divergence(count: np.ndarray, total: np.ndarray, criterion: int) -> float:
    ctrl = total[0] / count[0]
    return float(np.mean([_pair(total[a] / count[a], ctrl, criterion) for a in range(1, count.size)]))


def _stats(arm: np.ndarray, y: np.ndarray, mask: np.ndarray, n_arms: int) -> tuple[np.ndarray, np.ndarray]:
    count = np.bincount(arm[mask], minlength=n_arms).astype(float)
    total = np.bincount(arm[mask], weights=y[mask], minlength=n_arms)
    return count, total


def _gain(Xb: np.ndarray, arm: np.ndarray, y: np.ndarray, f: int, t: int, n_arms: int, criterion: int) -> float:
    everything = np.ones(arm.size, dtype=bool)
    left = Xb[:, f] <= t
    share = left.mean()
    return (
        share * _divergence(*_stats(arm, y, left, n_arms), criterion)
        + (1 - share) * _divergence(*_stats(arm, y, ~left, n_arms), criterion)
        - _divergence(*_stats(arm, y, everything, n_arms), criterion)
    )


def _brute_force(
    Xb: np.ndarray, arm: np.ndarray, y: np.ndarray, n_bins: np.ndarray, n_arms: int, min_leaf: int, criterion: int
) -> float | None:
    count, _ = _stats(arm, y, np.ones(arm.size, dtype=bool), n_arms)
    if (count < 2 * min_leaf).any():
        return None
    best: float | None = None
    for f in range(Xb.shape[1]):
        for t in range(n_bins[f] - 1):
            left = Xb[:, f] <= t
            lc, _ = _stats(arm, y, left, n_arms)
            rc, _ = _stats(arm, y, ~left, n_arms)
            if (lc < min_leaf).any() or (rc < min_leaf).any():
                continue
            gain = _gain(Xb, arm, y, f, t, n_arms, criterion)
            if gain > 0 and (best is None or gain > best):
                best = gain
    return best


@st.composite
def split_problems(draw: st.DrawFn) -> tuple:
    n_arms = draw(st.integers(2, 3))
    n = draw(st.integers(4 * n_arms, 90))
    n_features = draw(st.integers(1, 3))
    n_bins = np.array([draw(st.integers(2, 6)) for _ in range(n_features)], dtype=np.int64)
    seed = draw(st.integers(0, 2**31 - 1))
    criterion = draw(st.sampled_from([kernels.KL, kernels.ED]))
    rng = np.random.default_rng(seed)
    Xb = np.column_stack([rng.integers(0, b, n) for b in n_bins]).astype(np.uint8)
    arm = rng.permutation(np.arange(n) % n_arms).astype(np.int64)
    binary = (rng.random(n) < rng.uniform(0.1, 0.9)).astype(np.float64)
    y = binary if criterion == kernels.KL else rng.normal(size=n)
    return Xb, arm, y, n_bins, n_arms, draw(st.integers(1, 4)), criterion


@given(split_problems())
def test_kernel_split_matches_brute_force(problem: tuple) -> None:
    Xb, arm, y, n_bins, n_arms, min_leaf, criterion = problem
    rows = np.arange(arm.size, dtype=np.int64)
    candidates = np.arange(Xb.shape[1], dtype=np.int64)
    f, t, gain = kernels.best_split(
        Xb, arm, y, rows, 0, rows.size, candidates, n_bins, n_arms, min_leaf, criterion, 0.0
    )
    expected = _brute_force(Xb, arm, y, n_bins, n_arms, min_leaf, criterion)
    if expected is None:
        assert f == -1
    else:
        assert f >= 0
        assert gain == pytest.approx(expected, rel=1e-9, abs=1e-12)
        assert _gain(Xb, arm, y, f, t, n_arms, criterion) == pytest.approx(expected, rel=1e-9, abs=1e-12)


def test_honest_means_back_off_to_the_parent() -> None:
    left = np.array([1, -1, -1])
    right = np.array([2, -1, -1])
    count = np.array([[10.0, 10.0], [8.0, 2.0], [2.0, 8.0]])
    total = np.array([[5.0, 6.0], [3.0, 2.0], [1.0, 6.0]])
    means = honest_means(count, total, left, right, min_count=3)
    np.testing.assert_allclose(means[0], [0.5, 0.6])
    np.testing.assert_allclose(means[1], [3.0 / 8.0, 0.6])  # treated arm has 2 < 3 rows: parent's mean
    np.testing.assert_allclose(means[2], [0.5, 6.0 / 8.0])  # control has 2 < 3 rows: parent's mean


def test_honest_draw_splits_every_arm_into_disjoint_halves() -> None:
    arm = np.repeat([0, 1, 2], [40, 30, 20])
    forest = UpliftForest(subsample=1.0)
    forest.n_arms_ = 3
    structure, estimation = forest._draw_rows(arm, np.random.default_rng(0))
    assert np.intersect1d(structure, estimation).size == 0
    np.testing.assert_array_equal(np.bincount(arm[structure]), [20, 15, 10])
    adaptive = UpliftForest(subsample=0.5, honest=False)
    adaptive.n_arms_ = 3
    s2, e2 = adaptive._draw_rows(arm, np.random.default_rng(0))
    np.testing.assert_array_equal(s2, e2)


def test_stump_predicts_difference_in_means_of_the_estimation_half() -> None:
    data = make_synthetic("step", 900, seed=4)
    forest = UpliftForest("ed", n_trees=1, max_depth=0, subsample=1.0, seed=7).fit(data.X, data.arm, data.y)
    tree_seed = int(np.random.SeedSequence(7).generate_state(1)[0])
    _, estimation = forest._draw_rows(data.arm, np.random.default_rng(tree_seed))
    arm, y = data.arm[estimation], data.y[estimation]
    expected = [y[arm == a].mean() - y[arm == 0].mean() for a in (1, 2)]
    np.testing.assert_allclose(forest.predict(data.X[:5]), np.tile(expected, (5, 1)))


def test_forest_is_deterministic_across_thread_counts() -> None:
    data = make_synthetic("step", 4000, seed=1)
    a = UpliftForest(n_trees=8, min_samples_leaf=30, seed=3, n_jobs=1).fit(data.X, data.arm, data.y)
    b = UpliftForest(n_trees=8, min_samples_leaf=30, seed=3, n_jobs=3).fit(data.X, data.arm, data.y)
    np.testing.assert_array_equal(a.predict(data.X), b.predict(data.X))
    c = UpliftForest(n_trees=8, min_samples_leaf=30, seed=4).fit(data.X, data.arm, data.y)
    assert not np.array_equal(a.predict(data.X), c.predict(data.X))


def test_euclidean_forest_handles_continuous_outcomes_and_kl_refuses_them() -> None:
    rng = np.random.default_rng(0)
    X = rng.standard_normal((6000, 3))
    arm = rng.integers(0, 2, 6000)
    tau = np.where(X[:, 0] > 0, 2.0, -1.0)
    y = X[:, 1] + arm * tau + rng.normal(0, 1.0, 6000)
    with pytest.raises(ValueError, match="binary outcome"):
        UpliftForest("kl").fit(X, arm, y)
    forest = UpliftForest("ed", n_trees=30, min_samples_leaf=50).fit(X, arm, y)
    estimate = forest.predict(X)[:, 0]
    assert np.sqrt(np.mean((estimate - tau) ** 2)) < 0.5


def test_binner_codes_are_monotone_and_handle_constants_and_nan() -> None:
    X = np.column_stack([np.arange(100.0), np.full(100, 3.0), np.r_[np.arange(99.0), np.nan]])
    binner = Binner(max_bins=8).fit(X)
    codes = binner.transform(X)
    assert np.all(np.diff(codes[:, 0].astype(int)) >= 0)
    assert binner.n_bins_[0] == 8
    assert binner.n_bins_[1] == 1
    assert codes[-1, 2] == binner.n_bins_[2] - 1
    with pytest.raises(ValueError, match="max_bins"):
        Binner(max_bins=300)
