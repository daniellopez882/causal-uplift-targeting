from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from causal_uplift.metrics import (
    RankingBootstrap,
    auuc,
    perfect_qini_curve,
    qini_coefficient,
    qini_curve,
    uplift_curve,
)


@st.composite
def experiments(draw: st.DrawFn, min_size: int = 6, max_size: int = 60) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = draw(st.integers(min_size, max_size))
    seed = draw(st.integers(0, 2**31 - 1))
    rng = np.random.default_rng(seed)
    treated = rng.permutation(np.arange(n) % 2).astype(float)
    y = (rng.random(n) < rng.uniform(0.2, 0.8)).astype(float)
    if draw(st.booleans()):
        score = rng.integers(0, 4, n).astype(float)  # heavy ties
    else:
        score = rng.normal(size=n)
    return y, score, treated


def _has_signal(y: np.ndarray, treated: np.ndarray) -> bool:
    perfect = perfect_qini_curve(y, treated)
    baseline = perfect.n_targeted[-1] * perfect.value[-1] / 2.0
    return perfect.area() - baseline > 1e-9


@given(experiments())
def test_no_ranking_beats_the_perfect_ranking(data: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
    y, score, treated = data
    if not _has_signal(y, treated):
        return
    assert qini_coefficient(y, score, treated) <= 1.0 + 1e-9
    assert qini_coefficient(y, y * treated - y * (1 - treated), treated) == pytest.approx(1.0)


def test_random_ranking_has_zero_qini_in_expectation() -> None:
    rng = np.random.default_rng(0)
    n = 4000
    treated = rng.permutation(np.arange(n) % 2).astype(float)
    y = (rng.random(n) < 0.2 + 0.1 * treated).astype(float)
    draws = np.array([qini_coefficient(y, rng.random(n), treated) for _ in range(300)])
    assert abs(draws.mean()) < 4 * draws.std(ddof=1) / np.sqrt(draws.size)


@given(experiments(), st.integers(0, 10_000))
def test_integer_weights_equal_duplicated_rows(data: tuple[np.ndarray, np.ndarray, np.ndarray], seed: int) -> None:
    y, score, treated = data
    counts = np.random.default_rng(seed).integers(1, 4, y.size)
    rows = np.repeat(np.arange(y.size), counts)
    for curve in (qini_curve, uplift_curve):
        weighted = curve(y, score, treated, counts.astype(float))
        duplicated = curve(y[rows], score[rows], treated[rows])
        np.testing.assert_allclose(weighted.n_targeted, duplicated.n_targeted)
        np.testing.assert_allclose(weighted.value, duplicated.value, atol=1e-9)


def test_curves_start_at_origin_and_end_at_the_total_effect() -> None:
    y = np.array([1, 0, 1, 0, 0, 1.0])
    treated = np.array([1, 1, 1, 0, 0, 0.0])
    curve = qini_curve(y, np.arange(6.0), treated)
    assert (curve.n_targeted[0], curve.value[0]) == (0.0, 0.0)
    assert curve.n_targeted[-1] == 6
    assert curve.value[-1] == pytest.approx(2.0 - 1.0 * 3 / 3)
    resampled = curve.resample(11)
    assert resampled.n_targeted.size == 11
    assert resampled.value[-1] == pytest.approx(curve.value[-1])


def test_bootstrap_point_matches_direct_scores_and_is_bracketed() -> None:
    rng = np.random.default_rng(3)
    n = 3000
    treated = rng.permutation(np.arange(n) % 2).astype(float)
    x = rng.normal(size=n)
    y = (rng.random(n) < 0.2 + 0.15 * treated * (x > 0)).astype(float)
    result = RankingBootstrap(y, x, treated).run(n_boot=200, seed=1)
    assert result["qini"].value == pytest.approx(qini_coefficient(y, x, treated))
    assert result["auuc"].value == pytest.approx(auuc(y, x, treated))
    for interval in result.values():
        assert interval.lo <= interval.value <= interval.hi
    assert result["qini"].lo > 0  # the score carries real signal here
    again = RankingBootstrap(y, x, treated).run(n_boot=200, seed=1)
    assert again == result


def test_treatment_must_be_binary() -> None:
    with pytest.raises(ValueError, match="0/1"):
        qini_curve(np.zeros(3), np.zeros(3), np.array([0, 1, 2]))
