from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from causal_uplift.metrics import (
    RankingBootstrap,
    auuc,
    perfect_qini_curve,
    perfect_uplift_curve,
    qini_coefficient,
    qini_curve,
    uplift_curve,
)
from causal_uplift.synthetic import make_synthetic, potential_means


@st.composite
def experiments(draw: st.DrawFn, min_size: int = 6, max_size: int = 60) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = draw(st.integers(min_size, max_size))
    seed = draw(st.integers(0, 2**31 - 1))
    rng = np.random.default_rng(seed)
    treated = rng.permutation(np.arange(n) % 2).astype(float)
    y = (rng.random(n) < rng.uniform(0.2, 0.8)).astype(float)
    heavy_ties = draw(st.booleans())
    score = rng.integers(0, 4, n).astype(float) if heavy_ties else rng.normal(size=n)
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


@given(experiments())
def test_no_ranking_beats_the_perfect_uplift_curve(data: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
    y, score, treated = data
    perfect = perfect_uplift_curve(y, treated)
    if perfect.area() - perfect.n_targeted[-1] * perfect.value[-1] / 2.0 <= 1e-9:
        return
    assert auuc(y, score, treated) <= 1.0 + 1e-9


def test_normalised_scores_need_a_binary_outcome() -> None:
    """With spend-like outcomes the "perfect" orderings are not upper bounds, so there is no normalised score."""
    rng = np.random.default_rng(0)
    n = 4000
    treated = (rng.random(n) < 0.5).astype(float)
    x = rng.normal(size=n)
    buys = rng.random(n) < 0.05 + 0.05 * treated * (x > 0)
    spend = np.where(buys, rng.gamma(2.0, 50.0, n), 0.0)
    for score in (qini_coefficient, auuc):
        with pytest.raises(ValueError, match="0/1 outcome"):
            score(spend, x, treated)
    with pytest.raises(ValueError, match="0/1 outcome"):
        RankingBootstrap(spend, x, treated)
    assert qini_curve(spend, x, treated).value[-1] == pytest.approx(
        spend[treated == 1].sum() - spend[treated == 0].sum() * treated.sum() / (n - treated.sum())
    )


def test_random_ranking_has_zero_qini_in_expectation() -> None:
    rng = np.random.default_rng(0)
    n = 4000
    treated = rng.permutation(np.arange(n) % 2).astype(float)
    y = (rng.random(n) < 0.2 + 0.1 * treated).astype(float)
    draws = np.array([qini_coefficient(y, rng.random(n), treated) for _ in range(300)])
    assert abs(draws.mean()) < 4 * draws.std(ddof=1) / np.sqrt(draws.size)


@pytest.mark.parametrize("scenario", ["linear", "step", "sleeping_dogs"])
def test_ranking_by_the_true_effect_has_the_highest_qini_in_expectation(scenario: str) -> None:
    """Qini is computed from realised outcomes, so on a single sample a worse ranking can score
    higher than the true effect (the synthetic benchmark shows this on sleeping_dogs). Averaged
    over many independent samples, no ranking beats the oracle beyond Monte Carlo error."""
    rng = np.random.default_rng(0)
    names = ("noisy, sd 0.03", "noisy, sd 0.1", "purchase probability", "random")
    gaps: dict[str, list[float]] = {name: [] for name in names}
    for draw in range(150):
        data = make_synthetic(scenario, 3000, seed=500 + draw)
        assert data.true_cate is not None
        rows = data.arm <= 1
        y, treated, tau = data.y[rows], (data.arm[rows] == 1).astype(float), data.true_cate[rows, 0]
        oracle = qini_coefficient(y, tau, treated)
        rivals = {
            names[0]: tau + rng.normal(0, 0.03, tau.size),
            names[1]: tau + rng.normal(0, 0.1, tau.size),
            names[2]: potential_means(scenario, data.X[rows])[:, 0],
            names[3]: rng.random(tau.size),
        }
        for name, score in rivals.items():
            gaps[name].append(oracle - qini_coefficient(y, score, treated))
    for name, values in gaps.items():
        gap = np.asarray(values)
        se = gap.std(ddof=1) / np.sqrt(gap.size)
        assert gap.mean() > -3 * se, name
    for name in names[1:]:
        assert np.mean(gaps[name]) > 3 * np.std(gaps[name], ddof=1) / np.sqrt(len(gaps[name])), name


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
