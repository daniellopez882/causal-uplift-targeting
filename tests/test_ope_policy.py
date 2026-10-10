from __future__ import annotations

import itertools

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from causal_uplift.crossfit import experiment_strata, stratified_folds
from causal_uplift.data import UpliftData
from causal_uplift.evaluation import (
    BEST_ARM_POLICY,
    PROPENSITY_ESTIMATED,
    PROPENSITY_RECORDED,
    PROPENSITY_SHARES,
    RANDOM_POLICY,
    assignment_propensity,
    best_single_arm,
    policy_metrics,
)
from causal_uplift.ope import (
    aipw_scores,
    bootstrap_ratio_draws,
    ipw_scores,
    one_hot_policy,
    percentile_intervals,
    policy_value,
    random_policy,
    simultaneous_intervals,
    single_arm_policy,
)
from causal_uplift.policy import allocate
from causal_uplift.synthetic import make_synthetic, potential_means


def _target_policy(X: np.ndarray) -> np.ndarray:
    """A fixed rule: arm 1 when x2 > 0.5, arm 2 when x3 < -0.5, else no contact."""
    assign = np.where(X[:, 2] > 0.5, 1, np.where(X[:, 3] < -0.5, 2, 0))
    return one_hot_policy(assign, 3)


def test_aipw_and_ipw_are_unbiased_even_with_a_wrong_outcome_model() -> None:
    errors: dict[str, list[float]] = {"ipw": [], "aipw_wrong_mu": [], "aipw_true_mu": []}
    for rep in range(400):
        data = make_synthetic("linear", 1500, seed=10_000 + rep)
        means = potential_means("linear", data.X)
        assert data.propensity is not None
        pi = _target_policy(data.X)
        truth = policy_value(pi, means).mean()
        wrong_mu = np.full_like(means, 0.6)
        estimates = {
            "ipw": policy_value(pi, ipw_scores(data.arm, data.y, data.propensity)).mean(),
            "aipw_wrong_mu": policy_value(pi, aipw_scores(data.arm, data.y, data.propensity, wrong_mu)).mean(),
            "aipw_true_mu": policy_value(pi, aipw_scores(data.arm, data.y, data.propensity, means)).mean(),
        }
        for key, value in estimates.items():
            errors[key].append(value - truth)
    for key, values in errors.items():
        err = np.asarray(values)
        se = err.std(ddof=1) / np.sqrt(err.size)
        assert abs(err.mean()) < 4 * se, key
    assert np.std(errors["aipw_true_mu"]) < np.std(errors["ipw"])


def test_random_policy_rows_are_distributions() -> None:
    pi = random_policy(5, 3, 0.2)
    np.testing.assert_allclose(pi.sum(axis=1), 1.0)
    np.testing.assert_allclose(pi[0], [0.8, 0.1, 0.1])
    np.testing.assert_array_equal(one_hot_policy(np.array([2, 0]), 3), [[0, 0, 1], [1, 0, 0]])


def test_bootstrap_ratios_are_paired_and_reproducible() -> None:
    rng = np.random.default_rng(0)
    values = rng.normal(1.0, 1.0, 2000)
    num = np.column_stack([values, values, 2 * values])
    den = np.column_stack([np.ones(2000), np.ones(2000), np.full(2000, 2.0)])
    point, draws = bootstrap_ratio_draws(num, den, n_boot=300, seed=4)
    first = percentile_intervals(point, draws)
    assert first[0] == first[1] == first[2]  # identical ratios in every replicate
    assert first[0].value == pytest.approx(values.mean())
    assert first[0].lo < values.mean() < first[0].hi
    again = bootstrap_ratio_draws(num, den, n_boot=300, seed=4)
    assert percentile_intervals(*again) == first
    # Identical columns add no multiplicity, so the simultaneous band is a single column's band.
    joint = simultaneous_intervals(point, draws)
    assert joint[0] == joint[1] == joint[2]
    assert joint[0].hi - joint[0].lo == pytest.approx(first[0].hi - first[0].lo, rel=0.1)
    with pytest.raises(ValueError, match="same shape"):
        bootstrap_ratio_draws(num, den[:, :2], n_boot=10, seed=0)


def test_simultaneous_intervals_cover_every_column_at_once() -> None:
    """Eight independent means: per-column 95% intervals all cover far less often than 95%
    of the time, which is the winner's curse a best-of-eight decision runs into."""
    rng = np.random.default_rng(1)
    n, m, reps = 400, 8, 150
    covered = {"percentile": 0, "simultaneous": 0}
    for rep in range(reps):
        values = rng.normal(0.0, 1.0, (n, m))
        point, draws = bootstrap_ratio_draws(values, np.ones((n, m)), n_boot=300, seed=rep)
        for kind, intervals in (
            ("percentile", percentile_intervals(point, draws)),
            ("simultaneous", simultaneous_intervals(point, draws)),
        ):
            covered[kind] += all(i.lo <= 0.0 <= i.hi for i in intervals)
    assert covered["simultaneous"] / reps > 0.88
    assert covered["percentile"] / reps < 0.8


def test_best_single_arm_is_the_arm_worth_most_at_the_budget() -> None:
    gamma = np.column_stack([np.zeros(4), np.full(4, 0.02), np.full(4, 0.05)])
    assert best_single_arm(gamma, 0.2, None) == (2, 0.2)
    # At four times the cost the budget buys a quarter of the contacts, and arm 1 is worth more.
    assert best_single_arm(gamma, 0.2, np.array([1.0, 4.0])) == (1, 0.2)
    np.testing.assert_allclose(single_arm_policy(3, 3, 2, 0.2)[0], [0.8, 0.0, 0.2])


def test_policy_metrics_score_the_no_model_references() -> None:
    data = make_synthetic("step", 6000, seed=5)
    assert data.propensity is not None
    fold = stratified_folds(experiment_strata(data.arm, data.y), 2, 0)
    mu = np.zeros((data.n, 3))  # IPW-like AIPW: the estimates are still unbiased with known propensities
    cates = {"oracle": data.true_cate, "backwards": -data.true_cate}
    out = policy_metrics(data, cates, mu, data.propensity, fold, budgets=[0.2], costs=None, n_boot=200, seed=0)["0.2"]
    assert set(out) == {"oracle", "backwards", RANDOM_POLICY, BEST_ARM_POLICY}
    gamma = aipw_scores(data.arm, data.y, data.propensity, mu)
    arm, share = best_single_arm(gamma, 0.2, None)
    reference = out[BEST_ARM_POLICY]
    assert reference["arm_mix"][data.arm_names[arm]] == pytest.approx(share)
    expected = 1000 * share * np.mean(gamma[:, arm] - gamma[:, 0])
    assert reference["aipw_per_1000_customers"]["value"] == pytest.approx(expected)
    for name in ("oracle", "backwards"):
        for label in ("random", "best_arm"):
            joint = out[name][f"aipw_gain_vs_{label}_simultaneous_per_1000_customers"]
            single = out[name][f"aipw_gain_vs_{label}_per_1000_customers"]
            assert joint["value"] == single["value"]
            assert joint["hi"] - joint["lo"] >= 0.95 * (single["hi"] - single["lo"])
    gains = {name: out[name]["aipw_gain_vs_best_arm_per_1000_customers"]["value"] for name in cates}
    assert gains["oracle"] > 0 > gains["backwards"]
    assert "aipw_gain_vs_random_simultaneous_per_1000_customers" not in reference


def _confounded_without_records(n: int, seed: int) -> tuple[UpliftData, UpliftData]:
    data = make_synthetic("confounded", n, seed=seed)
    stripped = UpliftData(data.X, data.arm, data.y, data.feature_names, data.arm_names, data.outcome)
    return data, stripped


def test_policy_value_uses_recorded_or_estimated_propensities_under_confounding() -> None:
    """Value of "everyone gets arm1": the arm shares are badly biased when assignment depends on x0."""
    data, stripped = _confounded_without_records(20_000, seed=4)
    truth = float(np.mean(potential_means("confounded", data.X)[:, 1] - potential_means("confounded", data.X)[:, 0]))
    fold = stratified_folds(experiment_strata(data.arm, data.y), 5, 0)
    estimates = {}
    for label, source, kind in (
        (PROPENSITY_RECORDED, data, "constant"),
        (PROPENSITY_ESTIMATED, stripped, "logistic"),
        (PROPENSITY_SHARES, stripped, "constant"),
    ):
        e, described = assignment_propensity(source, kind, fold)
        assert described == label
        gamma = ipw_scores(source.arm, source.y, e)
        estimates[label] = float(np.mean(gamma[:, 1] - gamma[:, 0]))
    assert abs(estimates[PROPENSITY_RECORDED] - truth) < 0.02
    assert abs(estimates[PROPENSITY_ESTIMATED] - truth) < 0.02
    assert estimates[PROPENSITY_SHARES] - truth > 0.04


def _brute_force(benefit: np.ndarray, budget: float, cost: np.ndarray) -> float:
    n, k = benefit.shape
    best = 0.0
    for choice in itertools.product(range(k + 1), repeat=n):
        a = np.array(choice)
        used = a > 0
        if cost[a[used] - 1].sum() <= budget + 1e-9:
            best = max(best, float(benefit[np.flatnonzero(used), a[used] - 1].sum()))
    return best


@st.composite
def knapsacks(draw: st.DrawFn, equal_costs: bool) -> tuple[np.ndarray, float, np.ndarray]:
    n = draw(st.integers(1, 6))
    k = draw(st.integers(1, 3))
    seed = draw(st.integers(0, 2**31 - 1))
    rng = np.random.default_rng(seed)
    benefit = np.round(rng.normal(0.05, 0.1, (n, k)), 3)
    cost = np.ones(k) if equal_costs else np.round(rng.uniform(0.5, 3.0, k), 2)
    budget = float(draw(st.integers(0, n))) if equal_costs else float(np.round(rng.uniform(0, n * 2.0), 2))
    return benefit, budget, cost


@given(knapsacks(equal_costs=True))
def test_equal_costs_allocation_is_optimal(problem: tuple[np.ndarray, float, np.ndarray]) -> None:
    benefit, budget, cost = problem
    result = allocate(benefit, budget, cost)
    assert result.spend <= budget + 1e-9
    assert result.benefit == pytest.approx(_brute_force(benefit, budget, cost), abs=1e-9)


@given(knapsacks(equal_costs=False))
def test_unequal_costs_allocation_is_feasible_and_bounded(problem: tuple[np.ndarray, float, np.ndarray]) -> None:
    benefit, budget, cost = problem
    result = allocate(benefit, budget, cost)
    optimum = _brute_force(benefit, budget, cost)
    assert result.spend <= budget + 1e-9
    assert result.benefit <= optimum + 1e-9
    assert optimum <= result.upper_bound + 1e-9
    # Never more than one customer's benefit short of the optimum ...
    assert optimum - result.benefit <= max(0.0, float(benefit.max())) + 1e-9
    # ... and nobody left out could still be added: no positive-benefit arm fits what is left.
    remaining = budget - result.spend
    for i in np.flatnonzero(result.assign == 0):
        assert not ((benefit[i] > 0) & (cost <= remaining + 1e-9)).any()


def test_fill_falls_back_to_a_cheaper_arm_that_fits() -> None:
    # The arm preferred at the shadow price costs 2 and the budget is 1; arm 1 costs 1 and still helps.
    result = allocate(np.array([[0.5, 2.0]]), 1.0, np.array([1.0, 2.0]))
    np.testing.assert_array_equal(result.assign, [1])
    assert result.benefit == pytest.approx(0.5)


def test_allocation_skips_negative_benefit_and_respects_caps() -> None:
    benefit = np.array([[0.3, 0.1], [-0.2, -0.1], [0.05, 0.2], [0.1, 0.0]])
    everyone = allocate(benefit, budget=10)
    np.testing.assert_array_equal(everyone.assign, [1, 0, 2, 1])
    assert everyone.shadow_price == 0.0
    two = allocate(benefit, budget=2)
    np.testing.assert_array_equal(two.assign, [1, 0, 2, 0])
    assert allocate(-np.abs(benefit), budget=3).spend == 0
    with pytest.raises(ValueError, match="costs"):
        allocate(benefit, 1, costs=np.array([1.0, 0.0]))
