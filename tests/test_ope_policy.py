from __future__ import annotations

import itertools

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from causal_uplift.ope import aipw_scores, bootstrap_ratios, ipw_scores, one_hot_policy, policy_value, random_policy
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
    first = bootstrap_ratios(num, den, n_boot=300, seed=4)
    assert first[0] == first[1] == first[2]  # identical ratios in every replicate
    assert first[0].value == pytest.approx(values.mean())
    assert first[0].lo < values.mean() < first[0].hi
    assert bootstrap_ratios(num, den, n_boot=300, seed=4) == first
    with pytest.raises(ValueError, match="same shape"):
        bootstrap_ratios(num, den[:, :2], n_boot=10, seed=0)


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
