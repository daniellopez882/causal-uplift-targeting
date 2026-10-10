from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from causal_uplift.base import BaseLearner, Propensity, fit_mean
from causal_uplift.crossfit import experiment_strata, splits, stratified_folds
from causal_uplift.data import UpliftData
from causal_uplift.learners import DRLearner, RLearner, SLearner, TLearner, XLearner
from causal_uplift.metrics import qini_coefficient
from causal_uplift.registry import LEARNER_NAMES, LearnerConfig, make_learner
from causal_uplift.synthetic import make_synthetic

META = ("s", "t", "x", "r", "dr")
FORESTS = ("forest-kl", "forest-ed")


def _pehe(estimate: np.ndarray, truth: np.ndarray) -> float:
    return float(np.sqrt(((estimate - truth) ** 2).mean()))


@pytest.fixture(scope="module")
def sleeping_dogs() -> tuple[UpliftData, UpliftData]:
    return make_synthetic("sleeping_dogs", 16_000, seed=11), make_synthetic("sleeping_dogs", 8_000, seed=12)


def _constant_effect_pehe(train: UpliftData, test: UpliftData) -> float:
    assert test.true_cate is not None
    ate = np.array([train.y[train.arm == a].mean() - train.y[train.arm == 0].mean() for a in (1, 2)])
    return _pehe(np.broadcast_to(ate, test.true_cate.shape), test.true_cate)


@pytest.mark.parametrize("name", META + FORESTS)
def test_learner_recovers_heterogeneous_effects(
    name: str, sleeping_dogs: tuple[UpliftData, UpliftData], small_base: BaseLearner
) -> None:
    train, test = sleeping_dogs
    assert test.true_cate is not None
    config = LearnerConfig(base=small_base, forest={"n_trees": 50, "min_samples_leaf": 50})
    learner = make_learner(name, config).fit(train.X, train.arm, train.y)
    estimate = learner.predict(test.X)
    assert estimate.shape == (test.n, 2)
    bound = 0.6 if name in FORESTS else 0.85
    assert _pehe(estimate, test.true_cate) < bound * _constant_effect_pehe(train, test)


def test_response_model_fails_by_construction(
    sleeping_dogs: tuple[UpliftData, UpliftData], small_base: BaseLearner
) -> None:
    """Customers most likely to buy are the ones the e-mail puts off, so ranking by
    purchase probability ranks backwards while an uplift learner ranks forwards."""
    train, test = sleeping_dogs
    config = LearnerConfig(base=small_base)
    response = make_learner("response", config).fit(train.X, train.arm, train.y).predict(test.X)
    uplift = make_learner("t", config).fit(train.X, train.arm, train.y).predict(test.X)
    rows = test.arm <= 1
    treated = (test.arm[rows] == 1).astype(float)
    assert qini_coefficient(test.y[rows], response[rows, 0], treated) < 0
    assert qini_coefficient(test.y[rows], uplift[rows, 0], treated) > 0


UNEQUAL_SHARES = np.array([0.7, 0.2, 0.1])


def _continuous_linear(n: int, seed: int, assignment: str = "uniform") -> tuple[UpliftData, np.ndarray]:
    """Linear outcome and effects. ``uniform`` draws arms 1/3 each, ``unequal`` with 70/20/10 odds,
    ``confounded`` with odds that move with x0, which also drives the outcome."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 4))
    if assignment == "uniform":
        arm = rng.integers(0, 3, n)
    else:
        if assignment == "unequal":
            probs = np.broadcast_to(UNEQUAL_SHARES, (n, 3))
        else:
            odds = np.exp(np.column_stack([np.zeros(n), X[:, 0], -X[:, 0]]))
            probs = odds / odds.sum(axis=1, keepdims=True)
        arm = (rng.random(n)[:, None] > probs.cumsum(axis=1)).sum(axis=1)
    tau = np.column_stack([0.5 + X[:, 1], -0.3 + 0.5 * X[:, 2]])
    effect = np.where(arm == 0, 0.0, tau[np.arange(n), np.maximum(arm - 1, 0)])
    y = 1.0 + X[:, 0] + effect + rng.normal(0, 0.5, n)
    names = ("x0", "x1", "x2", "x3")
    return UpliftData(X, arm, y, names, ("c", "a", "b"), "y", true_cate=tau), tau


@pytest.mark.parametrize("cls", [TLearner, XLearner, RLearner, DRLearner])
def test_linear_base_is_nearly_exact_when_effects_are_linear(cls: type) -> None:
    train, _ = _continuous_linear(6000, 1)
    test, tau = _continuous_linear(2000, 2)
    estimate = cls(BaseLearner(kind="linear")).fit(train.X, train.arm, train.y).predict(test.X)
    assert _pehe(estimate, tau) < 0.1


@pytest.mark.parametrize("cls", [XLearner, RLearner, DRLearner])
@pytest.mark.parametrize(("assignment", "propensity"), [("unequal", "constant"), ("confounded", "logistic")])
def test_linear_base_stays_nearly_exact_when_arms_are_unbalanced(cls: type, assignment: str, propensity: str) -> None:
    train, _ = _continuous_linear(6000, 1, assignment)
    test, tau = _continuous_linear(2000, 2, assignment)
    estimate = cls(BaseLearner(kind="linear"), propensity).fit(train.X, train.arm, train.y).predict(test.X)
    assert _pehe(estimate, tau) < 0.1


def test_r_learner_needs_the_propensity_when_assignment_is_confounded() -> None:
    """With odds that move with x0, residualising on constant arm shares leaves confounding in."""
    train, _ = _continuous_linear(6000, 1, "confounded")
    test, tau = _continuous_linear(2000, 2, "confounded")
    errors = {
        kind: _pehe(RLearner(BaseLearner(kind="linear"), kind).fit(train.X, train.arm, train.y).predict(test.X), tau)
        for kind in ("constant", "logistic")
    }
    assert errors["logistic"] < 0.8 * errors["constant"]


def test_x_learner_weights_the_control_side_model_by_the_propensity() -> None:
    """g = e_a / (e_a + e_0) multiplies the effect model fitted on control rows (Kuenzel et al. 2019)."""
    train, _ = _continuous_linear(6000, 1, "unequal")
    learner = XLearner(BaseLearner(kind="linear")).fit(train.X, train.arm, train.y)
    X = train.X[:50]
    shares = np.bincount(train.arm, minlength=3) / train.n
    expected = []
    for a, (tau_treated, tau_control) in enumerate(learner.effect_models_, start=1):
        g = shares[a] / (shares[a] + shares[0])
        assert abs(g - 0.5) > 0.2  # far from the balanced case, where swapping the two models changes nothing
        expected.append(g * tau_control.predict(X) + (1.0 - g) * tau_treated.predict(X))
    np.testing.assert_allclose(learner.predict(X), np.column_stack(expected), rtol=1e-6, atol=1e-9)


@pytest.mark.parametrize("name", ["x", "r", "dr"])
def test_learner_recovers_effects_under_confounded_assignment(name: str, small_base: BaseLearner) -> None:
    """The PEHE bound with estimated propensities: assignment odds depend on x0, which also moves
    the outcome, so the naive difference in means (the constant-effect reference) is biased."""
    train, test = make_synthetic("confounded", 16_000, seed=11), make_synthetic("confounded", 8_000, seed=12)
    assert test.true_cate is not None
    learner = make_learner(name, LearnerConfig(base=small_base, propensity="logistic"))
    estimate = learner.fit(train.X, train.arm, train.y).predict(test.X)
    assert _pehe(estimate, test.true_cate) < 0.85 * _constant_effect_pehe(train, test)


def test_s_learner_with_linear_base_cannot_express_heterogeneity() -> None:
    train, _ = _continuous_linear(6000, 1)
    test, tau = _continuous_linear(2000, 2)
    estimate = SLearner(BaseLearner(kind="linear")).fit(train.X, train.arm, train.y).predict(test.X)
    assert np.ptp(estimate[:, 0]) < 1e-8  # arm indicators enter additively, so the effect is constant
    assert _pehe(estimate, tau) > 0.5


@pytest.mark.parametrize("name", LEARNER_NAMES)
def test_every_registered_learner_handles_a_single_treatment(name: str, small_base: BaseLearner) -> None:
    data = make_synthetic("step", 3000, seed=3)
    keep = data.arm < 2
    learner = make_learner(name, LearnerConfig(base=small_base, forest={"n_trees": 5, "min_samples_leaf": 20}))
    learner.fit(data.X[keep], data.arm[keep], data.y[keep])
    assert learner.predict(data.X[:10]).shape == (10, 1)


def test_unknown_learner_and_missing_arm_are_rejected() -> None:
    with pytest.raises(ValueError, match="unknown learner"):
        make_learner("causal-forest")
    X = np.zeros((4, 1))
    with pytest.raises(ValueError, match="every arm"):
        TLearner().fit(X, np.array([0, 0, 2, 2]), np.array([0.0, 1.0, 0.0, 1.0]))


def test_fit_mean_handles_single_class_and_weights(small_base: BaseLearner) -> None:
    X = np.random.default_rng(0).standard_normal((400, 2))
    assert np.all(fit_mean(small_base, X, np.ones(400), binary=True).predict(X) == 1.0)
    # Weights must reach the final step of the scaler + ridge pipeline: rows with
    # near-zero weight carry a corrupted target that the fit should ignore.
    clean = X[:, 0] * 2.0
    corrupted = clean + np.where(np.arange(400) < 200, 10.0, 0.0)
    weight = np.where(np.arange(400) < 200, 1e-9, 1.0)
    model = fit_mean(BaseLearner(kind="linear"), X, corrupted, binary=False, sample_weight=weight)
    np.testing.assert_allclose(model.predict(X), clean, atol=0.1)


def test_logistic_propensity_tracks_confounding() -> None:
    data = make_synthetic("confounded", 6000, seed=2)
    assert data.propensity is not None
    estimated = Propensity("logistic").fit(data.X, data.arm, 3).predict(data.X)
    assert np.abs(estimated - data.propensity).mean() < 0.03
    constant = Propensity("constant").fit(data.X, data.arm, 3).predict(data.X)
    assert np.ptp(constant[:, 1]) == 0.0


@given(
    n=st.integers(min_value=10, max_value=400),
    n_folds=st.integers(min_value=2, max_value=7),
    n_strata=st.integers(min_value=1, max_value=6),
    seed=st.integers(min_value=0, max_value=10_000),
)
def test_cross_fit_folds_are_disjoint_and_balanced(n: int, n_folds: int, n_strata: int, seed: int) -> None:
    strata = np.random.default_rng(seed).integers(0, n_strata, n)
    fold = stratified_folds(strata, n_folds, seed)
    seen = np.zeros(n, dtype=int)
    for train, test in splits(fold):
        assert np.intersect1d(train, test).size == 0
        assert train.size + test.size == n
        seen[test] += 1
    assert np.all(seen == 1)
    for value in np.unique(strata):
        per_fold = np.bincount(fold[strata == value], minlength=n_folds)
        assert per_fold.max() - per_fold.min() <= 1


def test_experiment_strata_separate_arm_and_outcome() -> None:
    strata = experiment_strata(np.array([0, 0, 1, 1, 2]), np.array([0.0, 1.0, 0.0, 3.0, 0.0]))
    np.testing.assert_array_equal(strata, [0, 1, 2, 3, 4])
    with pytest.raises(ValueError, match="n_folds"):
        stratified_folds(strata, 1, 0)
