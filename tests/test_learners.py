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


def _continuous_linear(n: int, seed: int) -> tuple[UpliftData, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 4))
    arm = rng.integers(0, 3, n)
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
