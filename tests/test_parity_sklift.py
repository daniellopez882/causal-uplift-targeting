"""Parity with scikit-uplift, used here only as a reference implementation."""

from __future__ import annotations

import numpy as np
import pytest
from sklift import metrics as sk

from bench.baselines import SkliftTwoModels
from causal_uplift.base import BaseLearner
from causal_uplift.learners import TLearner
from causal_uplift.metrics import (
    auuc,
    perfect_qini_curve,
    perfect_uplift_curve,
    qini_coefficient,
    qini_curve,
    uplift_curve,
)
from causal_uplift.synthetic import make_synthetic


def _case(seed: int, ties: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = 500
    treated = rng.integers(0, 2, n)
    y = (rng.random(n) < 0.25 + 0.1 * treated).astype(int)
    score = rng.integers(0, 6, n).astype(float) if ties else rng.normal(size=n)
    return y, score, treated


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("ties", [False, True])
def test_curves_match_scikit_uplift(seed: int, ties: bool) -> None:
    y, score, treated = _case(seed, ties)
    for ours, theirs in ((qini_curve, sk.qini_curve), (uplift_curve, sk.uplift_curve)):
        curve = ours(y, score, treated)
        x_ref, v_ref = theirs(y, score, treated)
        np.testing.assert_allclose(curve.n_targeted, x_ref)
        np.testing.assert_allclose(curve.value, v_ref, atol=1e-9)
    perfect = perfect_qini_curve(y, treated)
    x_ref, v_ref = sk.perfect_qini_curve(y, treated)
    np.testing.assert_allclose(perfect.n_targeted, x_ref)
    np.testing.assert_allclose(perfect.value, v_ref, atol=1e-9)
    perfect = perfect_uplift_curve(y, treated)
    x_ref, v_ref = sk.perfect_uplift_curve(y, treated)
    np.testing.assert_allclose(perfect.value, v_ref, atol=1e-9)


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("ties", [False, True])
def test_normalised_scores_match_scikit_uplift(seed: int, ties: bool) -> None:
    y, score, treated = _case(seed, ties)
    assert qini_coefficient(y, score, treated) == pytest.approx(sk.qini_auc_score(y, score, treated), abs=1e-12)
    assert auuc(y, score, treated) == pytest.approx(sk.uplift_auc_score(y, score, treated), abs=1e-12)


def test_t_learner_matches_scikit_uplift_two_models() -> None:
    data = make_synthetic("step", 3000, seed=8)
    base = BaseLearner(params={"n_estimators": 60}, n_jobs=1)
    ours = TLearner(base).fit(data.X, data.arm, data.y).predict(data.X)
    theirs = SkliftTwoModels(base).fit(data.X, data.arm, data.y).predict(data.X)
    np.testing.assert_allclose(ours, theirs, atol=1e-12)
