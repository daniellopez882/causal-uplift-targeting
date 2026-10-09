"""scikit-uplift's two-model approach, wrapped in this package's learner contract.

Used only as a benchmark baseline and a parity reference in the tests.
"""

from __future__ import annotations

import warnings

import numpy as np
from sklift.models import TwoModels

from causal_uplift.base import BaseLearner
from causal_uplift.learners import CateLearner


class SkliftTwoModels(CateLearner):
    """One ``sklift.models.TwoModels(method='vanilla')`` per treatment-versus-control pair."""

    name = "sklift-two-model"

    def __init__(self, base: BaseLearner | None = None) -> None:
        self.base = base or BaseLearner()

    def _fit(self, X: np.ndarray, arm: np.ndarray, y: np.ndarray) -> None:
        self.models_: list[TwoModels] = []
        for a in range(1, self.n_arms_):
            rows = np.flatnonzero((arm == 0) | (arm == a))
            model = TwoModels(
                estimator_trmnt=self.base.classifier(), estimator_ctrl=self.base.classifier(), method="vanilla"
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                model.fit(X[rows], y[rows].astype(np.int64), (arm[rows] == a).astype(np.int64))
            self.models_.append(model)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.column_stack([m.predict(X) for m in self.models_])
