"""Base learners (LightGBM or scikit-learn linear models) used inside the meta-learners."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

import lightgbm as lgb
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

BaseKind = Literal["lightgbm", "linear"]
BASE_KINDS: tuple[BaseKind, ...] = ("lightgbm", "linear")

# Deliberately shallow and slow: effects are a few points on a noisy binary outcome, and the
# second stages of X/R/DR regress noisy pseudo-outcomes. Set by hand and shared by every
# learner and data set; there is no recorded search behind these values.
LIGHTGBM_DEFAULTS: dict[str, Any] = {
    "n_estimators": 300,
    "learning_rate": 0.03,
    "num_leaves": 4,
    "min_child_samples": 200,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "verbose": -1,
    "deterministic": True,
    "force_row_wise": True,
}


class MeanModel(Protocol):
    def predict(self, X: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class BaseLearner:
    """Factory for the regressors and classifiers a meta-learner needs."""

    kind: BaseKind = "lightgbm"
    params: dict[str, Any] = field(default_factory=dict)
    seed: int = 0
    n_jobs: int = 1

    def __post_init__(self) -> None:
        if self.kind not in BASE_KINDS:
            raise ValueError(f"base learner must be one of {BASE_KINDS}, got {self.kind!r}")

    def with_seed(self, seed: int) -> BaseLearner:
        return BaseLearner(kind=self.kind, params=self.params, seed=seed, n_jobs=self.n_jobs)

    def regressor(self) -> lgb.LGBMRegressor | Pipeline:
        if self.kind == "lightgbm":
            return lgb.LGBMRegressor(**self._lgbm_params())
        return make_pipeline(StandardScaler(), Ridge(alpha=self.params.get("alpha", 1.0)))

    def classifier(self) -> lgb.LGBMClassifier | Pipeline:
        if self.kind == "lightgbm":
            return lgb.LGBMClassifier(**self._lgbm_params())
        return make_pipeline(StandardScaler(), LogisticRegression(C=self.params.get("C", 1.0), max_iter=2000))

    def _lgbm_params(self) -> dict[str, Any]:
        return {**LIGHTGBM_DEFAULTS, **self.params, "random_state": self.seed, "n_jobs": self.n_jobs}

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "params": dict(self.params), "seed": self.seed, "n_jobs": self.n_jobs}


class _Constant:
    def __init__(self, value: float) -> None:
        self.value = value

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full(X.shape[0], self.value)


class _Probability:
    def __init__(self, model: lgb.LGBMClassifier | Pipeline) -> None:
        self.model = model

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]


class _Regression:
    def __init__(self, model: lgb.LGBMRegressor | Pipeline) -> None:
        self.model = model

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self.model.predict(X), dtype=np.float64)


def is_binary(y: np.ndarray) -> bool:
    return bool(np.isin(y, (0.0, 1.0)).all())


def _fit_kwargs(model: object, sample_weight: np.ndarray | None) -> dict[str, Any]:
    if sample_weight is None:
        return {}
    if isinstance(model, Pipeline):
        return {f"{model.steps[-1][0]}__sample_weight": sample_weight}
    return {"sample_weight": sample_weight}


def fit_mean(
    base: BaseLearner,
    X: np.ndarray,
    y: np.ndarray,
    *,
    binary: bool,
    sample_weight: np.ndarray | None = None,
) -> MeanModel:
    """Fit E[y | X]; binary targets use a classifier so predictions stay in [0, 1]."""
    if y.size == 0:
        raise ValueError("cannot fit an outcome model on zero rows")
    if binary:
        if np.unique(y).size < 2:
            return _Constant(float(y[0]))
        clf = base.classifier()
        clf.fit(X, y.astype(np.int64), **_fit_kwargs(clf, sample_weight))
        return _Probability(clf)
    reg = base.regressor()
    reg.fit(X, y, **_fit_kwargs(reg, sample_weight))
    return _Regression(reg)


PropensityKind = Literal["constant", "logistic"]
PROPENSITY_KINDS: tuple[PropensityKind, ...] = ("constant", "logistic")


class Propensity:
    """P(arm = a | X). ``constant`` uses arm shares, which is exact for a randomised test."""

    def __init__(self, kind: PropensityKind = "constant", clip: float = 0.01) -> None:
        if kind not in PROPENSITY_KINDS:
            raise ValueError(f"propensity must be one of {PROPENSITY_KINDS}, got {kind!r}")
        self.kind = kind
        self.clip = clip

    def fit(self, X: np.ndarray, labels: np.ndarray, n_classes: int) -> Propensity:
        self.n_classes_ = n_classes
        self.shares_ = np.bincount(labels, minlength=n_classes) / labels.size
        if self.kind == "logistic":
            self.model_ = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
            self.model_.fit(X, labels)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.kind == "constant":
            probs = np.broadcast_to(self.shares_, (X.shape[0], self.n_classes_)).copy()
        else:
            probs = np.zeros((X.shape[0], self.n_classes_))
            probs[:, self.model_.classes_] = self.model_.predict_proba(X)
        probs = np.clip(probs, self.clip, 1.0 - self.clip)
        return probs / probs.sum(axis=1, keepdims=True)
