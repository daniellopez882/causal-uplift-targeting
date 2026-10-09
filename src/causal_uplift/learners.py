"""CATE meta-learners (S, T, X, R, DR) and the two ranking baselines.

All learners share one contract: ``fit(X, arm, y)`` with arm 0 = control and 1..K the
treatments, then ``predict(X)`` returns an (n, K) matrix whose column k-1 estimates
E[Y(k) - Y(0) | X]. Multi-arm problems are handled per treatment-versus-control pair,
except for the S-learner (one model with arm indicators) and the forest (native).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar, Self

import numpy as np

from causal_uplift.base import BaseLearner, MeanModel, Propensity, PropensityKind, fit_mean, is_binary
from causal_uplift.crossfit import experiment_strata, splits, stratified_folds


class CateLearner(ABC):
    name: ClassVar[str]

    def fit(self, X: np.ndarray, arm: np.ndarray, y: np.ndarray, n_arms: int | None = None) -> Self:
        arm = np.asarray(arm, dtype=np.int64)
        n_arms = int(arm.max()) + 1 if n_arms is None else n_arms
        if n_arms < 2:
            raise ValueError("need a control arm and at least one treatment")
        if (np.bincount(arm, minlength=n_arms) == 0).any():
            raise ValueError("every arm needs at least one row in the training data")
        self.n_arms_ = n_arms
        self._fit(np.asarray(X, dtype=np.float64), arm, np.asarray(y, dtype=np.float64))
        return self

    @abstractmethod
    def _fit(self, X: np.ndarray, arm: np.ndarray, y: np.ndarray) -> None: ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray: ...


class SLearner(CateLearner):
    """One outcome model on [X, arm indicators]; effect = prediction(arm=k) - prediction(arm=0)."""

    name = "s"

    def __init__(self, base: BaseLearner | None = None) -> None:
        self.base = base or BaseLearner()

    def _design(self, X: np.ndarray, arm: np.ndarray) -> np.ndarray:
        indicators = np.zeros((X.shape[0], self.n_arms_ - 1))
        treated = np.flatnonzero(arm > 0)
        indicators[treated, arm[treated] - 1] = 1.0
        return np.hstack([X, indicators])

    def _fit(self, X: np.ndarray, arm: np.ndarray, y: np.ndarray) -> None:
        self.model_ = fit_mean(self.base, self._design(X, arm), y, binary=is_binary(y))

    def predict(self, X: np.ndarray) -> np.ndarray:
        n = X.shape[0]
        mu0 = self.model_.predict(self._design(X, np.zeros(n, dtype=np.int64)))
        return np.column_stack(
            [self.model_.predict(self._design(X, np.full(n, a, dtype=np.int64))) - mu0 for a in range(1, self.n_arms_)]
        )


class TLearner(CateLearner):
    """A separate outcome model per arm; effect = mu_k(x) - mu_0(x)."""

    name = "t"

    def __init__(self, base: BaseLearner | None = None) -> None:
        self.base = base or BaseLearner()

    def _fit(self, X: np.ndarray, arm: np.ndarray, y: np.ndarray) -> None:
        binary = is_binary(y)
        self.models_: list[MeanModel] = [
            fit_mean(self.base, X[arm == a], y[arm == a], binary=binary) for a in range(self.n_arms_)
        ]

    def predict_outcomes(self, X: np.ndarray) -> np.ndarray:
        """(n, K + 1) predicted outcome under every arm, including control."""
        return np.column_stack([m.predict(X) for m in self.models_])

    def predict(self, X: np.ndarray) -> np.ndarray:
        mu = self.predict_outcomes(X)
        return mu[:, 1:] - mu[:, :1]


class XLearner(CateLearner):
    """Kuenzel et al. (2019): impute individual effects with the other arm's model, then blend
    the treated-side and control-side effect models by the propensity."""

    name = "x"

    def __init__(self, base: BaseLearner | None = None, propensity: PropensityKind = "constant") -> None:
        self.base = base or BaseLearner()
        self.propensity = propensity

    def _fit(self, X: np.ndarray, arm: np.ndarray, y: np.ndarray) -> None:
        outcome = TLearner(self.base).fit(X, arm, y, self.n_arms_)
        mu0 = outcome.models_[0]
        control = arm == 0
        self.propensity_ = Propensity(self.propensity).fit(X, arm, self.n_arms_)
        self.effect_models_: list[tuple[MeanModel, MeanModel]] = []
        for a in range(1, self.n_arms_):
            treated = arm == a
            imputed_treated = y[treated] - mu0.predict(X[treated])
            imputed_control = outcome.models_[a].predict(X[control]) - y[control]
            self.effect_models_.append(
                (
                    fit_mean(self.base, X[treated], imputed_treated, binary=False),
                    fit_mean(self.base, X[control], imputed_control, binary=False),
                )
            )

    def predict(self, X: np.ndarray) -> np.ndarray:
        e = self.propensity_.predict(X)
        columns = []
        for a, (tau_treated, tau_control) in enumerate(self.effect_models_, start=1):
            g = e[:, a] / (e[:, a] + e[:, 0])
            columns.append(g * tau_control.predict(X) + (1.0 - g) * tau_treated.predict(X))
        return np.column_stack(columns)


def _pair(arm: np.ndarray, a: int) -> tuple[np.ndarray, np.ndarray]:
    rows = np.flatnonzero((arm == 0) | (arm == a))
    return rows, (arm[rows] == a).astype(np.int64)


class RLearner(CateLearner):
    """Nie and Wager (2021) with cross-fitted nuisances: regress (y - m(x)) / (w - e(x)) on x
    with weights (w - e(x))^2, which minimises the R-loss."""

    name = "r"

    def __init__(
        self,
        base: BaseLearner | None = None,
        propensity: PropensityKind = "constant",
        n_folds: int = 2,
        seed: int = 0,
    ) -> None:
        self.base = base or BaseLearner()
        self.propensity = propensity
        self.n_folds = n_folds
        self.seed = seed

    def _fit(self, X: np.ndarray, arm: np.ndarray, y: np.ndarray) -> None:
        binary = is_binary(y)
        self.effect_models_: list[MeanModel] = []
        for a in range(1, self.n_arms_):
            rows, w = _pair(arm, a)
            Xs, ys = X[rows], y[rows]
            m_hat = np.empty(rows.size)
            e_hat = np.empty(rows.size)
            fold = stratified_folds(experiment_strata(w, ys), self.n_folds, self.seed + a)
            for train, test in splits(fold):
                m_hat[test] = fit_mean(self.base, Xs[train], ys[train], binary=binary).predict(Xs[test])
                e_hat[test] = Propensity(self.propensity).fit(Xs[train], w[train], 2).predict(Xs[test])[:, 1]
            y_res = ys - m_hat
            w_res = w - e_hat
            self.effect_models_.append(fit_mean(self.base, Xs, y_res / w_res, binary=False, sample_weight=w_res**2))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.column_stack([m.predict(X) for m in self.effect_models_])


class DRLearner(CateLearner):
    """Kennedy (2023): regress the cross-fitted AIPW pseudo-outcome on x."""

    name = "dr"

    def __init__(
        self,
        base: BaseLearner | None = None,
        propensity: PropensityKind = "constant",
        n_folds: int = 2,
        seed: int = 0,
    ) -> None:
        self.base = base or BaseLearner()
        self.propensity = propensity
        self.n_folds = n_folds
        self.seed = seed

    def _fit(self, X: np.ndarray, arm: np.ndarray, y: np.ndarray) -> None:
        binary = is_binary(y)
        self.effect_models_: list[MeanModel] = []
        for a in range(1, self.n_arms_):
            rows, w = _pair(arm, a)
            Xs, ys = X[rows], y[rows]
            pseudo = np.empty(rows.size)
            fold = stratified_folds(experiment_strata(w, ys), self.n_folds, self.seed + a)
            for train, test in splits(fold):
                ctrl = train[w[train] == 0]
                trt = train[w[train] == 1]
                mu0 = fit_mean(self.base, Xs[ctrl], ys[ctrl], binary=binary).predict(Xs[test])
                mu1 = fit_mean(self.base, Xs[trt], ys[trt], binary=binary).predict(Xs[test])
                e = Propensity(self.propensity).fit(Xs[train], w[train], 2).predict(Xs[test])[:, 1]
                wt, yt = w[test], ys[test]
                pseudo[test] = mu1 - mu0 + wt * (yt - mu1) / e - (1 - wt) * (yt - mu0) / (1.0 - e)
            self.effect_models_.append(fit_mean(self.base, Xs, pseudo, binary=False))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.column_stack([m.predict(X) for m in self.effect_models_])


class ResponseModel(CateLearner):
    """Baseline: predicted purchase probability if contacted with arm k.

    This is the usual propensity-to-buy ranking. It ignores what the customer would have
    done without the contact, so it targets people who buy anyway.
    """

    name = "response"

    def __init__(self, base: BaseLearner | None = None) -> None:
        self.base = base or BaseLearner()

    def _fit(self, X: np.ndarray, arm: np.ndarray, y: np.ndarray) -> None:
        binary = is_binary(y)
        self.models_ = [fit_mean(self.base, X[arm == a], y[arm == a], binary=binary) for a in range(1, self.n_arms_)]

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.column_stack([m.predict(X) for m in self.models_])


class RandomScores(CateLearner):
    """Baseline: uniform random scores, i.e. random targeting with a random arm."""

    name = "random"

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed

    def _fit(self, X: np.ndarray, arm: np.ndarray, y: np.ndarray) -> None:
        self.rng_ = np.random.default_rng(self.seed)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.rng_.random((X.shape[0], self.n_arms_ - 1))
