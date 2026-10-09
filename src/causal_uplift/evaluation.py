"""Cross-fitted evaluation: ranking quality, known-effect error and budgeted policy value.

Every number is computed on rows the learner did not see: the data are split into folds,
each learner is fitted on all but one fold and scores the held-out fold. Budgets are applied
within each held-out fold, and the AIPW outcome models are cross-fitted the same way.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from causal_uplift.base import BaseLearner
from causal_uplift.crossfit import experiment_strata, splits, stratified_folds
from causal_uplift.data import UpliftData
from causal_uplift.learners import CateLearner, TLearner
from causal_uplift.metrics import Interval, RankingBootstrap, qini_curve
from causal_uplift.ope import aipw_scores, bootstrap_ratios, ipw_scores, one_hot_policy, random_policy
from causal_uplift.policy import allocate

LearnerFactory = Callable[[int], CateLearner]
RANDOM_POLICY = "random targeting"


@dataclass
class CrossFitted:
    cate: np.ndarray
    fit_seconds: list[float] = field(default_factory=list)
    predict_seconds: list[float] = field(default_factory=list)


def cross_fit(data: UpliftData, factory: LearnerFactory, fold: np.ndarray) -> CrossFitted:
    out = CrossFitted(cate=np.zeros((data.n, data.n_treatments)))
    n_arms = len(data.arm_names)
    for k, (train, test) in enumerate(splits(fold)):
        learner = factory(k)
        start = time.perf_counter()
        learner.fit(data.X[train], data.arm[train], data.y[train], n_arms)
        out.fit_seconds.append(time.perf_counter() - start)
        start = time.perf_counter()
        out.cate[test] = learner.predict(data.X[test])
        out.predict_seconds.append(time.perf_counter() - start)
    return out


def cross_fit_outcomes(data: UpliftData, base: BaseLearner, fold: np.ndarray) -> np.ndarray:
    """Held-out predictions of E[Y | X, arm = a] for every arm: the AIPW outcome model."""
    mu = np.zeros((data.n, len(data.arm_names)))
    for train, test in splits(fold):
        model = TLearner(base).fit(data.X[train], data.arm[train], data.y[train], len(data.arm_names))
        mu[test] = model.predict_outcomes(data.X[test])
    return mu


def pehe(true_cate: np.ndarray, cate: np.ndarray, treatment_names: Sequence[str]) -> dict[str, float]:
    """Root mean squared error of the effect estimates, overall and per treatment."""
    err = (cate - true_cate) ** 2
    out = {"all": float(np.sqrt(err.mean()))}
    out.update({name: float(np.sqrt(err[:, k].mean())) for k, name in enumerate(treatment_names)})
    return out


def ranking_metrics(
    data: UpliftData, cate: np.ndarray, *, n_boot: int, seed: int, curve_points: int = 51
) -> dict[str, Any]:
    """Per treatment arm: Qini coefficient and AUUC with bootstrap intervals, plus a Qini curve."""
    out: dict[str, Any] = {}
    for k, arm_name in enumerate(data.arm_names[1:], start=1):
        rows = np.flatnonzero((data.arm == 0) | (data.arm == k))
        y = data.y[rows]
        treated = (data.arm[rows] == k).astype(np.float64)
        score = cate[rows, k - 1]
        intervals = RankingBootstrap(y, score, treated).run(n_boot, seed + k)
        curve = qini_curve(y, score, treated).resample(curve_points)
        out[arm_name] = {
            "qini": intervals["qini"].to_dict(),
            "auuc": intervals["auuc"].to_dict(),
            "curve": {
                "fraction": (curve.n_targeted / rows.size).round(4).tolist(),
                "incremental_per_customer": (curve.value / rows.size).round(6).tolist(),
            },
        }
    return out


def _fold_allocation(cate: np.ndarray, fold: np.ndarray, budget: float, costs: np.ndarray | None) -> np.ndarray:
    assign = np.zeros(cate.shape[0], dtype=np.int64)
    for _, test in splits(fold):
        assign[test] = allocate(cate[test], budget * test.size, costs).assign
    return assign


def policy_metrics(
    data: UpliftData,
    cates: Mapping[str, np.ndarray],
    mu: np.ndarray,
    fold: np.ndarray,
    *,
    budgets: Sequence[float],
    costs: np.ndarray | None,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    """AIPW and IPW value of each learner's budgeted policy, per 1,000 customers and per 1,000 contacts.

    Budgets are a share of customers when costs are 1 (the default); with costs the budget
    is ``share * n`` cost units. ``random targeting`` is scored in expectation over its
    randomisation, and every policy also gets a paired AIPW difference against it.
    """
    n_arms = len(data.arm_names)
    propensity = data.design_propensity()
    gamma_aipw = aipw_scores(data.arm, data.y, propensity, mu)
    gamma_ipw = ipw_scores(data.arm, data.y, propensity)
    out: dict[str, Any] = {}
    for b, budget in enumerate(budgets):
        policies: dict[str, np.ndarray] = {}
        for name, cate in cates.items():
            if name != "random":
                policies[name] = one_hot_policy(_fold_allocation(cate, fold, budget, costs), n_arms)
        mean_cost = 1.0 if costs is None else float(np.mean(costs))
        policies[RANDOM_POLICY] = random_policy(data.n, n_arms, min(1.0, budget / mean_cost))
        names = list(policies)
        ones = np.ones(data.n)
        numerators: list[np.ndarray] = []
        denominators: list[np.ndarray] = []
        reference = np.sum(policies[RANDOM_POLICY] * gamma_aipw, axis=1)
        for name in names:
            pi = policies[name]
            contact = 1.0 - pi[:, 0]
            aipw_lift = np.sum(pi * gamma_aipw, axis=1) - gamma_aipw[:, 0]
            ipw_lift = np.sum(pi * gamma_ipw, axis=1) - gamma_ipw[:, 0]
            gain = np.sum(pi * gamma_aipw, axis=1) - reference
            numerators += [1000 * aipw_lift, 1000 * aipw_lift, 1000 * ipw_lift, 1000 * gain]
            denominators += [ones, contact, contact, ones]
        intervals = bootstrap_ratios(np.column_stack(numerators), np.column_stack(denominators), n_boot, seed + b)
        per_budget: dict[str, Any] = {}
        for j, name in enumerate(names):
            pi = policies[name]
            block: list[Interval] = intervals[4 * j : 4 * j + 4]
            per_budget[name] = {
                "contact_rate": float(np.mean(1.0 - pi[:, 0])),
                "arm_mix": {arm: float(pi[:, a].mean()) for a, arm in enumerate(data.arm_names)},
                "aipw_per_1000_customers": block[0].to_dict(),
                "aipw_per_1000_contacts": block[1].to_dict(),
                "ipw_per_1000_contacts": block[2].to_dict(),
                "aipw_gain_vs_random_per_1000_customers": block[3].to_dict(),
            }
        out[f"{budget:g}"] = per_budget
    return out


def describe(data: UpliftData) -> dict[str, Any]:
    counts = np.bincount(data.arm, minlength=len(data.arm_names))
    means = [float(data.y[data.arm == a].mean()) for a in range(len(data.arm_names))]
    return {
        "rows": data.n,
        "outcome": data.outcome,
        "binary_outcome": data.binary_outcome,
        "features": list(data.feature_names),
        "arms": list(data.arm_names),
        "rows_per_arm": dict(zip(data.arm_names, counts.tolist(), strict=True)),
        "outcome_mean_per_arm": dict(zip(data.arm_names, means, strict=True)),
    }


def evaluate(
    data: UpliftData,
    learners: Mapping[str, LearnerFactory],
    *,
    base: BaseLearner,
    n_folds: int = 5,
    n_boot: int = 1000,
    seed: int = 0,
    budgets: Sequence[float] | None = None,
    costs: np.ndarray | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Cross-fit every learner and collect the metrics into a JSON-ready dict."""
    say = progress or (lambda _msg: None)
    fold = stratified_folds(experiment_strata(data.arm, data.y), n_folds, seed)
    results: dict[str, Any] = {
        "dataset": describe(data),
        "settings": {"folds": n_folds, "n_boot": n_boot, "seed": seed, "base": base.to_dict()},
        "learners": {},
    }
    cates: dict[str, np.ndarray] = {}
    for name, factory in learners.items():
        say(f"cross-fitting {name}")
        fitted = cross_fit(data, factory, fold)
        cates[name] = fitted.cate
        entry: dict[str, Any] = {
            "fit_seconds_per_fold": [round(s, 4) for s in fitted.fit_seconds],
            "fit_seconds_mean": float(np.mean(fitted.fit_seconds)),
            "predict_seconds_mean": float(np.mean(fitted.predict_seconds)),
            "ranking": ranking_metrics(data, fitted.cate, n_boot=n_boot, seed=seed),
        }
        if data.true_cate is not None:
            entry["pehe"] = pehe(data.true_cate, fitted.cate, data.arm_names[1:])
        results["learners"][name] = entry
    if budgets:
        say("cross-fitting AIPW outcome models")
        mu = cross_fit_outcomes(data, base, fold)
        results["settings"]["budgets"] = list(budgets)
        results["settings"]["costs"] = None if costs is None else np.asarray(costs).tolist()
        say("scoring budgeted policies")
        results["policy"] = policy_metrics(
            data, cates, mu, fold, budgets=budgets, costs=costs, n_boot=n_boot, seed=seed
        )
    return results
