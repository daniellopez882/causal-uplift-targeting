"""Cross-fitted evaluation: ranking quality, known-effect error and budgeted policy value.

Every number is computed on rows the learner did not see: the data are split into folds,
each learner is fitted on all but one fold and scores the held-out fold. Budgets are applied
within each held-out fold, and the AIPW outcome models (and estimated propensities, when
assignment probabilities are not known) are cross-fitted the same way.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from causal_uplift.base import BaseLearner, Propensity, PropensityKind
from causal_uplift.crossfit import experiment_strata, splits, stratified_folds
from causal_uplift.data import UpliftData
from causal_uplift.learners import CateLearner, TLearner
from causal_uplift.metrics import Interval, RankingBootstrap, qini_curve
from causal_uplift.ope import (
    aipw_scores,
    bootstrap_ratio_draws,
    ipw_scores,
    one_hot_policy,
    percentile_intervals,
    random_policy,
    simultaneous_intervals,
    single_arm_policy,
)
from causal_uplift.policy import allocate

LearnerFactory = Callable[[int], CateLearner]
RANDOM_POLICY = "random targeting"
BEST_ARM_POLICY = "best arm at random"
REFERENCE_POLICIES = (RANDOM_POLICY, BEST_ARM_POLICY)
RESPONSE_POLICY = "response"
CI_LEVEL = 0.95


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


PROPENSITY_RECORDED = "recorded with the data (known assignment probabilities)"
PROPENSITY_ESTIMATED = "estimated by cross-fitted logistic regression on the features"
PROPENSITY_SHARES = "observed arm shares (correct only for a randomised test)"


def assignment_propensity(data: UpliftData, kind: PropensityKind, fold: np.ndarray) -> tuple[np.ndarray, str]:
    """Assignment probabilities for IPW/AIPW, and where they came from.

    Recorded probabilities always win. Without them, ``logistic`` cross-fits a multinomial
    logistic model on the evaluation folds, and ``constant`` uses the observed arm shares,
    which is only right when the arm was randomised with the same odds for everyone.
    """
    n_arms = len(data.arm_names)
    if data.propensity is not None:
        return data.propensity, PROPENSITY_RECORDED
    if kind == "logistic":
        estimated = np.zeros((data.n, n_arms))
        for train, test in splits(fold):
            model = Propensity("logistic").fit(data.X[train], data.arm[train], n_arms)
            estimated[test] = model.predict(data.X[test])
        return estimated, PROPENSITY_ESTIMATED
    return np.broadcast_to(data.arm_shares(), (data.n, n_arms)).copy(), PROPENSITY_SHARES


def pehe(true_cate: np.ndarray, cate: np.ndarray, treatment_names: Sequence[str]) -> dict[str, float]:
    """Root mean squared error of the effect estimates, overall and per treatment."""
    err = (cate - true_cate) ** 2
    out = {"all": float(np.sqrt(err.mean()))}
    out.update({name: float(np.sqrt(err[:, k].mean())) for k, name in enumerate(treatment_names)})
    return out


def ranking_metrics(
    data: UpliftData, cate: np.ndarray, *, n_boot: int, seed: int, level: float = CI_LEVEL, curve_points: int = 51
) -> dict[str, Any]:
    """Per treatment arm: Qini coefficient and AUUC with bootstrap intervals, plus a Qini curve.

    The normalised scores are only defined for a 0/1 outcome; for any other outcome they are
    ``None`` and the Qini curve is the ranking summary.
    """
    out: dict[str, Any] = {}
    for k, arm_name in enumerate(data.arm_names[1:], start=1):
        rows = np.flatnonzero((data.arm == 0) | (data.arm == k))
        y = data.y[rows]
        treated = (data.arm[rows] == k).astype(np.float64)
        score = cate[rows, k - 1]
        curve = qini_curve(y, score, treated).resample(curve_points)
        entry: dict[str, Any] = {"qini": None, "auuc": None}
        if data.binary_outcome:
            intervals = RankingBootstrap(y, score, treated).run(n_boot, seed + k, level)
            entry = {"qini": intervals["qini"].to_dict(), "auuc": intervals["auuc"].to_dict()}
        out[arm_name] = entry | {
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


def best_single_arm(gamma: np.ndarray, budget: float, costs: np.ndarray | None) -> tuple[int, float]:
    """The treatment worth most when sent to a random share of customers, and that share.

    Each arm gets the share the budget pays for at its cost; the arm is chosen on the AIPW
    average effects of the same rows the policies are scored on. That hindsight favours this
    reference, so a model that beats it has done so against the strongest no-model rule.
    """
    effect = (gamma[:, 1:] - gamma[:, [0]]).mean(axis=0)
    cost = np.ones(effect.size) if costs is None else np.asarray(costs, dtype=np.float64)
    share = np.minimum(1.0, budget / cost)
    k = int(np.argmax(share * effect))
    return k + 1, float(share[k])


def policy_metrics(
    data: UpliftData,
    cates: Mapping[str, np.ndarray],
    mu: np.ndarray,
    propensity: np.ndarray,
    fold: np.ndarray,
    *,
    budgets: Sequence[float],
    costs: np.ndarray | None,
    n_boot: int,
    seed: int,
    level: float = CI_LEVEL,
) -> dict[str, Any]:
    """AIPW and IPW value of each learner's budgeted policy, per 1,000 customers and per 1,000 contacts.

    Budgets are a share of customers when costs are 1 (the default); with costs the budget
    is ``share * n`` cost units. Two policies that use no model are scored next to the
    learners: ``random targeting`` (random customers, random treatment) and, with more than one
    treatment, ``best arm at random`` (random customers, all given the treatment with the
    highest average effect). Both are scored in expectation over their randomisation.

    Every policy gets paired AIPW differences against each of these references, and against
    the ``response`` learner's policy when one was evaluated (the comparison that says whether
    modelling the effect beat modelling the outcome). Differences against the two no-model
    references also get a simultaneous (max-t) interval over all learner policies at that
    budget. That is the interval a decision should use, because the recommended policy is
    the best of the learners on these same rows.
    """
    n_arms = len(data.arm_names)
    gamma_aipw = aipw_scores(data.arm, data.y, propensity, mu)
    gamma_ipw = ipw_scores(data.arm, data.y, propensity)
    mean_cost = 1.0 if costs is None else float(np.mean(costs))
    out: dict[str, Any] = {}
    for b, budget in enumerate(budgets):
        policies: dict[str, np.ndarray] = {}
        for name, cate in cates.items():
            if name != "random":
                policies[name] = one_hot_policy(_fold_allocation(cate, fold, budget, costs), n_arms)
        policies[RANDOM_POLICY] = random_policy(data.n, n_arms, min(1.0, budget / mean_cost))
        references = {"random": RANDOM_POLICY}
        if n_arms > 2:
            best, share = best_single_arm(gamma_aipw, budget, costs)
            policies[BEST_ARM_POLICY] = single_arm_policy(data.n, n_arms, best, share)
            references["best_arm"] = BEST_ARM_POLICY
        if RESPONSE_POLICY in policies:
            references["response"] = RESPONSE_POLICY
        names = list(policies)
        value = {name: np.sum(pi * gamma_aipw, axis=1) for name, pi in policies.items()}
        ones = np.ones(data.n)
        numerators: list[np.ndarray] = []
        denominators: list[np.ndarray] = []
        for name in names:
            pi = policies[name]
            contact = 1.0 - pi[:, 0]
            aipw_lift = value[name] - gamma_aipw[:, 0]
            ipw_lift = np.sum(pi * gamma_ipw, axis=1) - gamma_ipw[:, 0]
            numerators += [1000 * aipw_lift, 1000 * aipw_lift, 1000 * ipw_lift]
            denominators += [ones, contact, contact]
            for reference in references.values():
                numerators.append(1000 * (value[name] - value[reference]))
                denominators.append(ones)
        point, draws = bootstrap_ratio_draws(
            np.column_stack(numerators), np.column_stack(denominators), n_boot, seed + b
        )
        intervals = percentile_intervals(point, draws, level)
        width = 3 + len(references)
        candidates = [j for j, name in enumerate(names) if name not in REFERENCE_POLICIES]
        joint: dict[tuple[int, str], Interval] = {}
        for r, label in enumerate(references):
            if label == "response" or not candidates:
                continue
            columns = [width * j + 3 + r for j in candidates]
            band = simultaneous_intervals(point[columns], draws[:, columns], level)
            joint.update({(j, label): interval for j, interval in zip(candidates, band, strict=True)})
        per_budget: dict[str, Any] = {}
        for j, name in enumerate(names):
            pi = policies[name]
            block: list[Interval] = intervals[width * j : width * (j + 1)]
            per_budget[name] = {
                "contact_rate": float(np.mean(1.0 - pi[:, 0])),
                "arm_mix": {arm: float(pi[:, a].mean()) for a, arm in enumerate(data.arm_names)},
                "aipw_per_1000_customers": block[0].to_dict(),
                "aipw_per_1000_contacts": block[1].to_dict(),
                "ipw_per_1000_contacts": block[2].to_dict(),
            }
            for label, interval in zip(references, block[3:], strict=True):
                per_budget[name][f"aipw_gain_vs_{label}_per_1000_customers"] = interval.to_dict()
                if (j, label) in joint:
                    key = f"aipw_gain_vs_{label}_simultaneous_per_1000_customers"
                    per_budget[name][key] = joint[(j, label)].to_dict()
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
    level: float = CI_LEVEL,
    propensity: PropensityKind = "constant",
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Cross-fit every learner and collect the metrics into a JSON-ready dict.

    ``propensity`` only matters for policy value, and only when the data carry no recorded
    assignment probabilities; see :func:`assignment_propensity`.
    """
    say = progress or (lambda _msg: None)
    fold = stratified_folds(experiment_strata(data.arm, data.y), n_folds, seed)
    results: dict[str, Any] = {
        "dataset": describe(data),
        "settings": {
            "folds": n_folds,
            "train_rows_per_fold": [int(np.sum(fold != k)) for k in range(n_folds)],
            "n_boot": n_boot,
            "ci_level": level,
            "seed": seed,
            "base": base.to_dict(),
        },
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
            "fit_seconds_median": float(np.median(fitted.fit_seconds)),
            "predict_seconds_mean": float(np.mean(fitted.predict_seconds)),
            "ranking": ranking_metrics(data, fitted.cate, n_boot=n_boot, seed=seed, level=level),
        }
        if data.true_cate is not None:
            entry["pehe"] = pehe(data.true_cate, fitted.cate, data.arm_names[1:])
        results["learners"][name] = entry
    if budgets:
        say("cross-fitting AIPW outcome models")
        mu = cross_fit_outcomes(data, base, fold)
        e, source = assignment_propensity(data, propensity, fold)
        results["settings"]["budgets"] = list(budgets)
        results["settings"]["costs"] = None if costs is None else np.asarray(costs).tolist()
        results["settings"]["propensity_source"] = source
        results["settings"]["candidate_policies"] = sum(1 for name in cates if name != "random")
        say("scoring budgeted policies")
        results["policy"] = policy_metrics(
            data, cates, mu, e, fold, budgets=budgets, costs=costs, n_boot=n_boot, seed=seed, level=level
        )
    return results
