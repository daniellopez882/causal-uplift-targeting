"""Synthetic three-arm experiments with known conditional effects, for PEHE and sanity tests.

Every scenario has a control arm and two treatments. Outcomes are binary; the true CATE of
arm ``a`` is the difference of the (clipped) purchase probabilities ``p_a(x) - p_0(x)``.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from causal_uplift.data import PROPENSITY_PREFIX, TRUE_CATE_PREFIX, UpliftData

SCENARIOS = ("linear", "step", "sleeping_dogs", "confounded")
ARM_NAMES = ("control", "arm1", "arm2")
MIN_FEATURES = 4


def potential_means(scenario: str, X: np.ndarray) -> np.ndarray:
    """Purchase probability under each arm, shape (n, 3)."""
    x0, x1, x2, x3 = (X[:, j] for j in range(MIN_FEATURES))
    if scenario == "linear":
        base = 0.25 + 0.06 * x0 - 0.04 * x1
        tau1 = 0.06 + 0.04 * x2
        tau2 = 0.03 - 0.05 * x3
    elif scenario == "step":
        base = 0.15 + 0.10 * (x0 > 0)
        tau1 = 0.12 * (x1 > 0.5) - 0.06 * (x2 < -1.0)
        tau2 = 0.08 * ((x1 <= 0.5) & (x3 > 0))
    elif scenario == "sleeping_dogs":
        # Customers likely to buy anyway are the ones the e-mail puts off.
        likely = x0 > 0
        base = np.where(likely, 0.55, 0.08) + 0.02 * x1
        tau1 = np.where(likely, -0.10, 0.12)
        tau2 = np.where(likely, -0.05, 0.06)
    elif scenario == "confounded":
        base = 0.25 + 0.06 * x0
        tau1 = 0.08 * (x0 > 0) + 0.03 * x1
        tau2 = 0.04 - 0.03 * x1
    else:
        raise ValueError(f"unknown scenario {scenario!r}; choose from {SCENARIOS}")
    return np.clip(np.column_stack([base, base + tau1, base + tau2]), 0.02, 0.98)


def assignment_probabilities(scenario: str, X: np.ndarray) -> np.ndarray:
    n = X.shape[0]
    if scenario != "confounded":
        return np.full((n, 3), 1.0 / 3.0)
    logits = np.column_stack([np.zeros(n), 0.8 * X[:, 0], -0.8 * X[:, 0]])
    weights = np.exp(logits - logits.max(axis=1, keepdims=True))
    return weights / weights.sum(axis=1, keepdims=True)


def make_synthetic(scenario: str, n: int, *, n_features: int = 10, seed: int = 0) -> UpliftData:
    if n_features < MIN_FEATURES:
        raise ValueError(f"n_features must be at least {MIN_FEATURES}")
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, n_features))
    means = potential_means(scenario, X)
    propensity = assignment_probabilities(scenario, X)
    cumulative = propensity.cumsum(axis=1)
    arm = (rng.random(n)[:, None] > cumulative).sum(axis=1).astype(np.int64)
    y = (rng.random(n) < means[np.arange(n), arm]).astype(np.float64)
    return UpliftData(
        X=X,
        arm=arm,
        y=y,
        feature_names=tuple(f"x{j}" for j in range(n_features)),
        arm_names=ARM_NAMES,
        outcome="purchase",
        true_cate=means[:, 1:] - means[:, :1],
        propensity=propensity,
    )


def to_frame(data: UpliftData) -> pl.DataFrame:
    """Table layout readable by ``frame_to_data`` with treatment ``arm`` and outcome ``purchase``.

    The assignment probabilities travel with the rows, so a file written here is evaluated with
    the same propensities as the in-memory experiment (this matters for ``confounded``).
    """
    columns: dict[str, object] = {name: data.X[:, j] for j, name in enumerate(data.feature_names)}
    columns["arm"] = np.asarray(data.arm_names)[data.arm]
    columns[data.outcome] = data.y
    if data.true_cate is not None:
        for k, name in enumerate(data.arm_names[1:]):
            columns[f"{TRUE_CATE_PREFIX}{name}"] = data.true_cate[:, k]
    if data.propensity is not None:
        for a, name in enumerate(data.arm_names):
            columns[f"{PROPENSITY_PREFIX}{name}"] = data.propensity[:, a]
    return pl.DataFrame(columns)
