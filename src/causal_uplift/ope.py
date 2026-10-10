"""Off-policy value of a targeting policy from randomised (or known-propensity) data.

A policy is an (n, K + 1) matrix of assignment probabilities, so deterministic rules
(one-hot rows) and random targeting (fractional rows) are scored the same way. Per-row
scores Gamma[i, a] estimate the outcome customer i would have had under arm a:

* IPW:  1{A_i = a} Y_i / e_a(X_i)
* AIPW: mu_a(X_i) + 1{A_i = a} (Y_i - mu_a(X_i)) / e_a(X_i)

The value of a policy is mean_i sum_a pi[i, a] Gamma[i, a]. AIPW stays unbiased when mu is
wrong as long as the propensities are right, which they are by design in an experiment.
"""

from __future__ import annotations

import numpy as np

from causal_uplift.metrics import Interval


def ipw_scores(arm: np.ndarray, y: np.ndarray, propensity: np.ndarray) -> np.ndarray:
    hit = arm[:, None] == np.arange(propensity.shape[1])[None, :]
    return hit * (y[:, None] / propensity)


def aipw_scores(arm: np.ndarray, y: np.ndarray, propensity: np.ndarray, mu: np.ndarray) -> np.ndarray:
    hit = arm[:, None] == np.arange(propensity.shape[1])[None, :]
    return mu + hit * ((y[:, None] - mu) / propensity)


def one_hot_policy(assign: np.ndarray, n_arms: int) -> np.ndarray:
    policy = np.zeros((assign.size, n_arms))
    policy[np.arange(assign.size), assign] = 1.0
    return policy


def random_policy(n: int, n_arms: int, contact_rate: float) -> np.ndarray:
    """Contact a random ``contact_rate`` share of customers with a uniformly random treatment."""
    policy = np.full((n, n_arms), contact_rate / (n_arms - 1))
    policy[:, 0] = 1.0 - contact_rate
    return policy


def single_arm_policy(n: int, n_arms: int, arm: int, contact_rate: float) -> np.ndarray:
    """Contact a random ``contact_rate`` share of customers, always with treatment ``arm``."""
    policy = np.zeros((n, n_arms))
    policy[:, 0] = 1.0 - contact_rate
    policy[:, arm] = contact_rate
    return policy


def policy_value(policy: np.ndarray, gamma: np.ndarray) -> np.ndarray:
    """Per-row contribution to the policy value; its mean is the value estimate."""
    return np.sum(policy * gamma, axis=1)


def bootstrap_ratio_draws(
    numerator: np.ndarray, denominator: np.ndarray, n_boot: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Point estimates and bootstrap replicates of sum(num[:, j]) / sum(den[:, j]).

    All columns share each replicate's resampled rows, so differences between policies are
    resampled jointly (paired) rather than as independent estimates. Returns ``point`` (m,)
    and ``draws`` (n_boot, m).
    """
    if numerator.shape != denominator.shape:
        raise ValueError("numerator and denominator must have the same shape")
    n = numerator.shape[0]
    point = numerator.sum(axis=0) / denominator.sum(axis=0)
    rng = np.random.default_rng(seed)
    draws = np.empty((n_boot, numerator.shape[1]))
    for b in range(n_boot):
        counts = np.bincount(rng.integers(0, n, n), minlength=n).astype(np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            draws[b] = (counts @ numerator) / (counts @ denominator)
    return point, draws


def percentile_intervals(point: np.ndarray, draws: np.ndarray, level: float = 0.95) -> list[Interval]:
    """One percentile interval per column; each covers its own column at ``level``."""
    tail = (1.0 - level) / 2.0 * 100.0
    lo, hi = np.nanpercentile(draws, [tail, 100.0 - tail], axis=0)
    return [Interval(float(p), float(a), float(b)) for p, a, b in zip(point, lo, hi, strict=True)]


def simultaneous_intervals(point: np.ndarray, draws: np.ndarray, level: float = 0.95) -> list[Interval]:
    """Max-t bootstrap band: intervals that cover every column at once with probability ``level``.

    Each column is standardised by its bootstrap standard deviation; the critical value is the
    ``level`` quantile of the largest absolute standardised deviation across columns in each
    replicate. Because the band covers all columns together, it also covers whichever column
    was picked after looking at the point estimates. Strongly correlated columns (policies that
    contact mostly the same customers) cost little width; independent ones cost more.
    """
    sd = np.nanstd(draws, axis=0, ddof=1)
    spread = sd > 0
    if not spread.any():
        return [Interval(float(p), float(p), float(p)) for p in point]
    t = np.abs(draws[:, spread] - point[spread]) / sd[spread]
    critical = float(np.nanquantile(np.nanmax(t, axis=1), level))
    return [
        Interval(float(p), float(p - critical * w), float(p + critical * w)) for p, w in zip(point, sd, strict=True)
    ]
