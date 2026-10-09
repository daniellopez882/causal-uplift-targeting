"""Choose who gets which treatment when contacts cost money and the budget is fixed.

maximise   sum_i benefit[i, a_i]
subject to sum_i cost[a_i] <= budget,  a_i in {no contact, 1..K}

This is a multiple-choice knapsack. It is solved by Lagrangian relaxation: for a price
lambda per unit of cost each customer independently takes the arm with the largest
benefit - lambda * cost (or nothing if that is not positive); lambda is bisected to the
smallest value whose spend fits the budget, and leftover budget is filled greedily. With
equal costs this is exactly "top-k by best predicted uplift". The dual function at the
final lambda is returned as an upper bound on the best achievable predicted benefit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Allocation:
    assign: np.ndarray  # 0 = no contact, k = treatment k
    spend: float
    benefit: float
    upper_bound: float
    shadow_price: float


def _respond(net: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    best = np.argmax(net, axis=1)
    best_net = net[np.arange(net.shape[0]), best]
    return best, best_net


def allocate(
    benefit: np.ndarray,
    budget: float,
    costs: np.ndarray | None = None,
    *,
    tol: float = 1e-12,
    max_iter: int = 200,
) -> Allocation:
    """Optimise the assignment for ``benefit`` (n, K), e.g. predicted uplift times value."""
    benefit = np.asarray(benefit, dtype=np.float64)
    k = benefit.shape[1]
    cost = np.ones(k) if costs is None else np.asarray(costs, dtype=np.float64)
    if cost.shape != (k,) or (cost <= 0).any():
        raise ValueError("costs must be positive, one per treatment")
    if budget < 0:
        raise ValueError("budget must be non-negative")

    def spend_at(lam: float) -> tuple[np.ndarray, np.ndarray, float]:
        best, best_net = _respond(benefit - lam * cost)
        chosen = best_net > 0
        return best, chosen, float(cost[best][chosen].sum())

    lam = 0.0
    best, chosen, spend = spend_at(0.0)
    if spend > budget:
        # Just above the largest benefit/cost ratio every net value is negative, so spend is 0;
        # at exactly the ratio, rounding in benefit - ratio * cost can leave a tiny positive net.
        top = float(np.max(benefit / cost))
        lo, hi = 0.0, top * (1.0 + 1e-9) + 1e-300
        for _ in range(max_iter):
            if hi - lo <= tol * max(1.0, hi):
                break
            mid = 0.5 * (lo + hi)
            if spend_at(mid)[2] > budget:
                lo = mid
            else:
                hi = mid
        lam = hi
        best, chosen, spend = spend_at(lam)

    assign = np.where(chosen, best + 1, 0)
    # Greedy fill of the budget left by the discrete jump at lambda, closest-to-chosen first.
    leftover = np.flatnonzero(~chosen)
    order = leftover[np.argsort(-(benefit - lam * cost)[leftover, best[leftover]], kind="stable")]
    for i in order:
        if budget - spend < cost.min():
            break
        a = best[i]
        if benefit[i, a] > 0 and spend + cost[a] <= budget + 1e-9:
            assign[i] = a + 1
            spend += float(cost[a])

    contacted = np.flatnonzero(assign > 0)
    total = float(benefit[contacted, assign[contacted] - 1].sum())
    _, dual_net = _respond(benefit - lam * cost)
    upper = float(np.clip(dual_net, 0.0, None).sum() + lam * budget)
    return Allocation(assign=assign, spend=spend, benefit=total, upper_bound=upper, shadow_price=lam)
