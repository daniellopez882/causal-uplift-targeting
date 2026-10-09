"""Numba kernels for growing and applying divergence-split uplift trees on binned features.

A node's divergence is the mean over treatment arms of D(P_arm || P_control); a split is
scored by the size-weighted divergence of the children minus that of the parent
(Rzepakowski and Jaroszewicz, 2012). Features are pre-binned to uint8 codes so every
candidate threshold of a feature is evaluated from one histogram pass.
"""

from __future__ import annotations

import math

import numba as nb
import numpy as np

KL = 0
ED = 1
_EPS = 1e-6


@nb.njit(cache=True, nogil=True)
def _pair_divergence(p_treat: float, p_ctrl: float, criterion: int) -> float:
    if criterion == KL:
        p = min(max(p_treat, _EPS), 1.0 - _EPS)
        q = min(max(p_ctrl, _EPS), 1.0 - _EPS)
        return p * math.log(p / q) + (1.0 - p) * math.log((1.0 - p) / (1.0 - q))
    diff = p_treat - p_ctrl
    return diff * diff


@nb.njit(cache=True, nogil=True)
def node_divergence(count: np.ndarray, total: np.ndarray, criterion: int) -> float:
    n_arms = count.shape[0]
    ctrl = total[0] / count[0]
    acc = 0.0
    for a in range(1, n_arms):
        acc += _pair_divergence(total[a] / count[a], ctrl, criterion)
    return acc / (n_arms - 1)


@nb.njit(cache=True, nogil=True)
def best_split(
    Xb: np.ndarray,
    arm: np.ndarray,
    y: np.ndarray,
    work: np.ndarray,
    start: int,
    end: int,
    candidates: np.ndarray,
    n_bins: np.ndarray,
    n_arms: int,
    min_leaf: int,
    criterion: int,
    min_gain: float,
) -> tuple[int, int, float]:
    """Best (feature, bin threshold, gain) over ``candidates`` for rows ``work[start:end]``.

    Rows with bin <= threshold go left. Every arm must keep ``min_leaf`` rows on each side.
    Returns feature -1 when no split beats ``min_gain``.
    """
    tot_c = np.zeros(n_arms)
    tot_s = np.zeros(n_arms)
    for i in range(start, end):
        r = work[i]
        tot_c[arm[r]] += 1.0
        tot_s[arm[r]] += y[r]
    for a in range(n_arms):
        if tot_c[a] < 2 * min_leaf:
            return -1, -1, 0.0
    n_total = end - start
    parent = node_divergence(tot_c, tot_s, criterion)
    max_bins = n_bins.max()
    hist_c = np.zeros((max_bins, n_arms))
    hist_s = np.zeros((max_bins, n_arms))
    left_c = np.zeros(n_arms)
    left_s = np.zeros(n_arms)
    right_c = np.zeros(n_arms)
    right_s = np.zeros(n_arms)
    best_feature = -1
    best_threshold = -1
    best_gain = min_gain
    for f in candidates:
        nb_f = n_bins[f]
        if nb_f < 2:
            continue
        hist_c[:nb_f, :] = 0.0
        hist_s[:nb_f, :] = 0.0
        for i in range(start, end):
            r = work[i]
            b = Xb[r, f]
            hist_c[b, arm[r]] += 1.0
            hist_s[b, arm[r]] += y[r]
        left_c[:] = 0.0
        left_s[:] = 0.0
        for t in range(nb_f - 1):
            feasible = True
            n_left = 0.0
            for a in range(n_arms):
                left_c[a] += hist_c[t, a]
                left_s[a] += hist_s[t, a]
                right_c[a] = tot_c[a] - left_c[a]
                right_s[a] = tot_s[a] - left_s[a]
                n_left += left_c[a]
                if left_c[a] < min_leaf or right_c[a] < min_leaf:
                    feasible = False
            if not feasible:
                continue
            share = n_left / n_total
            gain = (
                share * node_divergence(left_c, left_s, criterion)
                + (1.0 - share) * node_divergence(right_c, right_s, criterion)
                - parent
            )
            if gain > best_gain:
                best_gain = gain
                best_feature = f
                best_threshold = t
    return best_feature, best_threshold, best_gain


@nb.njit(cache=True, nogil=True)
def grow_tree(
    Xb: np.ndarray,
    arm: np.ndarray,
    y: np.ndarray,
    rows: np.ndarray,
    n_bins: np.ndarray,
    n_arms: int,
    max_depth: int,
    min_leaf: int,
    max_features: int,
    criterion: int,
    min_gain: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Grow one tree depth-first on ``rows``; returns (feature, threshold, left, right) per node.

    Leaves have feature -1. Children always get larger ids than their parent.
    """
    np.random.seed(seed)
    n_features = Xb.shape[1]
    max_nodes = 2 ** (max_depth + 1) - 1
    feature = np.full(max_nodes, -1, dtype=np.int32)
    threshold = np.zeros(max_nodes, dtype=np.int32)
    left = np.full(max_nodes, -1, dtype=np.int32)
    right = np.full(max_nodes, -1, dtype=np.int32)
    work = rows.copy()
    order = np.arange(n_features)
    stack = np.empty((max_nodes, 4), dtype=np.int64)
    stack[0, 0] = 0
    stack[0, 1] = 0
    stack[0, 2] = work.shape[0]
    stack[0, 3] = 0
    top = 1
    n_nodes = 1
    while top > 0:
        top -= 1
        node = stack[top, 0]
        start = stack[top, 1]
        end = stack[top, 2]
        depth = stack[top, 3]
        if depth >= max_depth:
            continue
        for j in range(max_features):
            k = j + np.random.randint(n_features - j)
            tmp = order[j]
            order[j] = order[k]
            order[k] = tmp
        f, t, _ = best_split(
            Xb, arm, y, work, start, end, order[:max_features], n_bins, n_arms, min_leaf, criterion, min_gain
        )
        if f < 0:
            continue
        i = start
        j = end - 1
        while i <= j:
            if Xb[work[i], f] <= t:
                i += 1
            else:
                tmp = work[i]
                work[i] = work[j]
                work[j] = tmp
                j -= 1
        feature[node] = f
        threshold[node] = t
        left[node] = n_nodes
        right[node] = n_nodes + 1
        stack[top, 0] = n_nodes + 1
        stack[top, 1] = i
        stack[top, 2] = end
        stack[top, 3] = depth + 1
        stack[top + 1, 0] = n_nodes
        stack[top + 1, 1] = start
        stack[top + 1, 2] = i
        stack[top + 1, 3] = depth + 1
        top += 2
        n_nodes += 2
    return feature[:n_nodes], threshold[:n_nodes], left[:n_nodes], right[:n_nodes]


@nb.njit(cache=True, nogil=True)
def path_sums(
    Xb: np.ndarray,
    arm: np.ndarray,
    y: np.ndarray,
    rows: np.ndarray,
    feature: np.ndarray,
    threshold: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    n_arms: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Per node and arm, the row count and outcome sum of ``rows`` routed through that node."""
    n_nodes = feature.shape[0]
    count = np.zeros((n_nodes, n_arms))
    total = np.zeros((n_nodes, n_arms))
    for r in rows:
        a = arm[r]
        node = 0
        while True:
            count[node, a] += 1.0
            total[node, a] += y[r]
            f = feature[node]
            if f < 0:
                break
            node = left[node] if Xb[r, f] <= threshold[node] else right[node]
    return count, total


@nb.njit(cache=True, nogil=True)
def apply_tree(
    Xb: np.ndarray, feature: np.ndarray, threshold: np.ndarray, left: np.ndarray, right: np.ndarray
) -> np.ndarray:
    out = np.empty(Xb.shape[0], dtype=np.int64)
    for r in range(Xb.shape[0]):
        node = 0
        while feature[node] >= 0:
            node = left[node] if Xb[r, feature[node]] <= threshold[node] else right[node]
        out[r] = node
    return out
