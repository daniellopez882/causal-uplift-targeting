"""Qini and uplift curves, AUUC and the Qini coefficient, with optional row weights.

Conventions follow scikit-uplift so results are comparable: rows are ranked by descending
score, tied scores are evaluated together at the end of their group, both curves start at
(0, 0), and the normalised scores divide the area above the random line by that of the
perfect ranking. Row weights make bootstrap replicates cheap: a replicate is the same sorted
data with multinomial counts as weights.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Curve:
    n_targeted: np.ndarray
    value: np.ndarray

    def area(self) -> float:
        return float(np.trapezoid(self.value, self.n_targeted))

    def resample(self, n_points: int = 101) -> Curve:
        """Values on an even grid of targeted fractions, for plotting and JSON output."""
        grid = np.linspace(0.0, self.n_targeted[-1], n_points)
        return Curve(grid, np.interp(grid, self.n_targeted, self.value))


@dataclass(frozen=True)
class _Ranked:
    """Rows sorted by descending score, with the index of the last row of each tied group."""

    order: np.ndarray
    group_end: np.ndarray

    @classmethod
    def from_score(cls, score: np.ndarray) -> _Ranked:
        order = np.argsort(score, kind="mergesort")[::-1]
        ranked = score[order]
        return cls(order, np.r_[np.flatnonzero(np.diff(ranked)), ranked.size - 1])


def _cumulants(
    ranked: _Ranked, y: np.ndarray, treated: np.ndarray, weight: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    w = weight[ranked.order]
    t = treated[ranked.order]
    wy = w * y[ranked.order]
    ends = ranked.group_end
    n_all = np.cumsum(w)[ends]
    n_t = np.cumsum(w * t)[ends]
    y_t = np.cumsum(wy * t)[ends]
    y_c = np.cumsum(wy * (1.0 - t))[ends]
    return n_all, n_t, y_t, y_c


def _safe_div(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    return np.divide(num, den, out=np.zeros_like(num), where=den != 0)


def _start_at_origin(x: np.ndarray, v: np.ndarray) -> Curve:
    if x.size and x[0] == 0 and v[0] == 0:
        return Curve(x, v)
    return Curve(np.r_[0.0, x], np.r_[0.0, v])


def _qini(ranked: _Ranked, y: np.ndarray, treated: np.ndarray, weight: np.ndarray) -> Curve:
    n_all, n_t, y_t, y_c = _cumulants(ranked, y, treated, weight)
    return _start_at_origin(n_all, y_t - y_c * _safe_div(n_t, n_all - n_t))


def _uplift(ranked: _Ranked, y: np.ndarray, treated: np.ndarray, weight: np.ndarray) -> Curve:
    n_all, n_t, y_t, y_c = _cumulants(ranked, y, treated, weight)
    return _start_at_origin(n_all, (_safe_div(y_t, n_t) - _safe_div(y_c, n_all - n_t)) * n_all)


def _normalised(actual: Curve, perfect: Curve) -> float:
    baseline = perfect.n_targeted[-1] * perfect.value[-1] / 2.0
    return (actual.area() - baseline) / (perfect.area() - baseline)


def _prepare(
    y: np.ndarray, treated: np.ndarray, weight: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=np.float64)
    t = np.asarray(treated, dtype=np.float64)
    if not np.isin(t, (0.0, 1.0)).all():
        raise ValueError("treated must be 0/1")
    w = np.ones_like(y) if weight is None else np.asarray(weight, dtype=np.float64)
    if not (y.shape == t.shape == w.shape):
        raise ValueError("y, treated and weight must have the same length")
    return y, t, w


def _perfect_qini_score(y: np.ndarray, t: np.ndarray) -> np.ndarray:
    # Treated responders first, control responders last (negative effects allowed).
    return y * t - y * (1.0 - t)


def _perfect_uplift_score(y: np.ndarray, t: np.ndarray, w: np.ndarray) -> np.ndarray:
    control_responders = np.sum(w * ((y == 1) & (t == 0)))
    treated_non_responders = np.sum(w * ((y == 0) & (t == 1)))
    summand = y if control_responders > treated_non_responders else t
    return 2.0 * (y == t) + summand


def qini_curve(y: np.ndarray, score: np.ndarray, treated: np.ndarray, weight: np.ndarray | None = None) -> Curve:
    """Incremental responders among the top-k: Y_T(k) - Y_C(k) * N_T(k) / N_C(k)."""
    y, t, w = _prepare(y, treated, weight)
    return _qini(_Ranked.from_score(np.asarray(score, dtype=np.float64)), y, t, w)


def uplift_curve(y: np.ndarray, score: np.ndarray, treated: np.ndarray, weight: np.ndarray | None = None) -> Curve:
    """Difference in response rates within the top-k, times k."""
    y, t, w = _prepare(y, treated, weight)
    return _uplift(_Ranked.from_score(np.asarray(score, dtype=np.float64)), y, t, w)


def perfect_qini_curve(y: np.ndarray, treated: np.ndarray, weight: np.ndarray | None = None) -> Curve:
    y, t, w = _prepare(y, treated, weight)
    return _qini(_Ranked.from_score(_perfect_qini_score(y, t)), y, t, w)


def perfect_uplift_curve(y: np.ndarray, treated: np.ndarray, weight: np.ndarray | None = None) -> Curve:
    y, t, w = _prepare(y, treated, weight)
    return _uplift(_Ranked.from_score(_perfect_uplift_score(y, t, w)), y, t, w)


def qini_coefficient(y: np.ndarray, score: np.ndarray, treated: np.ndarray, weight: np.ndarray | None = None) -> float:
    """Area between the Qini curve and the random line, divided by the same for the perfect ranking."""
    return _normalised(qini_curve(y, score, treated, weight), perfect_qini_curve(y, treated, weight))


def auuc(y: np.ndarray, score: np.ndarray, treated: np.ndarray, weight: np.ndarray | None = None) -> float:
    """Normalised area under the uplift curve (scikit-uplift's ``uplift_auc_score``)."""
    return _normalised(uplift_curve(y, score, treated, weight), perfect_uplift_curve(y, treated, weight))


@dataclass(frozen=True)
class Interval:
    value: float
    lo: float
    hi: float

    def to_dict(self) -> dict[str, float]:
        return {"value": self.value, "lo": self.lo, "hi": self.hi}


class RankingBootstrap:
    """Qini coefficient and AUUC with percentile bootstrap intervals.

    The three orderings (model, perfect Qini, perfect uplift) are sorted once; each replicate
    only recomputes weighted cumulative sums. The perfect-uplift ordering depends on the
    replicate's weighted counts, so both of its candidate orderings are kept.
    """

    def __init__(self, y: np.ndarray, score: np.ndarray, treated: np.ndarray) -> None:
        self.y, self.t, _ = _prepare(y, treated, None)
        self.model = _Ranked.from_score(np.asarray(score, dtype=np.float64))
        self.perfect_qini = _Ranked.from_score(_perfect_qini_score(self.y, self.t))
        self.perfect_uplift_y = _Ranked.from_score(2.0 * (self.y == self.t) + self.y)
        self.perfect_uplift_t = _Ranked.from_score(2.0 * (self.y == self.t) + self.t)

    def _scores(self, w: np.ndarray) -> tuple[float, float]:
        y, t = self.y, self.t
        control_responders = np.sum(w * ((y == 1) & (t == 0)))
        treated_non_responders = np.sum(w * ((y == 0) & (t == 1)))
        uplift_ref = self.perfect_uplift_y if control_responders > treated_non_responders else self.perfect_uplift_t
        qini = _normalised(_qini(self.model, y, t, w), _qini(self.perfect_qini, y, t, w))
        upl = _normalised(_uplift(self.model, y, t, w), _uplift(uplift_ref, y, t, w))
        return qini, upl

    def run(self, n_boot: int, seed: int, level: float = 0.95) -> dict[str, Interval]:
        n = self.y.size
        point_qini, point_auuc = self._scores(np.ones(n))
        rng = np.random.default_rng(seed)
        draws = np.empty((n_boot, 2))
        for b in range(n_boot):
            draws[b] = self._scores(np.bincount(rng.integers(0, n, n), minlength=n).astype(np.float64))
        tail = (1.0 - level) / 2.0 * 100.0
        lo, hi = np.nanpercentile(draws, [tail, 100.0 - tail], axis=0)
        return {
            "qini": Interval(point_qini, float(lo[0]), float(hi[0])),
            "auuc": Interval(point_auuc, float(lo[1]), float(hi[1])),
        }
