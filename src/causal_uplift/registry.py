"""Name-based construction of learners for the CLI and the benchmark."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from causal_uplift.base import BaseLearner, PropensityKind
from causal_uplift.forest import Criterion, UpliftForest
from causal_uplift.learners import (
    CateLearner,
    DRLearner,
    RandomScores,
    ResponseModel,
    RLearner,
    SLearner,
    TLearner,
    XLearner,
)

LEARNER_NAMES = ("s", "t", "x", "r", "dr", "forest-kl", "forest-ed", "response", "random")
BASELINES = ("response", "random")


@dataclass(frozen=True)
class LearnerConfig:
    base: BaseLearner = field(default_factory=BaseLearner)
    propensity: PropensityKind = "constant"
    seed: int = 0
    forest: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"base": self.base.to_dict(), "propensity": self.propensity, "seed": self.seed, "forest": self.forest}


def make_learner(name: str, config: LearnerConfig | None = None) -> CateLearner:
    cfg = config or LearnerConfig()
    base = cfg.base.with_seed(cfg.seed)
    match name:
        case "s":
            return SLearner(base)
        case "t":
            return TLearner(base)
        case "x":
            return XLearner(base, cfg.propensity)
        case "r":
            return RLearner(base, cfg.propensity, seed=cfg.seed)
        case "dr":
            return DRLearner(base, cfg.propensity, seed=cfg.seed)
        case "forest-kl" | "forest-ed":
            criterion: Criterion = "kl" if name == "forest-kl" else "ed"
            options = {"n_jobs": base.n_jobs, **cfg.forest}
            return UpliftForest(criterion, seed=cfg.seed, **options)
        case "response":
            return ResponseModel(base)
        case "random":
            return RandomScores(cfg.seed)
    raise ValueError(f"unknown learner {name!r}; choose from {LEARNER_NAMES}")
