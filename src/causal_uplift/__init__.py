"""Uplift modelling for budgeted campaign targeting.

Typical library use::

    from causal_uplift import DRLearner, allocate, make_synthetic

    data = make_synthetic("step", 20_000, seed=0)
    cate = DRLearner().fit(data.X, data.arm, data.y).predict(data.X)
    plan = allocate(cate, budget=0.2 * data.n)  # 0 = no contact, k = treatment k
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("causal-uplift-targeting")
except PackageNotFoundError:  # running from a source tree without installation
    __version__ = "0.0.0"

from causal_uplift.base import BaseLearner, Propensity
from causal_uplift.data import FeatureSpec, TableSchema, UpliftData, frame_to_data, load_criteo, load_hillstrom
from causal_uplift.evaluation import evaluate
from causal_uplift.forest import UpliftForest
from causal_uplift.learners import CateLearner, DRLearner, RLearner, SLearner, TLearner, XLearner
from causal_uplift.metrics import RankingBootstrap, auuc, qini_coefficient, qini_curve, uplift_curve
from causal_uplift.ope import aipw_scores, ipw_scores, policy_value
from causal_uplift.policy import Allocation, allocate
from causal_uplift.registry import LearnerConfig, make_learner
from causal_uplift.synthetic import make_synthetic

__all__ = [
    "Allocation",
    "BaseLearner",
    "CateLearner",
    "DRLearner",
    "FeatureSpec",
    "LearnerConfig",
    "Propensity",
    "RLearner",
    "RankingBootstrap",
    "SLearner",
    "TLearner",
    "TableSchema",
    "UpliftData",
    "UpliftForest",
    "XLearner",
    "__version__",
    "aipw_scores",
    "allocate",
    "auuc",
    "evaluate",
    "frame_to_data",
    "ipw_scores",
    "load_criteo",
    "load_hillstrom",
    "make_learner",
    "make_synthetic",
    "policy_value",
    "qini_coefficient",
    "qini_curve",
    "uplift_curve",
]
