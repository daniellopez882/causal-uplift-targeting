from __future__ import annotations

import pytest
from hypothesis import HealthCheck, settings

from causal_uplift.base import BaseLearner

settings.register_profile(
    "default", deadline=None, max_examples=60, suppress_health_check=[HealthCheck.too_slow], derandomize=True
)
settings.load_profile("default")


@pytest.fixture
def small_base() -> BaseLearner:
    """A LightGBM configuration small enough for unit tests."""
    return BaseLearner(params={"n_estimators": 120, "learning_rate": 0.05}, n_jobs=2)
