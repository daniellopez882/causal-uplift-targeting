from __future__ import annotations

from hypothesis import HealthCheck, settings

settings.register_profile(
    "default", deadline=None, max_examples=60, suppress_health_check=[HealthCheck.too_slow], derandomize=True
)
settings.load_profile("default")
