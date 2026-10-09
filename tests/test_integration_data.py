"""Checks against the real downloaded files; skipped when they are not present."""

from __future__ import annotations

import numpy as np
import pytest

from causal_uplift.data import load_hillstrom
from causal_uplift.fetch import SOURCES, local_path, sha256_of

pytestmark = pytest.mark.integration

needs_hillstrom = pytest.mark.skipif(not local_path("hillstrom").exists(), reason="run `uplift fetch hillstrom`")


@needs_hillstrom
def test_hillstrom_file_matches_pinned_checksum() -> None:
    assert sha256_of(local_path("hillstrom")) == SOURCES["hillstrom"].sha256


@needs_hillstrom
def test_hillstrom_shape_and_randomisation() -> None:
    data, spec = load_hillstrom("visit")
    assert data.n == 64_000
    assert len(spec.names) == 11
    counts = dict(zip(data.arm_names, np.bincount(data.arm).tolist(), strict=True))
    assert counts == {"No E-Mail": 21_306, "Mens E-Mail": 21_307, "Womens E-Mail": 21_387}
    rates = [data.y[data.arm == a].mean() for a in range(3)]
    assert rates[0] < rates[2] < rates[1]
