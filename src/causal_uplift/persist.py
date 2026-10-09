"""Save and load fitted learners together with the feature encoding they expect.

A model directory holds ``model.json`` (readable metadata) and ``learner.pkl``. Pickle runs
code on load, so only load model directories you produced yourself.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from causal_uplift import __version__
from causal_uplift.data import FeatureSpec
from causal_uplift.learners import CateLearner

META_FILE = "model.json"
LEARNER_FILE = "learner.pkl"
FORMAT_VERSION = 1


@dataclass(frozen=True)
class SavedModel:
    learner: CateLearner
    spec: FeatureSpec
    arm_names: tuple[str, ...]
    meta: dict[str, Any]


def save_model(
    directory: Path, learner: CateLearner, spec: FeatureSpec, arm_names: tuple[str, ...], meta: dict[str, Any]
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": FORMAT_VERSION,
        "package_version": __version__,
        "learner": learner.name,
        "arm_names": list(arm_names),
        "feature_spec": spec.to_dict(),
        **meta,
    }
    (directory / META_FILE).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    with (directory / LEARNER_FILE).open("wb") as handle:
        pickle.dump(learner, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return directory


def load_model(directory: Path) -> SavedModel:
    meta = json.loads((directory / META_FILE).read_text(encoding="utf-8"))
    if meta.get("format") != FORMAT_VERSION:
        raise ValueError(f"unsupported model format {meta.get('format')!r} in {directory}")
    with (directory / LEARNER_FILE).open("rb") as handle:
        learner = pickle.load(handle)
    if not isinstance(learner, CateLearner):
        raise TypeError(f"{directory / LEARNER_FILE} does not contain a fitted learner")
    return SavedModel(
        learner=learner,
        spec=FeatureSpec.from_dict(meta["feature_spec"]),
        arm_names=tuple(meta["arm_names"]),
        meta=meta,
    )
