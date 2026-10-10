"""Rebuild ``tests/golden/results.json`` from a small cross-fitted run on synthetic data.

    uv run python tests/golden/regenerate.py
    UPDATE_GOLDEN=1 uv run pytest tests/test_cli_report.py   # then refresh report.html

Floats are rounded and timings replaced by fixed values, so the fixture is readable and does
not change between machines. The report test checks rendering, not these numbers.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from causal_uplift.base import BaseLearner
from causal_uplift.evaluation import evaluate
from causal_uplift.registry import LearnerConfig, make_learner
from causal_uplift.synthetic import make_synthetic

LEARNERS = ("t", "forest-ed", "response", "random")


def _rounded(obj: Any) -> Any:
    if isinstance(obj, float):
        return round(obj, 4)
    if isinstance(obj, dict):
        return {k: _rounded(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_rounded(v) for v in obj]
    return obj


def build() -> dict[str, Any]:
    data = make_synthetic("step", 3000, seed=0)
    base = BaseLearner(params={"n_estimators": 60}, n_jobs=1)

    def factory(name: str) -> Any:
        return lambda k: make_learner(name, LearnerConfig(base=base, seed=k, forest={"n_trees": 20}))

    results = evaluate(
        data, {name: factory(name) for name in LEARNERS}, base=base, n_folds=2, n_boot=50, seed=0, budgets=(0.1, 0.3)
    )
    for entry in results["learners"].values():
        entry["fit_seconds_per_fold"] = [0.1] * results["settings"]["folds"]
        entry.update(fit_seconds_mean=0.1, fit_seconds_median=0.1, predict_seconds_mean=0.01)
    return _rounded(results)


def main() -> None:
    path = Path(__file__).parent / "results.json"
    path.write_text(json.dumps(build(), indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
