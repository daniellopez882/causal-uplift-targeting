"""Benchmark harness.

    uv run python bench/run.py --profile full        # Hillstrom (visit, conversion) + synthetic PEHE
    uv run python bench/run.py --profile smoke --out-dir /tmp/uplift-smoke
    uv run python bench/run.py --profile criteo      # optional, needs `uplift fetch criteo`

Each suite writes one JSON file with machine, date, git_sha, command and the metrics.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench.baselines import SkliftTwoModels
from bench.environment import provenance
from causal_uplift.base import BaseLearner, PropensityKind
from causal_uplift.data import UpliftData, load_criteo, load_hillstrom
from causal_uplift.evaluation import evaluate, pehe
from causal_uplift.fetch import local_path
from causal_uplift.forest import UpliftForest
from causal_uplift.learners import CateLearner
from causal_uplift.metrics import qini_coefficient
from causal_uplift.registry import BASELINES, LearnerConfig, make_learner
from causal_uplift.synthetic import SCENARIOS, make_synthetic

UPLIFT_LEARNERS = ("s", "t", "x", "r", "dr", "forest-kl", "forest-ed")
BUDGETS = (0.1, 0.2, 0.3)


@dataclass(frozen=True)
class Profile:
    name: str
    hillstrom_outcomes: tuple[str, ...]
    hillstrom_rows: int | None
    folds: int
    n_boot: int
    trees: int
    synth_scenarios: tuple[str, ...]
    synth_train: int
    synth_test: int
    synth_seeds: tuple[int, ...]
    criteo_rows: int


PROFILES = {
    "full": Profile(
        "full", ("visit", "conversion"), None, 5, 1000, 200, SCENARIOS, 20_000, 10_000, (100, 101, 102), 1_000_000
    ),
    "smoke": Profile("smoke", ("visit",), 6_000, 2, 50, 20, ("step", "sleeping_dogs"), 3_000, 2_000, (100,), 50_000),
}
PROFILES["criteo"] = PROFILES["full"]


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", file=sys.stderr, flush=True)


def _factories(
    base: BaseLearner, trees: int, propensity: PropensityKind = "constant"
) -> dict[str, Callable[[int], CateLearner]]:
    def registry_factory(name: str) -> Callable[[int], CateLearner]:
        return lambda k: make_learner(
            name, LearnerConfig(base=base, propensity=propensity, seed=k, forest={"n_trees": trees})
        )

    out = {name: registry_factory(name) for name in (*UPLIFT_LEARNERS, *BASELINES)}
    out["sklift-two-model"] = lambda k: SkliftTwoModels(base.with_seed(k))
    return out


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    log(f"wrote {path}")


def warm_up(base: BaseLearner, trees: int) -> None:
    """Fit every learner once on a small synthetic sample before anything is timed.

    The first fit in a process pays for numba compilation (or loading its cache) and for
    starting LightGBM's OpenMP thread pool; without this the first fold of the first learner
    absorbs those one-off costs.
    """
    data = make_synthetic("step", 2_000, seed=0)
    for factory in _factories(base, trees).values():
        factory(0).fit(data.X, data.arm, data.y, len(data.arm_names))


def run_experiment(name: str, data: UpliftData, profile: Profile, base: BaseLearner, command: str, out: Path) -> None:
    start = time.perf_counter()
    results = evaluate(
        data,
        _factories(base, profile.trees),
        base=base,
        n_folds=profile.folds,
        n_boot=profile.n_boot,
        seed=0,
        budgets=BUDGETS,
        progress=lambda m: log(f"{name}: {m}"),
    )
    payload = {
        **provenance(command),
        "suite": name,
        "profile": asdict(profile),
        "wall_seconds": round(time.perf_counter() - start, 1),
        **results,
    }
    _write(out, payload)


def hillstrom_suite(profile: Profile, base: BaseLearner, command: str, out_dir: Path, prefix: str) -> None:
    for outcome in profile.hillstrom_outcomes:
        data, _ = load_hillstrom(outcome)
        if profile.hillstrom_rows is not None:
            keep = np.sort(np.random.default_rng(0).choice(data.n, profile.hillstrom_rows, replace=False))
            data = data.subset(keep)
        run_experiment(
            f"hillstrom-{outcome}", data, profile, base, command, out_dir / f"{prefix}hillstrom_{outcome}.json"
        )


def synthetic_suite(profile: Profile, base: BaseLearner, command: str, out_dir: Path, prefix: str) -> None:
    start = time.perf_counter()
    scenarios: dict[str, Any] = {}
    for scenario in profile.synth_scenarios:
        randomised = scenario != "confounded"
        propensity: PropensityKind = "constant" if randomised else "logistic"
        factories = _factories(base, profile.trees, propensity)
        factories.pop("random")
        factories["forest-kl-adaptive"] = lambda k: UpliftForest(
            "kl", n_trees=profile.trees, honest=False, seed=k, n_jobs=base.n_jobs
        )
        runs: dict[str, list[dict[str, Any]]] = {name: [] for name in (*factories, "oracle")}
        for seed in profile.synth_seeds:
            train = make_synthetic(scenario, profile.synth_train, seed=seed)
            test = make_synthetic(scenario, profile.synth_test, seed=seed + 1000)
            assert test.true_cate is not None
            for name, factory in factories.items():
                log(f"synthetic {scenario} seed {seed}: {name}")
                learner = factory(seed)
                t0 = time.perf_counter()
                learner.fit(train.X, train.arm, train.y, len(train.arm_names))
                fit_seconds = time.perf_counter() - t0
                runs[name].append(_score_test(test, learner.predict(test.X), randomised) | {"fit_seconds": fit_seconds})
            runs["oracle"].append(_score_test(test, test.true_cate, randomised) | {"fit_seconds": 0.0})
        scenarios[scenario] = {
            "randomised": randomised,
            "propensity_model": propensity,
            "learners": {name: _aggregate(r) for name, r in runs.items()},
        }
    payload = {
        **provenance(command),
        "suite": "synthetic",
        "profile": asdict(profile),
        "base": base.to_dict(),
        "wall_seconds": round(time.perf_counter() - start, 1),
        "scenarios": scenarios,
    }
    _write(out_dir / f"{prefix}synthetic.json", payload)


def _score_test(test: UpliftData, cate: np.ndarray, randomised: bool) -> dict[str, Any]:
    assert test.true_cate is not None
    scores: dict[str, Any] = {"pehe": pehe(test.true_cate, cate, test.arm_names[1:])["all"]}
    if randomised:
        for k, arm in enumerate(test.arm_names[1:], start=1):
            rows = (test.arm == 0) | (test.arm == k)
            scores[f"qini_{arm}"] = qini_coefficient(
                test.y[rows], cate[rows, k - 1], (test.arm[rows] == k).astype(float)
            )
    return scores


def _aggregate(runs: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"n_seeds": len(runs)}
    for key in runs[0]:
        values = np.array([r[key] for r in runs], dtype=np.float64)
        out[key] = {"mean": float(values.mean()), "sd": float(values.std(ddof=1)) if values.size > 1 else 0.0}
    return out


def criteo_suite(profile: Profile, base: BaseLearner, command: str, out_dir: Path, prefix: str) -> None:
    if not local_path("criteo").exists():
        log("criteo: not run (data not fetched; run `uplift fetch criteo` first)")
        return
    data, _ = load_criteo("conversion", n_rows=profile.criteo_rows, seed=0)
    run_experiment("criteo-conversion", data, profile, base, command, out_dir / f"{prefix}criteo_conversion.json")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--profile", choices=sorted(PROFILES), default="full")
    parser.add_argument("--out-dir", default=str(Path(__file__).resolve().parent / "results"))
    parser.add_argument("--n-jobs", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--suite", choices=("all", "hillstrom", "synthetic"), default="all", help="full/smoke only")
    args = parser.parse_args(argv)
    profile = PROFILES[args.profile]
    command = "uv run python bench/run.py " + shlex.join(argv if argv is not None else sys.argv[1:])
    base = BaseLearner(n_jobs=args.n_jobs)
    log("warming up learners (not timed)")
    warm_up(base, trees=5)
    out_dir = Path(args.out_dir)
    prefix = "smoke-" if args.profile == "smoke" else ""
    start = time.perf_counter()
    if args.profile == "criteo":
        criteo_suite(profile, base, command, out_dir, prefix)
    else:
        if args.suite in ("all", "hillstrom"):
            hillstrom_suite(profile, base, command, out_dir, prefix)
        if args.suite in ("all", "synthetic"):
            synthetic_suite(profile, base, command, out_dir, prefix)
    log(f"done in {time.perf_counter() - start:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
