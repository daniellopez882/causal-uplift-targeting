"""Command line: fetch, synth, fit, evaluate, policy, score."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from causal_uplift import __version__
from causal_uplift.base import BASE_KINDS, PROPENSITY_KINDS, BaseLearner
from causal_uplift.data import (
    FeatureSpec,
    TableSchema,
    UpliftData,
    frame_to_data,
    load_criteo,
    load_hillstrom,
    read_table,
    write_table,
)
from causal_uplift.evaluation import evaluate
from causal_uplift.fetch import SOURCES, ChecksumError, fetch
from causal_uplift.persist import load_model, save_model
from causal_uplift.policy import allocate
from causal_uplift.registry import LEARNER_NAMES, LearnerConfig, make_learner
from causal_uplift.report import recommend, render_report
from causal_uplift.synthetic import SCENARIOS, make_synthetic, to_frame

DEFAULT_LEARNERS = "s,t,x,r,dr,forest-kl,forest-ed,response,random"


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def load_dataset(args: argparse.Namespace) -> tuple[UpliftData, FeatureSpec]:
    """Resolve ``--data``: hillstrom, criteo, synthetic:<scenario>, or a .parquet/.csv path."""
    source: str = args.data
    if source == "hillstrom":
        return load_hillstrom(args.outcome or "visit")
    if source == "criteo":
        return load_criteo(args.outcome or "conversion", n_rows=args.rows or 1_000_000, seed=args.seed)
    if source.startswith("synthetic:"):
        scenario = source.split(":", 1)[1]
        data = make_synthetic(scenario, args.rows or 20_000, seed=args.seed)
        _, spec = frame_to_data(to_frame(data), TableSchema(treatment="arm", outcome=data.outcome, control="control"))
        return data, spec
    frame = read_table(Path(source))
    schema = TableSchema(treatment=args.treatment_col, outcome=args.outcome or "purchase", control=args.control)
    return frame_to_data(frame, schema)


def _config(args: argparse.Namespace, seed: int) -> LearnerConfig:
    forest = {"n_trees": args.trees} if args.trees else {}
    return LearnerConfig(
        base=BaseLearner(kind=args.base, n_jobs=args.n_jobs), propensity=args.propensity, seed=seed, forest=forest
    )


def _learner_names(raw: str) -> list[str]:
    names = [n.strip() for n in raw.split(",") if n.strip()]
    unknown = sorted(set(names) - set(LEARNER_NAMES))
    if unknown:
        raise SystemExit(f"unknown learners {unknown}; choose from {', '.join(LEARNER_NAMES)}")
    return names


def _write_outputs(results: dict[str, Any], args: argparse.Namespace, title: str) -> None:
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        _log(f"wrote {args.json}")
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(render_report(results, title), encoding="utf-8")
        _log(f"wrote {args.report}")


def _print_ranking(results: dict[str, Any]) -> None:
    arms = results["dataset"]["arms"][1:]
    print("learner      " + "  ".join(f"qini[{a}]" for a in arms) + "  fit_s")
    for name, entry in results["learners"].items():
        cells = "  ".join(f"{entry['ranking'][a]['qini']['value']:+.4f}" for a in arms)
        print(f"{name:12s} {cells}  {entry['fit_seconds_mean']:.2f}")


def _run_evaluation(args: argparse.Namespace, budgets: Sequence[float] | None) -> dict[str, Any]:
    data, _ = load_dataset(args)
    names = _learner_names(args.learners)
    base = BaseLearner(kind=args.base, n_jobs=args.n_jobs)
    factories = {name: (lambda k, name=name: make_learner(name, _config(args, args.seed + k))) for name in names}
    costs = np.asarray(args.arm_cost, dtype=np.float64) if getattr(args, "arm_cost", None) else None
    results = evaluate(
        data,
        factories,
        base=base,
        n_folds=args.folds,
        n_boot=args.n_boot,
        seed=args.seed,
        budgets=budgets,
        costs=costs,
        progress=_log,
    )
    results["settings"]["learner_config"] = _config(args, args.seed).to_dict()
    results["dataset"]["source"] = args.data
    return results


def cmd_fetch(args: argparse.Namespace) -> int:
    try:
        path = fetch(args.name, force=args.force)
    except ChecksumError as err:
        _log(str(err))
        return 1
    source = SOURCES[args.name]
    print(f"{args.name}: {path} (sha256 {source.sha256[:12]}..., licence: {source.licence})")
    return 0


def cmd_synth(args: argparse.Namespace) -> int:
    data = make_synthetic(args.scenario, args.rows, seed=args.seed)
    frame = to_frame(data)
    write_table(frame, Path(args.out))
    print(f"wrote {data.n} rows of scenario {args.scenario!r} to {args.out}")
    return 0


def cmd_fit(args: argparse.Namespace) -> int:
    data, spec = load_dataset(args)
    learner = make_learner(args.learner, _config(args, args.seed))
    _log(f"fitting {args.learner} on {data.n} rows")
    learner.fit(data.X, data.arm, data.y, len(data.arm_names))
    meta = {"outcome": data.outcome, "rows": data.n, "data": args.data, "config": _config(args, args.seed).to_dict()}
    save_model(Path(args.out), learner, spec, data.arm_names, meta)
    print(f"saved {args.learner} model to {args.out}")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    results = _run_evaluation(args, budgets=None)
    _write_outputs(results, args, "Uplift evaluation report")
    _print_ranking(results)
    return 0


def cmd_policy(args: argparse.Namespace) -> int:
    if any(not 0 < b <= 1 for b in args.budget):
        raise SystemExit("--budget values are shares of customers in (0, 1]")
    results = _run_evaluation(args, budgets=args.budget)
    _write_outputs(results, args, "Uplift decision report")
    _print_ranking(results)
    for rec in recommend(results):
        g = rec["gain_vs_random"]
        verdict = "beats random" if rec["beats_random"] else "not distinguishable from random"
        print(
            f"budget {float(rec['budget']):.0%}: {rec['policy']} "
            f"{rec['per_1000_contacts']['value']:.1f} per 1,000 contacts; "
            f"gain vs random {g['value']:.1f} [{g['lo']:.1f}, {g['hi']:.1f}] per 1,000 customers ({verdict})"
        )
    return 0


def cmd_score(args: argparse.Namespace) -> int:
    model = load_model(Path(args.model))
    frame = read_table(Path(args.input))
    cate = model.learner.predict(model.spec.transform(frame))
    treatments = model.arm_names[1:]
    columns = {f"uplift__{name}": cate[:, k] for k, name in enumerate(treatments)}
    best = np.argmax(cate, axis=1)
    columns["best_arm"] = np.where(cate.max(axis=1) > 0, np.asarray(treatments)[best], model.arm_names[0])
    if args.budget is not None:
        costs = np.asarray(args.arm_cost, dtype=np.float64) if args.arm_cost else None
        allocation = allocate(cate, args.budget * frame.height, costs)
        columns["assigned_arm"] = np.asarray(model.arm_names)[allocation.assign]
        _log(f"contacting {int((allocation.assign > 0).sum())} of {frame.height} (spend {allocation.spend:g})")
    scored = frame.with_columns(pl.Series(name, values) for name, values in columns.items())
    write_table(scored, Path(args.output))
    print(f"scored {frame.height} rows -> {args.output}")
    return 0


def _add_data_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--data", required=True, help="hillstrom, criteo, synthetic:<scenario>, or a .parquet/.csv file")
    p.add_argument("--outcome", help="outcome column (hillstrom: visit|conversion|spend)")
    p.add_argument("--treatment-col", default="arm", help="treatment column for file input")
    p.add_argument("--control", default="control", help="control label for file input")
    p.add_argument("--rows", type=int, help="rows to generate (synthetic) or sample (criteo)")
    p.add_argument("--seed", type=int, default=0)


def _add_learner_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--base", choices=BASE_KINDS, default="lightgbm")
    p.add_argument("--propensity", choices=PROPENSITY_KINDS, default="constant")
    p.add_argument("--trees", type=int, help="trees per uplift forest")
    p.add_argument("--n-jobs", type=int, default=1)


def _add_eval_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--learners", default=DEFAULT_LEARNERS)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--json", help="write results JSON here")
    p.add_argument("--report", help="write the HTML report here")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="uplift", description=__doc__)
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("fetch", help="download a dataset and verify its sha256")
    p.add_argument("name", choices=sorted(SOURCES))
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_fetch)

    p = sub.add_parser("synth", help="write a synthetic experiment with known effects")
    p.add_argument("--scenario", choices=SCENARIOS, default="step")
    p.add_argument("--rows", type=int, default=20_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_synth)

    p = sub.add_parser("fit", help="fit one learner on all rows and save it")
    _add_data_args(p)
    _add_learner_args(p)
    p.add_argument("--learner", choices=LEARNER_NAMES, default="dr")
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_fit)

    p = sub.add_parser("evaluate", help="cross-fitted Qini/AUUC (and PEHE when effects are known)")
    _add_data_args(p)
    _add_learner_args(p)
    _add_eval_args(p)
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("policy", help="cross-fitted value of budgeted targeting, with an HTML decision report")
    _add_data_args(p)
    _add_learner_args(p)
    _add_eval_args(p)
    p.add_argument("--budget", type=float, nargs="+", default=[0.1, 0.2, 0.3], help="shares of customers")
    p.add_argument("--arm-cost", type=float, nargs="+", help="relative cost per treatment arm (default 1 each)")
    p.set_defaults(func=cmd_policy)

    p = sub.add_parser("score", help="score a Parquet/CSV file with a saved model")
    p.add_argument("--model", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--budget", type=float, help="also assign arms to this share of rows")
    p.add_argument("--arm-cost", type=float, nargs="+")
    p.set_defaults(func=cmd_score)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except FileNotFoundError as err:
        _log(f"error: {err}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
