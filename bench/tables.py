"""Print the README tables from the benchmark JSON files.

    uv run python bench/tables.py [--results-dir bench/results] [--prefix smoke-] [--timings-dir DIR]

Only numbers present in the result files are printed; nothing is typed by hand. The decision
tables apply the same rule as the HTML report (``causal_uplift.report.recommend``).

Fit times in the Hillstrom ranking tables come from ``--timings-dir`` when it holds a file of
the same name (by default the first full run, kept in ``bench/results/2026-10-09``). Later
re-runs changed only the policy evaluation, not the learners, and were made while other jobs
shared the machine, so their timings are recorded but not printed.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from causal_uplift.evaluation import BEST_ARM_POLICY, RANDOM_POLICY, RESPONSE_POLICY
from causal_uplift.report import gain_key, recommend

SKLIFT = "sklift-two-model"
FROM_SCRATCH = ("s", "t", "x", "r", "dr", "forest-kl", "forest-ed")
FIRST_FULL_RUN = Path(__file__).resolve().parent / "results" / "2026-10-09"
LABELS = {
    "s": "S-learner",
    "t": "T-learner",
    "x": "X-learner",
    "r": "R-learner",
    "dr": "DR-learner",
    "forest-kl": "uplift forest, KL (honest)",
    "forest-kl-adaptive": "uplift forest, KL (adaptive)",
    "forest-ed": "uplift forest, Euclidean (honest)",
    RESPONSE_POLICY: "response model",
    SKLIFT: "scikit-uplift two-model",
    "random": "random scores",
    RANDOM_POLICY: "random targeting",
    "oracle": "oracle (true effects)",
}


def label(name: str, result: dict[str, Any] | None = None, budget: str | None = None) -> str:
    if name == BEST_ARM_POLICY and result is not None and budget is not None:
        return f"best single e-mail at random ({_arm(result['policy'][budget][name], result)})"
    return LABELS.get(name, name)


def _short(arm: str) -> str:
    return arm.replace(" E-Mail", "")


def _arm(policy: dict[str, Any], result: dict[str, Any]) -> str:
    """The treatment a single-arm policy sends."""
    treatments = result["dataset"]["arms"][1:]
    return _short(max(treatments, key=lambda a: policy["arm_mix"][a]))


def _ci(interval: dict[str, float] | None, digits: int) -> str:
    if interval is None:
        return "n/a"
    return f"{interval['value']:.{digits}f} [{interval['lo']:.{digits}f}, {interval['hi']:.{digits}f}]"


def _arm_mix(result: dict[str, Any], policy: dict[str, Any]) -> str:
    return " / ".join(f"{policy['arm_mix'][a]:.0%}" for a in result["dataset"]["arms"][1:])


def _gain_cell(policy: dict[str, Any], name: str, reference: str, digits: int) -> str:
    own = {"random": RANDOM_POLICY, "best_arm": BEST_ARM_POLICY, "response": RESPONSE_POLICY}[reference]
    return "reference" if name == own else _ci(policy.get(gain_key(reference)), digits)


def _policy_row(result: dict[str, Any], budget: str, name: str, shown: str, digits: int) -> str:
    p = result["policy"][budget][name]
    gains = " | ".join(_gain_cell(p, name, r, digits) for r in ("random", "best_arm", "response"))
    return (
        f"| {float(budget):.0%} | {shown} | {_arm_mix(result, p)} | {_ci(p['aipw_per_1000_contacts'], 1)} "
        f"| {p['ipw_per_1000_contacts']['value']:.1f} | {gains} |"
    )


def _policy_header(result: dict[str, Any]) -> list[str]:
    units = result["dataset"]["outcome"]
    arms = " / ".join(_short(a) for a in result["dataset"]["arms"][1:])
    return [
        f"| budget | policy | share e-mailed ({arms}) | incremental {units}s per 1,000 contacts, AIPW [95% CI] "
        "| same, IPW | gain vs random targeting [95% CI] | gain vs best single e-mail at random [95% CI] "
        "| gain vs response model [95% CI] |",
        "|---|---|---|---|---|---|---|---|",
    ]


def best_from_scratch(result: dict[str, Any], budget: str) -> str:
    policies = result["policy"][budget]
    candidates = [n for n in FROM_SCRATCH if n in policies]
    return max(candidates, key=lambda n: policies[n]["aipw_per_1000_customers"]["value"])


def memo_policy_table(result: dict[str, Any], digits: int = 2) -> str:
    """Per budget: the best uplift learner of this package, the baselines and the no-model policies."""
    lines = _policy_header(result)
    for budget, policies in result["policy"].items():
        best = best_from_scratch(result, budget)
        rows = [(best, f"best uplift learner: {label(best)}")]
        rows += [
            (n, label(n, result, budget))
            for n in (RESPONSE_POLICY, SKLIFT, BEST_ARM_POLICY, RANDOM_POLICY)
            if n in policies
        ]
        lines += [_policy_row(result, budget, name, shown, digits) for name, shown in rows]
    return "\n".join(lines)


def full_policy_table(result: dict[str, Any], digits: int = 2) -> str:
    """Every evaluated policy, ordered by AIPW value within each budget."""
    lines = _policy_header(result)
    for budget, policies in result["policy"].items():
        ordered = sorted(policies, key=lambda n: -policies[n]["aipw_per_1000_customers"]["value"])
        lines += [_policy_row(result, budget, name, label(name, result, budget), digits) for name in ordered]
    return "\n".join(lines)


def decision_table(result: dict[str, Any], digits: int = 2) -> str:
    """Per budget: the winner, its gains with and without the allowance for picking it, and the verdicts."""
    n = result["settings"]["candidate_policies"]
    lines = [
        f"| budget | highest AIPW value | gain vs random targeting [95% CI] | same, simultaneous over {n} learners "
        "| gain vs best single e-mail at random [95% CI] | same, simultaneous "
        "| rule as first written | revised rule |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for rec in recommend(result):
        first = "target" if rec["first_rule"] else "no model"
        revised = "target" if rec["target"] else "no model"
        lines.append(
            f"| {float(rec['budget']):.0%} | {label(rec['policy'])} | {_ci(rec['gain_vs_random'], digits)} "
            f"| {_ci(rec['gain_vs_random_simultaneous'], digits)} | {_ci(rec['gain_vs_best_arm'], digits)} "
            f"| {_ci(rec['gain_vs_best_arm_simultaneous'], digits)} | {first} | {revised} |"
        )
    return "\n".join(lines)


def ranking_table(result: dict[str, Any], timings: dict[str, Any] | None = None) -> str:
    """Qini and AUUC per e-mail; fit times from ``timings`` (another run of the same suite) when given."""
    arms = result["dataset"]["arms"][1:]
    source = timings or result
    header = (
        "| learner | "
        + " | ".join(f"Qini coef., {a} [95% CI]" for a in arms)
        + " | "
        + " | ".join(f"AUUC, {a} [95% CI]" for a in arms)
        + " | median fit s / fold |"
    )
    lines = [header, "|---" * (2 * len(arms) + 2) + "|"]
    for name, entry in result["learners"].items():
        qini = " | ".join(_ci(entry["ranking"][a]["qini"], 3) for a in arms)
        auuc = " | ".join(_ci(entry["ranking"][a]["auuc"], 4) for a in arms)
        lines.append(f"| {label(name)} | {qini} | {auuc} | {source['learners'][name]['fit_seconds_median']:.2f} |")
    return "\n".join(lines)


def pehe_table(result: dict[str, Any]) -> str:
    scenarios = result["scenarios"]
    learners = list(next(iter(scenarios.values()))["learners"])
    header = "| learner | " + " | ".join(scenarios) + " | mean fit s (20k rows, all scenarios) |"
    lines = [header, "|---" * (len(scenarios) + 2) + "|"]
    for name in learners:
        cells = []
        fits = []
        for scenario in scenarios.values():
            p = scenario["learners"][name]["pehe"]
            cells.append(f"{p['mean']:.4f} ± {p['sd']:.4f}")
            fits.append(scenario["learners"][name]["fit_seconds"]["mean"])
        fit = "n/a" if name == "oracle" else f"{sum(fits) / len(fits):.2f}"
        lines.append(f"| {label(name)} | " + " | ".join(cells) + f" | {fit} |")
    return "\n".join(lines)


def synthetic_qini_table(result: dict[str, Any]) -> str:
    randomised = {k: v for k, v in result["scenarios"].items() if v["randomised"]}
    learners = list(next(iter(randomised.values()))["learners"])
    lines = [
        "| learner | " + " | ".join(f"{s}, arm1 | {s}, arm2" for s in randomised) + " |",
        "|---" * (2 * len(randomised) + 1) + "|",
    ]
    for name in learners:
        cells = []
        for s in randomised.values():
            entry = s["learners"][name]
            cells += [f"{entry['qini_arm1']['mean']:.3f}", f"{entry['qini_arm2']['mean']:.3f}"]
        lines.append(f"| {label(name)} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _machine(result: dict[str, Any]) -> str:
    m = result["machine"]
    cpu = re.sub(r"\s+", " ", re.sub(r"\((R|TM)\)|CPU @.*$", "", m["cpu"])).strip()
    match = re.match(r"(\w+)-(\d+)-([\d.]+)", m["os"])
    system = f"{match[1]} {match[2]} ({match[3]})" if match else m["os"]
    return (
        f"{cpu}, {m['logical_cores']} cores, {round(m['ram_gb'])} GB RAM ({m['ram_gb']} GB usable), {system}, CPU only"
    )


def _software(result: dict[str, Any]) -> str:
    packages = result["machine"]["packages"]
    shown = ("numpy", "polars", "lightgbm", "scikit-learn", "numba", "scikit-uplift")
    return f"Python {result['machine']['python']}; " + ", ".join(f"{p} {packages[p]}" for p in shown)


def run_details(runs: list[tuple[str, str, dict[str, Any]]]) -> str:
    """One row per result file: what it supplies, when and from which commit it was measured."""
    first = runs[0][2]
    lines = [
        "| run detail | value |",
        "|---|---|",
        f"| Measured on | {_machine(first)} |",
        f"| software | {_software(first)} |",
    ]
    for what, path, result in runs:
        lines.append(
            f"| {what} (`{path}`) | {result['date'][:10]}, commit `{result['git_sha'][:7]}`, "
            f"`{result['command']}`, wall time {result['wall_seconds']} s |"
        )
    return "\n".join(lines)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", default=str(Path(__file__).resolve().parent / "results"))
    parser.add_argument("--prefix", default="", help="file prefix, e.g. smoke-")
    parser.add_argument("--timings-dir", default=str(FIRST_FULL_RUN), help="run whose fit times are printed")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # the tables contain "±"; Windows consoles default to a code page
    root = Path(args.results_dir)
    timings_dir = Path(args.timings_dir)
    pre = args.prefix
    runs: list[tuple[str, str, dict[str, Any]]] = []
    for path in sorted(root.glob(f"{pre}hillstrom_*.json")) + sorted(root.glob(f"{pre}criteo_*.json")):
        result = _load(path)
        timing_path = timings_dir / path.name
        timings = _load(timing_path) if timing_path.exists() else None
        runs.append(("policy values and ranking", path.name, result))
        if timings is not None:
            runs.append(("fit times", f"{timings_dir.name}/{path.name}", timings))
        print(f"## {path.name}\n")
        print(decision_table(result), "\n")
        print(memo_policy_table(result), "\n")
        print(ranking_table(result, timings), "\n")
        print(full_policy_table(result), "\n")
    synthetic = root / f"{pre}synthetic.json"
    if synthetic.exists():
        result = _load(synthetic)
        runs.append(("synthetic suite", synthetic.name, result))
        print(f"## {synthetic.name}\n")
        print(pehe_table(result), "\n")
        print(synthetic_qini_table(result), "\n")
    if runs:
        print("## run details\n")
        print(run_details(runs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
