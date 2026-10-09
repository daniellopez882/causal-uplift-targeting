"""Print the README tables from the benchmark JSON files.

    uv run python bench/tables.py [--results-dir bench/results] [--prefix smoke-]

Only numbers present in the result files are printed; nothing is typed by hand. The
recommendation lines apply the same rule as the HTML report (``causal_uplift.report.recommend``).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from causal_uplift.evaluation import RANDOM_POLICY, RESPONSE_POLICY
from causal_uplift.report import recommend

SKLIFT = "sklift-two-model"
FROM_SCRATCH = ("s", "t", "x", "r", "dr", "forest-kl", "forest-ed")
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


def label(name: str) -> str:
    return LABELS.get(name, name)


def _ci(interval: dict[str, float], digits: int) -> str:
    return f"{interval['value']:.{digits}f} [{interval['lo']:.{digits}f}, {interval['hi']:.{digits}f}]"


def measured_on(result: dict[str, Any]) -> str:
    m = result["machine"]
    return (
        f"Measured on {m['cpu']}, {m['logical_cores']} logical cores, {m['ram_gb']} GB RAM, {m['os']}, "
        f"Python {m['python']}, CPU only; {result['date']}, commit {result['git_sha'][:12]}, "
        f"`{result['command']}`, wall time {result['wall_seconds']} s."
    )


def _arm_mix(result: dict[str, Any], policy: dict[str, Any]) -> str:
    return " / ".join(f"{policy['arm_mix'][a]:.0%}" for a in result["dataset"]["arms"][1:])


def _policy_row(result: dict[str, Any], budget: str, name: str, shown: str, gain_digits: int) -> str:
    p = result["policy"][budget][name]
    vs_random = "reference" if name == RANDOM_POLICY else _ci(p["aipw_gain_vs_random_per_1000_customers"], gain_digits)
    vs_response = p.get("aipw_gain_vs_response_per_1000_customers")
    vs_response_cell = (
        "reference" if name == RESPONSE_POLICY else "n/a" if vs_response is None else _ci(vs_response, gain_digits)
    )
    return (
        f"| {float(budget):.0%} | {shown} | {_arm_mix(result, p)} | {_ci(p['aipw_per_1000_contacts'], 1)} "
        f"| {p['ipw_per_1000_contacts']['value']:.1f} | {vs_random} | {vs_response_cell} |"
    )


def _policy_header(result: dict[str, Any]) -> list[str]:
    units = result["dataset"]["outcome"]
    arms = " / ".join(a.replace(" E-Mail", "") for a in result["dataset"]["arms"][1:])
    return [
        f"| budget | policy | share e-mailed ({arms}) | incremental {units}s per 1,000 contacts, AIPW [95% CI] "
        "| same, IPW | gain vs random targeting per 1,000 customers [95% CI] "
        "| gain vs response model per 1,000 customers [95% CI] |",
        "|---|---|---|---|---|---|---|",
    ]


def best_from_scratch(result: dict[str, Any], budget: str) -> str:
    policies = result["policy"][budget]
    candidates = [n for n in FROM_SCRATCH if n in policies]
    return max(candidates, key=lambda n: policies[n]["aipw_per_1000_customers"]["value"])


def memo_policy_table(result: dict[str, Any], gain_digits: int = 2) -> str:
    """Per budget: the best uplift learner of this package, the two baselines and random targeting."""
    lines = _policy_header(result)
    for budget in result["policy"]:
        best = best_from_scratch(result, budget)
        rows = [(best, f"best uplift learner: {label(best)}")]
        rows += [(n, label(n)) for n in (RESPONSE_POLICY, SKLIFT, RANDOM_POLICY) if n in result["policy"][budget]]
        lines += [_policy_row(result, budget, name, shown, gain_digits) for name, shown in rows]
    return "\n".join(lines)


def full_policy_table(result: dict[str, Any], gain_digits: int = 2) -> str:
    """Every evaluated policy, ordered by AIPW value within each budget."""
    lines = _policy_header(result)
    for budget, policies in result["policy"].items():
        ordered = sorted(policies, key=lambda n: -policies[n]["aipw_per_1000_customers"]["value"])
        lines += [_policy_row(result, budget, name, label(name), gain_digits) for name in ordered]
    return "\n".join(lines)


def recommendation_lines(result: dict[str, Any], gain_digits: int = 2) -> str:
    lines = []
    for rec in recommend(result):
        gain = _ci(rec["gain_vs_random"], gain_digits)
        verdict = "interval above zero: target" if rec["beats_random"] else "interval includes zero: no model"
        lines.append(
            f"* {float(rec['budget']):.0%}: highest AIPW value is {label(rec['policy'])}; "
            f"gain vs random {gain} per 1,000 customers ({verdict})"
        )
    return "\n".join(lines)


def ranking_table(result: dict[str, Any]) -> str:
    arms = result["dataset"]["arms"][1:]
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
        lines.append(f"| {label(name)} | {qini} | {auuc} | {entry['fit_seconds_median']:.2f} |")
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", default=str(Path(__file__).resolve().parent / "results"))
    parser.add_argument("--prefix", default="", help="file prefix, e.g. smoke-")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # the tables contain "±"; Windows consoles default to a code page
    root = Path(args.results_dir)
    pre = args.prefix
    for path in sorted(root.glob(f"{pre}hillstrom_*.json")) + sorted(root.glob(f"{pre}criteo_*.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        print(f"## {path.name}\n\n{measured_on(result)}\n")
        print(memo_policy_table(result), "\n")
        print(recommendation_lines(result), "\n")
        print(ranking_table(result), "\n")
        print(full_policy_table(result), "\n")
    synthetic = root / f"{pre}synthetic.json"
    if synthetic.exists():
        result = json.loads(synthetic.read_text(encoding="utf-8"))
        print(f"## {synthetic.name}\n\n{measured_on(result)}\n")
        print(pehe_table(result), "\n")
        print(synthetic_qini_table(result), "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
