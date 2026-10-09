"""Print the README tables from the benchmark JSON files.

    uv run python bench/tables.py [--results-dir bench/results]

Only numbers present in the result files are printed; nothing is typed by hand.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

RANDOM = "random targeting"


def _ci(interval: dict[str, float], digits: int = 1) -> str:
    return f"{interval['value']:.{digits}f} [{interval['lo']:.{digits}f}, {interval['hi']:.{digits}f}]"


def measured_on(result: dict[str, Any]) -> str:
    m = result["machine"]
    return (
        f"Measured on {m['cpu']}, {m['logical_cores']} logical cores, {m['ram_gb']} GB RAM, {m['os']}, "
        f"Python {m['python']}, CPU only; {result['date']}, commit {result['git_sha'][:12]}, "
        f"`{result['command']}`, wall time {result['wall_seconds']} s."
    )


def policy_table(result: dict[str, Any]) -> str:
    units = result["dataset"]["outcome"]
    lines = [
        f"| budget | policy | arm mix ({' / '.join(result['dataset']['arms'][1:])}) "
        f"| incremental {units}s per 1,000 contacts, AIPW [95% CI] | same, IPW "
        "| gain vs random per 1,000 customers [95% CI] |",
        "|---|---|---|---|---|---|",
    ]
    for budget, policies in result["policy"].items():
        ordered = sorted(policies.items(), key=lambda kv: -kv[1]["aipw_per_1000_customers"]["value"])
        for name, p in ordered:
            mix = " / ".join(f"{p['arm_mix'][a]:.0%}" for a in result["dataset"]["arms"][1:])
            gain = "reference" if name == RANDOM else _ci(p["aipw_gain_vs_random_per_1000_customers"])
            lines.append(
                f"| {float(budget):.0%} | {name} | {mix} | {_ci(p['aipw_per_1000_contacts'])} | "
                f"{p['ipw_per_1000_contacts']['value']:.1f} | {gain} |"
            )
    return "\n".join(lines)


def ranking_table(result: dict[str, Any]) -> str:
    arms = result["dataset"]["arms"][1:]
    header = "| learner | " + " | ".join(f"Qini coef. {a} | AUUC {a}" for a in arms) + " | fit s / fold |"
    lines = [header, "|---" * (2 * len(arms) + 2) + "|"]
    for name, entry in result["learners"].items():
        cells = " | ".join(
            f"{_ci(entry['ranking'][a]['qini'], 3)} | {_ci(entry['ranking'][a]['auuc'], 3)}" for a in arms
        )
        lines.append(f"| {name} | {cells} | {entry['fit_seconds_mean']:.2f} |")
    return "\n".join(lines)


def pehe_table(result: dict[str, Any]) -> str:
    scenarios = result["scenarios"]
    learners = list(next(iter(scenarios.values()))["learners"])
    lines = ["| learner | " + " | ".join(scenarios) + " |", "|---" * (len(scenarios) + 1) + "|"]
    for name in learners:
        cells = []
        for scenario in scenarios.values():
            p = scenario["learners"][name]["pehe"]
            cells.append(f"{p['mean']:.4f} ± {p['sd']:.4f}")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def synthetic_qini_table(result: dict[str, Any]) -> str:
    randomised = {k: v for k, v in result["scenarios"].items() if v["randomised"]}
    learners = list(next(iter(randomised.values()))["learners"])
    lines = [
        "| learner | " + " | ".join(f"Qini arm1, {s}" for s in randomised) + " |",
        "|---" * (len(randomised) + 1) + "|",
    ]
    for name in learners:
        cells = [f"{s['learners'][name]['qini_arm1']['mean']:.3f}" for s in randomised.values()]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", default=str(Path(__file__).resolve().parent / "results"))
    parser.add_argument("--prefix", default="", help="file prefix, e.g. smoke-")
    args = parser.parse_args()
    root = Path(args.results_dir)
    pre = args.prefix
    for path in sorted(root.glob(f"{pre}hillstrom_*.json")) + sorted(root.glob(f"{pre}criteo_*.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        print(f"## {path.name}\n\n{measured_on(result)}\n")
        print(policy_table(result), "\n")
        print(ranking_table(result), "\n")
    synthetic = root / f"{pre}synthetic.json"
    if synthetic.exists():
        result = json.loads(synthetic.read_text(encoding="utf-8"))
        print(f"## {synthetic.name}\n\n{measured_on(result)}\n")
        print(pehe_table(result), "\n")
        print(synthetic_qini_table(result), "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
