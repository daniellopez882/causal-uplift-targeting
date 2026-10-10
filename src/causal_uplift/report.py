"""Self-contained HTML decision report rendered from an evaluation results dict."""

from __future__ import annotations

import html
import io
from collections.abc import Mapping
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from causal_uplift.evaluation import BEST_ARM_POLICY, RANDOM_POLICY, REFERENCE_POLICIES

_STYLE = """
body{font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif;color:#1d2125;background:#fff;
max-width:980px;margin:2rem auto;padding:0 1rem}
h1{font-size:1.6rem;margin-bottom:.2rem}h2{font-size:1.2rem;margin-top:2rem;border-bottom:1px solid #d6d9dc}
.muted{color:#5c6670}.decision{border-left:4px solid #2f6f4f;background:#f3f8f5;padding:.6rem 1rem;margin:1rem 0}
.decision.none{border-color:#8a6d1d;background:#fbf7ec}
table{border-collapse:collapse;width:100%;margin:.6rem 0;font-variant-numeric:tabular-nums}
th,td{border-bottom:1px solid #e3e6e8;padding:.3rem .5rem;text-align:right}
th:first-child,td:first-child{text-align:left}th{background:#f5f6f7}
figure{margin:1rem 0}figure svg{max-width:100%;height:auto}
code{background:#f1f3f4;padding:0 .2rem}
"""


def _fmt(interval: Mapping[str, float], digits: int = 1) -> str:
    lo, hi = interval["lo"], interval["hi"]
    return f"{interval['value']:.{digits}f} <span class=muted>[{lo:.{digits}f}, {hi:.{digits}f}]</span>"


def _units(results: Mapping[str, Any]) -> str:
    dataset = results["dataset"]
    outcome = str(dataset["outcome"])
    return f"{outcome}s" if dataset.get("binary_outcome") and not outcome.endswith("s") else outcome


def gain_key(reference: str, simultaneous: bool = False) -> str:
    """Result key of the paired AIPW gain against ``reference`` (random, best_arm or response)."""
    return f"aipw_gain_vs_{reference}{'_simultaneous' if simultaneous else ''}_per_1000_customers"


def recommend(results: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Per budget: the learner policy with the highest AIPW value, and the tests it has to pass.

    The rule as first written compared that policy with random targeting on its own 95%
    percentile interval (``first_rule``). Because the winner is picked by looking at the
    estimates, the rule now uses the simultaneous interval over all learner policies, and it
    also asks the winner to beat the best single arm sent to random customers, the strongest
    rule that needs no model. ``target`` is true only when both simultaneous lower bounds are
    above zero.
    """
    out: list[dict[str, Any]] = []
    for budget, policies in results.get("policy", {}).items():
        candidates = {k: v for k, v in policies.items() if k not in REFERENCE_POLICIES}
        if not candidates:
            continue
        best = max(candidates, key=lambda k: candidates[k]["aipw_per_1000_customers"]["value"])
        entry = candidates[best]
        rec: dict[str, Any] = {
            "budget": budget,
            "policy": best,
            "candidates": len(candidates),
            "per_1000_contacts": entry["aipw_per_1000_contacts"],
            "arm_mix": entry["arm_mix"],
            "gain_vs_random": entry[gain_key("random")],
            "gain_vs_random_simultaneous": entry[gain_key("random", simultaneous=True)],
        }
        rec["first_rule"] = rec["gain_vs_random"]["lo"] > 0
        rec["beats_random"] = rec["gain_vs_random_simultaneous"]["lo"] > 0
        rec["target"] = rec["beats_random"]
        reference = policies.get(BEST_ARM_POLICY)
        if reference is not None:
            arm, share = max(list(reference["arm_mix"].items())[1:], key=lambda kv: kv[1])
            rec["best_arm"] = {"arm": arm, "share": share, "gain_vs_random": reference[gain_key("random")]}
            rec["gain_vs_best_arm"] = entry[gain_key("best_arm")]
            rec["gain_vs_best_arm_simultaneous"] = entry[gain_key("best_arm", simultaneous=True)]
            rec["beats_best_arm"] = rec["gain_vs_best_arm_simultaneous"]["lo"] > 0
            rec["target"] = rec["beats_random"] and rec["beats_best_arm"]
        out.append(rec)
    return out


def _svg(fig: plt.Figure) -> str:
    buffer = io.StringIO()
    with matplotlib.rc_context({"svg.hashsalt": "causal-uplift", "svg.fonttype": "none"}):
        fig.savefig(buffer, format="svg", metadata={"Date": None, "Creator": None}, bbox_inches="tight")
    plt.close(fig)
    text = buffer.getvalue()
    return text[text.index("<svg") :]


def qini_figure(results: Mapping[str, Any]) -> str:
    learners = results["learners"]
    arms = list(results["dataset"]["arms"])[1:]
    fig, axes = plt.subplots(1, len(arms), figsize=(5.2 * len(arms), 3.8), squeeze=False, layout="constrained")
    for ax, arm in zip(axes[0], arms, strict=True):
        for name, entry in learners.items():
            curve = entry["ranking"][arm]["curve"]
            style = {"linestyle": "--", "color": "#888"} if name == "random" else {}
            ax.plot(curve["fraction"], curve["incremental_per_customer"], label=name, linewidth=1.4, **style)
        ax.axhline(0.0, color="#bbb", linewidth=0.8)
        ax.set_title(f"{arm} vs {results['dataset']['arms'][0]}", fontsize=10)
        ax.set_xlabel("share of customers targeted")
        ax.set_ylabel("incremental outcome per customer")
        ax.grid(alpha=0.3)
    axes[0][-1].legend(fontsize=8, frameon=False, loc="best")
    return _svg(fig)


_REFERENCE_NAMES = {"random": RANDOM_POLICY, "best_arm": BEST_ARM_POLICY}


def _gain_cell(p: Mapping[str, Any], name: str, reference: str, *, simultaneous: bool = False) -> str:
    if name == _REFERENCE_NAMES[reference]:
        return "reference"
    interval = p.get(gain_key(reference, simultaneous))
    return "n/a" if interval is None else _fmt(interval, 2)


def _policy_table(budget: str, policies: Mapping[str, Any], units: str, arms: list[str]) -> str:
    rows = sorted(policies.items(), key=lambda kv: -kv[1]["aipw_per_1000_customers"]["value"])
    n_candidates = sum(1 for name in policies if name not in REFERENCE_POLICIES)
    references = [r for r, name in _REFERENCE_NAMES.items() if name in policies]
    head = (
        "<tr><th>policy</th><th>contacted</th><th>arm mix (" + " / ".join(html.escape(a) for a in arms[1:]) + ")</th>"
        f"<th>incremental {units} per 1,000 contacts (AIPW)</th><th>same, IPW</th><th>per 1,000 customers (AIPW)</th>"
        + "".join(
            f"<th>gain vs {_REFERENCE_NAMES[r]} per 1,000 customers</th>"
            f"<th>same, simultaneous over {n_candidates} learners</th>"
            for r in references
        )
        + "</tr>"
    )
    body = []
    for name, p in rows:
        mix = " / ".join(f"{p['arm_mix'][a]:.0%}" for a in arms[1:])
        gains = "".join(
            f"<td>{_gain_cell(p, name, r)}</td><td>{_gain_cell(p, name, r, simultaneous=True)}</td>" for r in references
        )
        body.append(
            f"<tr><td>{html.escape(name)}</td><td>{p['contact_rate']:.1%}</td><td>{mix}</td>"
            f"<td>{_fmt(p['aipw_per_1000_contacts'])}</td><td>{p['ipw_per_1000_contacts']['value']:.1f}</td>"
            f"<td>{_fmt(p['aipw_per_1000_customers'])}</td>{gains}</tr>"
        )
    return f"<h3>Budget: {float(budget):.0%} of customers</h3><table>{head}{''.join(body)}</table>"


def _score_cell(interval: Mapping[str, float] | None) -> str:
    return "n/a" if interval is None else _fmt(interval, 3)


def _ranking_table(results: Mapping[str, Any]) -> str:
    arms = list(results["dataset"]["arms"])[1:]
    head = "<tr><th>learner</th>" + "".join(
        f"<th>Qini coef. {html.escape(a)}</th><th>AUUC {html.escape(a)}</th>" for a in arms
    )
    head += "<th>fit s / fold</th></tr>"
    body = []
    for name, entry in results["learners"].items():
        cells = "".join(
            f"<td>{_score_cell(entry['ranking'][a]['qini'])}</td><td>{_score_cell(entry['ranking'][a]['auuc'])}</td>"
            for a in arms
        )
        body.append(f"<tr><td>{html.escape(name)}</td>{cells}<td>{entry['fit_seconds_mean']:.2f}</td></tr>")
    return f"<table>{head}{''.join(body)}</table>"


def _pehe_table(results: Mapping[str, Any]) -> str:
    rows = [(n, e["pehe"]) for n, e in results["learners"].items() if "pehe" in e]
    if not rows:
        return ""
    cols = list(rows[0][1])
    head = "<tr><th>learner</th>" + "".join(f"<th>PEHE {html.escape(c)}</th>" for c in cols) + "</tr>"
    body = "".join(
        f"<tr><td>{html.escape(n)}</td>" + "".join(f"<td>{p[c]:.4f}</td>" for c in cols) + "</tr>" for n, p in rows
    )
    return (
        "<h2>Error against the known effects</h2><p>Root mean squared error between estimated and true "
        f"effects (PEHE), on held-out folds.</p><table>{head}{body}</table>"
    )


def _gains_text(rec: Mapping[str, Any]) -> str:
    gain, joint = rec["gain_vs_random"], rec["gain_vs_random_simultaneous"]
    tests = [
        f"against random targeting {gain['value']:.2f} per 1,000 customers (95% CI {gain['lo']:.2f} to "
        f"{gain['hi']:.2f}; simultaneous over the {rec['candidates']} learners {joint['lo']:.2f} to {joint['hi']:.2f})"
    ]
    best_arm = rec.get("best_arm")
    if best_arm is not None:
        g, j = rec["gain_vs_best_arm"], rec["gain_vs_best_arm_simultaneous"]
        tests.append(
            f"against sending {html.escape(best_arm['arm'])} to a random {best_arm['share']:.0%} {g['value']:.2f} "
            f"(95% CI {g['lo']:.2f} to {g['hi']:.2f}; simultaneous {j['lo']:.2f} to {j['hi']:.2f})"
        )
    return "Gain " + "; ".join(tests) + "."


def _decision_block(results: Mapping[str, Any], units: str) -> str:
    items = []
    for rec in recommend(results):
        head = f"<b>Budget {float(rec['budget']):.0%}:</b> "
        policy = html.escape(rec["policy"])
        if rec["target"]:
            per = rec["per_1000_contacts"]
            treatments = list(rec["arm_mix"].items())[1:]
            mix = ", ".join(f"{a} to {share:.0%}" for a, share in treatments if share > 0) + " of customers"
            items.append(
                f"<div class=decision>{head}target with <b>{policy}</b> (sending {html.escape(mix)}). Estimated "
                f"{per['value']:.1f} incremental {units} per 1,000 contacts (95% CI {per['lo']:.1f} to "
                f"{per['hi']:.1f}). {_gains_text(rec)}</div>"
            )
            continue
        rival = "the best arm sent at random" if rec["beats_random"] else "random targeting"
        text = (
            f"{head}the best point estimate is <b>{policy}</b>, but it does not separate from {rival} once the "
            f"choice among the {rec['candidates']} learners is allowed for, so model-based targeting is not "
            f"justified at this budget on this evidence. {_gains_text(rec)}"
        )
        best_arm = rec.get("best_arm")
        if best_arm is not None and best_arm["gain_vs_random"]["lo"] > 0:
            g = best_arm["gain_vs_random"]
            text += (
                f" Without a model, sending {html.escape(best_arm['arm'])} to a random {best_arm['share']:.0%} is "
                f"worth {g['value']:.2f} more per 1,000 customers than a random arm (95% CI {g['lo']:.2f} to "
                f"{g['hi']:.2f})."
            )
        items.append(f"<div class='decision none'>{text}</div>")
    return "".join(items)


def _appendix(settings: Mapping[str, Any], has_policy: bool) -> str:
    folds = settings["folds"]
    items = [
        f"Rows were split into {folds} folds stratified by arm and outcome. Every learner was fitted on {folds - 1} "
        "folds and scored the remaining one, so each customer's score comes from a model that never saw them.",
        f"Intervals are 95% percentile bootstraps over customers ({settings['n_boot']} replicates). Fitted models are "
        "held fixed across replicates, so the intervals do not include refitting variance.",
    ]
    if has_policy:
        n = settings["candidate_policies"]
        items += [
            "Budgets are applied inside each held-out fold: the top share of that fold by best predicted uplift is "
            "contacted, each with the arm of highest predicted uplift, skipping customers whose best predicted uplift "
            "is not positive. With unequal arm costs the budget optimiser makes that choice.",
            "Policy value uses inverse propensity weighting (IPW) and augmented IPW (AIPW). Assignment probabilities: "
            f"{html.escape(settings['propensity_source'])}. The AIPW outcome model is a cross-fitted per-arm "
            f"{html.escape(str(settings['base']['kind']))} model. Random targeting is scored in expectation over its "
            "randomisation. Policy comparisons resample all policies jointly, so the gains are paired estimates.",
            "Two policies use no model: random targeting (random customers, random arm) and, with more than one "
            "treatment, the best arm at random (random customers, all sent the arm with the highest average effect "
            "on these rows; choosing it with hindsight favours this reference).",
            f"The recommendation at each budget is the best of {n} learner policies on the same customers, so its "
            f"gains over both no-model policies are judged on simultaneous 95% intervals (max-t bootstrap) that "
            f"cover the gains of all {n} at once. Targeting is recommended only when both lower bounds are above "
            "zero. The point estimate of the chosen policy is still optimistic.",
            "Assumptions: given the features, the arm received does not depend on anything that also drives the "
            "outcome (true by design in a randomised test), every customer could have received every arm, one "
            "customer's e-mail does not change another's outcome, and the future population resembles this one.",
        ]
    body = "".join(f"<li>{item}</li>" for item in items)
    return f"<h2>Appendix: how these numbers were produced</h2><ul>{body}</ul>"


def render_report(results: Mapping[str, Any], title: str = "Uplift decision report") -> str:
    dataset = results["dataset"]
    settings = results["settings"]
    units = _units(results)
    arms = list(dataset["arms"])
    rates = ", ".join(f"{html.escape(a)} {dataset['outcome_mean_per_arm'][a]:.4f}" for a in arms)
    parts = [
        "<!doctype html><html lang=en><head><meta charset=utf-8>",
        "<meta name=viewport content='width=device-width,initial-scale=1'>",
        f"<title>{html.escape(title)}</title><style>{_STYLE}</style></head><body>",
        f"<h1>{html.escape(title)}</h1>",
        f"<p class=muted>{dataset['rows']:,} customers, outcome <code>{html.escape(str(dataset['outcome']))}</code>. "
        f"Mean outcome by arm: {rates}.</p>",
    ]
    if "policy" in results:
        parts.append(
            f"<h2>The question</h2><p>With a contact budget, which customers should receive which message so that "
            f"the number of {units} caused by the campaign is as large as possible?</p>"
        )
        parts.append("<h2>Recommendation</h2>" + _decision_block(results, units))
        parts.append("<h2>Policy value by budget</h2>")
        parts.extend(_policy_table(b, p, units, arms) for b, p in results["policy"].items())
    parts.append(
        "<h2>Ranking quality</h2><p>Normalised Qini coefficient and AUUC per treatment arm, "
        "with 95% bootstrap intervals; both are defined for 0/1 outcomes only (n/a otherwise).</p>"
    )
    parts.append(_ranking_table(results))
    parts.append(f"<figure>{qini_figure(results)}<figcaption class=muted>Held-out Qini curves.</figcaption></figure>")
    parts.append(_pehe_table(results))
    parts.append(_appendix(settings, "policy" in results))
    parts.append("</body></html>\n")
    return "\n".join(p for p in parts if p)
