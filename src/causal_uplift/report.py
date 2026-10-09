"""Self-contained HTML decision report rendered from an evaluation results dict."""

from __future__ import annotations

import html
import io
from collections.abc import Mapping
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from causal_uplift.evaluation import RANDOM_POLICY

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


def recommend(results: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Per budget: the policy with the highest AIPW value, and whether it beats random targeting."""
    out: list[dict[str, Any]] = []
    for budget, policies in results.get("policy", {}).items():
        candidates = {k: v for k, v in policies.items() if k != RANDOM_POLICY}
        if not candidates:
            continue
        best = max(candidates, key=lambda k: candidates[k]["aipw_per_1000_customers"]["value"])
        gain = candidates[best]["aipw_gain_vs_random_per_1000_customers"]
        out.append(
            {
                "budget": budget,
                "policy": best,
                "beats_random": gain["lo"] > 0,
                "per_1000_contacts": candidates[best]["aipw_per_1000_contacts"],
                "gain_vs_random": gain,
                "arm_mix": candidates[best]["arm_mix"],
            }
        )
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


def _policy_table(budget: str, policies: Mapping[str, Any], units: str, arms: list[str]) -> str:
    rows = sorted(policies.items(), key=lambda kv: -kv[1]["aipw_per_1000_customers"]["value"])
    head = (
        "<tr><th>policy</th><th>contacted</th><th>arm mix (" + " / ".join(html.escape(a) for a in arms[1:]) + ")</th>"
        f"<th>incremental {units} per 1,000 contacts (AIPW)</th><th>same, IPW</th>"
        f"<th>per 1,000 customers (AIPW)</th><th>gain vs random per 1,000 customers</th></tr>"
    )
    body = []
    for name, p in rows:
        mix = " / ".join(f"{p['arm_mix'][a]:.0%}" for a in arms[1:])
        body.append(
            f"<tr><td>{html.escape(name)}</td><td>{p['contact_rate']:.1%}</td><td>{mix}</td>"
            f"<td>{_fmt(p['aipw_per_1000_contacts'])}</td><td>{p['ipw_per_1000_contacts']['value']:.1f}</td>"
            f"<td>{_fmt(p['aipw_per_1000_customers'])}</td><td>{_fmt(p['aipw_gain_vs_random_per_1000_customers'])}</td></tr>"
        )
    return f"<h3>Budget: {float(budget):.0%} of customers</h3><table>{head}{''.join(body)}</table>"


def _ranking_table(results: Mapping[str, Any]) -> str:
    arms = list(results["dataset"]["arms"])[1:]
    head = "<tr><th>learner</th>" + "".join(
        f"<th>Qini coef. {html.escape(a)}</th><th>AUUC {html.escape(a)}</th>" for a in arms
    )
    head += "<th>fit s / fold</th></tr>"
    body = []
    for name, entry in results["learners"].items():
        cells = "".join(
            f"<td>{_fmt(entry['ranking'][a]['qini'], 3)}</td><td>{_fmt(entry['ranking'][a]['auuc'], 3)}</td>"
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


def _decision_block(results: Mapping[str, Any], units: str) -> str:
    items = []
    for rec in recommend(results):
        budget = float(rec["budget"])
        per = rec["per_1000_contacts"]
        gain = rec["gain_vs_random"]
        treatments = list(rec["arm_mix"].items())[1:]
        mix = ", ".join(f"{a} to {share:.0%}" for a, share in treatments if share > 0) + " of customers"
        if rec["beats_random"]:
            items.append(
                f"<div class=decision><b>Budget {budget:.0%}:</b> target with <b>{html.escape(rec['policy'])}</b> "
                f"(sending {html.escape(mix)}). Estimated {per['value']:.1f} incremental {units} per 1,000 contacts "
                f"(95% CI {per['lo']:.1f} to {per['hi']:.1f}); {gain['value']:.1f} more per 1,000 customers than "
                f"random targeting (95% CI {gain['lo']:.1f} to {gain['hi']:.1f}).</div>"
            )
        else:
            items.append(
                f"<div class='decision none'><b>Budget {budget:.0%}:</b> the best point estimate is "
                f"<b>{html.escape(rec['policy'])}</b>, but its gain over random targeting "
                f"({gain['value']:.1f} per 1,000 customers, 95% CI {gain['lo']:.1f} to {gain['hi']:.1f}) "
                "is not distinguishable from zero. Model-based targeting is not justified at this budget "
                "on this evidence.</div>"
            )
    return "".join(items)


_APPENDIX = """
<h2>Appendix: how these numbers were produced</h2>
<ul>
<li>Rows were split into {folds} folds stratified by arm and outcome. Every learner was fitted on
{folds_minus} folds and scored the remaining one, so each customer's score comes from a model that never saw them.</li>
<li>Budgets are applied inside each held-out fold: the top share of that fold by best predicted uplift is contacted,
each with the arm of highest predicted uplift, skipping customers whose best predicted uplift is not positive.</li>
<li>Policy value uses inverse propensity weighting (IPW) and augmented IPW (AIPW). Propensities are known by
design (the arm shares of a randomised test); the AIPW outcome model is a cross-fitted per-arm {base} model.
Random targeting is scored in expectation over its randomisation.</li>
<li>Intervals are 95% percentile bootstraps over customers ({n_boot} replicates). Policy comparisons resample
all policies jointly, so the gain against random targeting is a paired estimate. Fitted models are held fixed
across replicates, so the intervals do not include refitting variance.</li>
<li>Assumptions: the arm was randomised (or its propensities are known), one customer's e-mail does not change
another's outcome, and the future population resembles this one. The recommended policy is the best of several
on the same data, so its estimate is slightly optimistic.</li>
</ul>
"""


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
        "with 95% bootstrap intervals.</p>"
    )
    parts.append(_ranking_table(results))
    parts.append(f"<figure>{qini_figure(results)}<figcaption class=muted>Held-out Qini curves.</figcaption></figure>")
    parts.append(_pehe_table(results))
    parts.append(
        _APPENDIX.format(
            folds=settings["folds"],
            folds_minus=settings["folds"] - 1,
            n_boot=settings["n_boot"],
            base=html.escape(str(settings["base"]["kind"])),
        )
    )
    parts.append("</body></html>\n")
    return "\n".join(p for p in parts if p)
