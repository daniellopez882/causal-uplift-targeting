from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest

from causal_uplift import fetch as fetch_module
from causal_uplift.cli import main
from causal_uplift.fetch import Source
from causal_uplift.persist import META_FILE, load_model
from causal_uplift.report import gain_key, recommend, render_report
from causal_uplift.synthetic import make_synthetic, to_frame

GOLDEN = Path(__file__).parent / "golden"
SVG = re.compile(r"<svg.*?</svg>", re.DOTALL)


def _without_figures(page: str) -> str:
    return SVG.sub("<svg/>", page)


def test_report_matches_golden() -> None:
    results = json.loads((GOLDEN / "results.json").read_text(encoding="utf-8"))
    page = render_report(results)
    assert len(SVG.findall(page)) == 1
    assert "<text" in SVG.findall(page)[0]  # labels stay text, so the figure is searchable
    expected_path = GOLDEN / "report.html"
    if os.environ.get("UPDATE_GOLDEN"):
        expected_path.write_text(_without_figures(page), encoding="utf-8", newline="\n")
    assert _without_figures(page) == expected_path.read_text(encoding="utf-8")


def _golden_with_gains(gains: dict[str, dict[str, float]]) -> dict[str, Any]:
    """The golden results with every learner's gains at the first budget replaced by ``gains``."""
    results = json.loads((GOLDEN / "results.json").read_text(encoding="utf-8"))
    budget = next(iter(results["policy"]))
    for name, entry in results["policy"][budget].items():
        if name not in ("random targeting", "best arm at random"):
            entry.update({key: dict(interval) for key, interval in gains.items()})
    return results


ABOVE = {"value": 3.0, "lo": 1.0, "hi": 5.0}
STRADDLING = {"value": 1.0, "lo": -2.0, "hi": 4.0}


def test_recommendation_flags_budgets_without_evidence() -> None:
    results = _golden_with_gains(
        {
            gain_key("random"): STRADDLING,
            gain_key("random", simultaneous=True): STRADDLING,
            gain_key("best_arm"): STRADDLING,
            gain_key("best_arm", simultaneous=True): STRADDLING,
        }
    )
    rec = recommend(results)[0]
    assert (rec["first_rule"], rec["beats_random"], rec["target"]) == (False, False, False)
    assert "does not separate from random targeting" in render_report(results)


def test_recommendation_uses_the_simultaneous_interval_and_the_best_arm() -> None:
    # The winner's own interval clears zero, but the band over all learners does not: no targeting.
    results = _golden_with_gains(
        {
            gain_key("random"): ABOVE,
            gain_key("random", simultaneous=True): STRADDLING,
            gain_key("best_arm"): ABOVE,
            gain_key("best_arm", simultaneous=True): ABOVE,
        }
    )
    rec = recommend(results)[0]
    assert (rec["first_rule"], rec["beats_random"], rec["target"]) == (True, False, False)
    # Beating random targeting is not enough when sending the best arm at random does as well.
    results = _golden_with_gains(
        {
            gain_key("random"): ABOVE,
            gain_key("random", simultaneous=True): ABOVE,
            gain_key("best_arm"): STRADDLING,
            gain_key("best_arm", simultaneous=True): STRADDLING,
        }
    )
    rec = recommend(results)[0]
    assert (rec["beats_random"], rec["beats_best_arm"], rec["target"]) == (True, False, False)
    assert "does not separate from the best arm sent at random" in render_report(results)
    results = _golden_with_gains(
        {
            gain_key("random"): ABOVE,
            gain_key("random", simultaneous=True): ABOVE,
            gain_key("best_arm"): ABOVE,
            gain_key("best_arm", simultaneous=True): ABOVE,
        }
    )
    rec = recommend(results)[0]
    assert rec["target"] is True
    assert rec["candidates"] == results["settings"]["candidate_policies"] == 3
    assert "target with" in render_report(results)


def test_cli_policy_end_to_end(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    table = tmp_path / "synth.parquet"
    assert main(["synth", "--scenario", "sleeping_dogs", "--rows", "2400", "--out", str(table)]) == 0
    args = ["policy", "--data", str(table), "--learners", "t,forest-ed,response,random", "--folds", "2"]
    args += ["--n-boot", "30", "--budget", "0.2", "--trees", "10"]
    args += ["--json", str(tmp_path / "p.json"), "--report", str(tmp_path / "p.html")]
    assert main(args) == 0
    out = capsys.readouterr().out
    assert "budget 20%:" in out
    results = json.loads((tmp_path / "p.json").read_text(encoding="utf-8"))
    assert set(results) == {"dataset", "settings", "learners", "policy"}
    assert set(results["policy"]["0.2"]) == {"t", "forest-ed", "response", "random targeting", "best arm at random"}
    assert results["settings"]["candidate_policies"] == 3
    assert results["settings"]["propensity_source"].startswith("recorded")  # the synthetic file carries them
    assert "pehe" in results["learners"]["t"]  # the synthetic file carries the true effects
    assert results["settings"]["ci_level"] == 0.95
    assert sum(results["settings"]["train_rows_per_fold"]) == 2400  # two folds: each row trains once
    policy = results["policy"]["0.2"]["t"]
    assert policy["contact_rate"] <= 0.2 + 1e-9
    response = results["policy"]["0.2"]["response"]
    assert response["aipw_gain_vs_response_per_1000_customers"] == {"value": 0.0, "lo": 0.0, "hi": 0.0}
    # Both gains are paired differences of the same AIPW scores, so they differ by the response gain.
    via_random = (
        policy["aipw_gain_vs_random_per_1000_customers"]["value"]
        - response["aipw_gain_vs_random_per_1000_customers"]["value"]
    )
    assert policy["aipw_gain_vs_response_per_1000_customers"]["value"] == pytest.approx(via_random, abs=1e-9)
    # The simultaneous band shares the point estimate and is at least as wide as the percentile interval.
    joint = policy["aipw_gain_vs_random_simultaneous_per_1000_customers"]
    single = policy["aipw_gain_vs_random_per_1000_customers"]
    assert joint["value"] == single["value"]
    assert joint["hi"] - joint["lo"] >= 0.9 * (single["hi"] - single["lo"])
    assert "aipw_gain_vs_random_simultaneous_per_1000_customers" not in results["policy"]["0.2"]["random targeting"]
    assert "Recommendation" in (tmp_path / "p.html").read_text(encoding="utf-8")


def test_cli_handles_a_continuous_outcome(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rng = np.random.default_rng(0)
    frame = to_frame(make_synthetic("step", 1200, seed=1))
    frame = frame.with_columns(spend=pl.Series(frame["purchase"].to_numpy() * rng.gamma(2.0, 40.0, frame.height)))
    table = tmp_path / "spend.parquet"
    frame.drop("purchase").write_parquet(table)
    args = ["evaluate", "--data", str(table), "--outcome", "spend", "--base", "linear", "--folds", "2"]
    args += ["--n-boot", "10", "--trees", "5", "--json", str(tmp_path / "e.json")]
    assert main(args) == 0
    captured = capsys.readouterr()
    assert "forest-kl is left out" in captured.err
    results = json.loads((tmp_path / "e.json").read_text(encoding="utf-8"))
    assert "forest-kl" not in results["learners"]
    assert results["learners"]["forest-ed"]["ranking"]["arm1"]["qini"] is None  # not defined for spend
    assert "n/a" in render_report(results)
    with pytest.raises(SystemExit, match="needs a 0/1 outcome"):
        main([*args[:-2], "--learners", "t,forest-kl"])


def test_cli_fit_then_score_with_budget(tmp_path: Path) -> None:
    table = tmp_path / "synth.parquet"
    main(["synth", "--scenario", "step", "--rows", "3000", "--out", str(table)])
    model_dir = tmp_path / "model"
    assert main(["fit", "--data", str(table), "--learner", "t", "--out", str(model_dir)]) == 0
    meta = json.loads((model_dir / META_FILE).read_text(encoding="utf-8"))
    assert meta["arm_names"] == ["control", "arm1", "arm2"]
    customers = pl.read_parquet(table).drop("arm", "purchase").head(500)
    customers.write_csv(tmp_path / "customers.csv")
    out = tmp_path / "scored.parquet"
    assert (
        main(
            [
                "score",
                "--model",
                str(model_dir),
                "--input",
                str(tmp_path / "customers.csv"),
                "--output",
                str(out),
                "--budget",
                "0.1",
            ]
        )
        == 0
    )
    scored = pl.read_parquet(out)
    assert {"uplift__arm1", "uplift__arm2", "best_arm", "assigned_arm"} <= set(scored.columns)
    assert (scored["assigned_arm"] != "control").sum() <= 50
    loaded = load_model(model_dir)
    np.testing.assert_allclose(
        loaded.learner.predict(loaded.spec.transform(customers))[:, 0], scored["uplift__arm1"].to_numpy()
    )


def test_cli_reports_missing_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("CAUSAL_UPLIFT_DATA_DIR", str(tmp_path))
    assert main(["evaluate", "--data", "hillstrom", "--learners", "t"]) == 2
    assert "uplift fetch hillstrom" in capsys.readouterr().err
    with pytest.raises(SystemExit, match="unknown learners"):
        main(["evaluate", "--data", "synthetic:step", "--learners", "t,magic"])


def test_cli_fetch_uses_the_pinned_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    payload = tmp_path / "remote.csv"
    payload.write_bytes(b"x\n1\n")

    source = Source("toy", payload.as_uri(), "toy.csv", hashlib.sha256(b"x\n1\n").hexdigest(), 4, "test", "test")
    monkeypatch.setitem(fetch_module.SOURCES, "toy", source)
    monkeypatch.setenv("CAUSAL_UPLIFT_DATA_DIR", str(tmp_path / "data"))
    assert main(["fetch", "toy"]) == 0
    assert (tmp_path / "data" / "toy.csv").exists()
    assert "toy:" in capsys.readouterr().out


def test_load_model_rejects_unknown_format(tmp_path: Path) -> None:
    (tmp_path / META_FILE).write_text(json.dumps({"format": 99}), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported model format"):
        load_model(tmp_path)
