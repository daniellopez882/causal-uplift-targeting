from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from causal_uplift import fetch as fetch_module
from causal_uplift.cli import main
from causal_uplift.fetch import Source
from causal_uplift.persist import META_FILE, load_model
from causal_uplift.report import recommend, render_report

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


def test_recommendation_flags_budgets_without_evidence() -> None:
    results = json.loads((GOLDEN / "results.json").read_text(encoding="utf-8"))
    budget = next(iter(results["policy"]))
    for name, entry in results["policy"][budget].items():
        if name != "random targeting":
            entry["aipw_gain_vs_random_per_1000_customers"] = {"value": 1.0, "lo": -2.0, "hi": 4.0}
    verdicts = {r["budget"]: r["beats_random"] for r in recommend(results)}
    assert verdicts[budget] is False
    assert "not distinguishable from zero" in render_report(results)


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
    assert set(results["policy"]["0.2"]) == {"t", "forest-ed", "response", "random targeting"}
    assert "pehe" in results["learners"]["t"]  # the synthetic file carries the true effects
    policy = results["policy"]["0.2"]["t"]
    assert policy["contact_rate"] <= 0.2 + 1e-9
    assert "Recommendation" in (tmp_path / "p.html").read_text(encoding="utf-8")


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
