from __future__ import annotations

import gzip
import hashlib
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from causal_uplift import fetch as fetch_module
from causal_uplift.data import (
    FeatureSpec,
    TableSchema,
    frame_to_data,
    load_criteo,
    load_hillstrom,
    read_table,
    write_table,
)
from causal_uplift.fetch import ChecksumError, Source, data_dir, fetch, sha256_of
from causal_uplift.synthetic import ARM_NAMES, SCENARIOS, assignment_probabilities, make_synthetic, potential_means

HILLSTROM_HEADER = "recency,history_segment,history,mens,womens,zip_code,newbie,channel,segment,visit,conversion,spend"


def _fake_hillstrom(path: Path, n: int = 60) -> Path:
    """Rows in the Hillstrom layout with invented values (the real file is never committed)."""
    rng = np.random.default_rng(1)
    segments = ["Mens E-Mail", "Womens E-Mail", "No E-Mail"]
    lines = [HILLSTROM_HEADER]
    for i in range(n):
        lines.append(
            f"{rng.integers(1, 13)},1) $0 - $100,{rng.uniform(29, 900):.2f},{i % 2},{(i + 1) % 2},"
            f"{['Urban', 'Rural', 'Surburban'][i % 3]},{i % 2},{['Web', 'Phone', 'Multichannel'][i % 3]},"
            f"{segments[i % 3]},{int(rng.random() < 0.3)},{int(rng.random() < 0.05)},{0.0 if i % 7 else 55.5}"
        )
    path.write_bytes(("\r\n".join(lines) + "\r\n").encode())  # CRLF, like the original file
    return path


def test_feature_spec_round_trip_and_unseen_levels() -> None:
    frame = pl.DataFrame({"a": [1.0, 2.0, None], "c": ["x", "y", "x"], "b": [True, False, True]})
    spec = FeatureSpec.infer(frame, ["a", "c", "b"])
    assert spec.names == ("a", "b", "c=x", "c=y")
    X = spec.transform(frame)
    assert X.shape == (3, 4)
    assert np.isnan(X[2, 0])
    np.testing.assert_array_equal(X[:, 2], [1.0, 0.0, 1.0])
    again = FeatureSpec.from_dict(spec.to_dict())
    unseen = pl.DataFrame({"a": [0.5], "c": ["z"], "b": [False]})
    np.testing.assert_array_equal(again.transform(unseen), [[0.5, 0.0, 0.0, 0.0]])
    with pytest.raises(KeyError, match="missing feature columns"):
        spec.transform(frame.drop("c"))


def test_frame_to_data_puts_control_first_and_reads_true_effects() -> None:
    frame = pl.DataFrame(
        {
            "x": [0.1, 0.2, 0.3, 0.4],
            "arm": ["b", "control", "a", "control"],
            "y": [1, 0, 1, 0],
            "true_cate__a": [0.1, 0.1, 0.1, 0.1],
            "true_cate__b": [0.2, 0.2, 0.2, 0.2],
        }
    )
    data, spec = frame_to_data(frame, TableSchema(treatment="arm", outcome="y", control="control"))
    assert data.arm_names == ("control", "a", "b")
    np.testing.assert_array_equal(data.arm, [2, 0, 1, 0])
    assert spec.names == ("x",)
    assert data.true_cate is not None
    assert data.true_cate.shape == (4, 2)
    np.testing.assert_allclose(data.design_propensity()[0], [0.5, 0.25, 0.25])
    with pytest.raises(ValueError, match="control label"):
        frame_to_data(frame, TableSchema(treatment="arm", outcome="y", control="none"))


def test_load_hillstrom_layout(tmp_path: Path) -> None:
    data, spec = load_hillstrom("conversion", path=_fake_hillstrom(tmp_path / "h.csv"))
    assert data.arm_names == ("No E-Mail", "Mens E-Mail", "Womens E-Mail")
    assert "history_segment" not in " ".join(spec.names)
    assert spec.numeric == ("recency", "history", "mens", "womens", "newbie")
    assert data.X.shape == (60, 5 + 3 + 3)
    assert data.binary_outcome
    spend, _ = load_hillstrom("spend", path=tmp_path / "h.csv")
    assert not spend.binary_outcome
    with pytest.raises(ValueError, match="outcome"):
        load_hillstrom("revenue", path=tmp_path / "h.csv")


def test_load_hillstrom_missing_file_names_the_fetch_command(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="uplift fetch hillstrom"):
        load_hillstrom(path=tmp_path / "absent.csv")


def test_load_criteo_subsamples_and_relabels(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    header = ",".join([*(f"f{i}" for i in range(12)), "treatment", "conversion", "visit", "exposure"])
    rows = [
        ",".join([*(f"{v:.3f}" for v in rng.normal(size=12)), str(i % 2), str(int(i % 9 == 0)), "1", "0"])
        for i in range(200)
    ]
    path = tmp_path / "criteo.csv.gz"
    with gzip.open(path, "wt") as handle:
        handle.write("\n".join([header, *rows]) + "\n")
    data, spec = load_criteo("conversion", n_rows=50, path=path)
    assert data.n == 50
    assert data.arm_names == ("no ad", "ad")
    assert spec.names == tuple(f"f{i}" for i in range(12))


def test_table_io_round_trip(tmp_path: Path) -> None:
    frame = pl.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    for name in ("t.parquet", "t.csv"):
        write_table(frame, tmp_path / "sub" / name)
        assert read_table(tmp_path / "sub" / name).equals(frame)
    with pytest.raises(ValueError, match="unsupported"):
        read_table(tmp_path / "t.json")


@pytest.fixture
def fake_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Source, Path]:
    payload = tmp_path / "remote.csv"
    payload.write_bytes(b"a,b\n1,2\n")
    source = Source(
        name="toy",
        url=payload.as_uri(),
        filename="toy.csv",
        sha256=hashlib.sha256(payload.read_bytes()).hexdigest(),
        size_bytes=8,
        licence="test",
        sha256_origin="test",
    )
    monkeypatch.setitem(fetch_module.SOURCES, "toy", source)
    return source, payload


def test_fetch_verifies_and_reuses_downloads(tmp_path: Path, fake_source: tuple[Source, Path]) -> None:
    source, payload = fake_source
    target = fetch("toy", tmp_path / "cache")
    assert sha256_of(target) == source.sha256
    payload.write_bytes(b"changed")  # a second call must not download again
    assert fetch("toy", tmp_path / "cache") == target


def test_fetch_rejects_wrong_checksum(tmp_path: Path, fake_source: tuple[Source, Path]) -> None:
    _, payload = fake_source
    payload.write_bytes(b"tampered")
    with pytest.raises(ChecksumError, match="sha256 mismatch"):
        fetch("toy", tmp_path / "cache")
    assert not any((tmp_path / "cache").iterdir())


def test_data_dir_honours_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("CAUSAL_UPLIFT_DATA_DIR", str(tmp_path))
    assert data_dir() == tmp_path
    monkeypatch.delenv("CAUSAL_UPLIFT_DATA_DIR")
    assert data_dir() == Path.home() / ".cache" / "causal-uplift-targeting"


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_synthetic_effects_match_potential_outcomes(scenario: str) -> None:
    data = make_synthetic(scenario, 3000, seed=5)
    means = potential_means(scenario, data.X)
    assert data.true_cate is not None
    assert data.propensity is not None
    np.testing.assert_allclose(data.true_cate, means[:, 1:] - means[:, :1])
    np.testing.assert_allclose(data.propensity.sum(axis=1), 1.0)
    assert data.arm_names == ARM_NAMES
    assert set(np.unique(data.arm)) == {0, 1, 2}
    again = make_synthetic(scenario, 3000, seed=5)
    np.testing.assert_array_equal(data.y, again.y)


def test_confounded_assignment_depends_on_features() -> None:
    X = np.array([[-2.0, 0, 0, 0], [2.0, 0, 0, 0]])
    probs = assignment_probabilities("confounded", X)
    assert probs[1, 1] > 0.6 > probs[0, 1]
    np.testing.assert_allclose(assignment_probabilities("step", X), 1.0 / 3.0)
