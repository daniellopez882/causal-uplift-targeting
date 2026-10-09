"""Uplift datasets: a common container, feature encoding, and loaders."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from causal_uplift.fetch import local_path

TRUE_CATE_PREFIX = "true_cate__"


@dataclass(frozen=True)
class UpliftData:
    """Features, assigned arm and outcome for one experiment.

    ``arm`` is 0 for control and 1..K for the K treatments; ``arm_names[0]`` names control.
    ``true_cate`` (n, K) and ``propensity`` (n, K + 1) are only present when known by design.
    """

    X: np.ndarray
    arm: np.ndarray
    y: np.ndarray
    feature_names: tuple[str, ...]
    arm_names: tuple[str, ...]
    outcome: str
    true_cate: np.ndarray | None = None
    propensity: np.ndarray | None = None

    def __post_init__(self) -> None:
        n = self.X.shape[0]
        if self.arm.shape != (n,) or self.y.shape != (n,):
            raise ValueError("X, arm and y must have the same number of rows")
        if self.X.shape[1] != len(self.feature_names):
            raise ValueError("feature_names does not match the number of columns of X")
        if self.arm.min() < 0 or self.arm.max() >= len(self.arm_names):
            raise ValueError("arm codes must lie in 0..len(arm_names)-1")

    @property
    def n(self) -> int:
        return int(self.X.shape[0])

    @property
    def n_treatments(self) -> int:
        return len(self.arm_names) - 1

    @property
    def binary_outcome(self) -> bool:
        return bool(np.isin(self.y, (0.0, 1.0)).all())

    def subset(self, index: np.ndarray) -> UpliftData:
        return UpliftData(
            X=self.X[index],
            arm=self.arm[index],
            y=self.y[index],
            feature_names=self.feature_names,
            arm_names=self.arm_names,
            outcome=self.outcome,
            true_cate=None if self.true_cate is None else self.true_cate[index],
            propensity=None if self.propensity is None else self.propensity[index],
        )

    def design_propensity(self) -> np.ndarray:
        """Known assignment probabilities, or the empirical arm shares for a randomised test."""
        if self.propensity is not None:
            return self.propensity
        shares = np.bincount(self.arm, minlength=len(self.arm_names)) / self.n
        return np.broadcast_to(shares, (self.n, len(self.arm_names))).copy()


@dataclass(frozen=True)
class FeatureSpec:
    """How raw table columns become the numeric matrix the learners see.

    Stored with every saved model so batch scoring encodes new files identically.
    """

    numeric: tuple[str, ...]
    categorical: Mapping[str, tuple[str, ...]] = field(default_factory=dict)

    @classmethod
    def infer(cls, frame: pl.DataFrame, columns: Sequence[str]) -> FeatureSpec:
        numeric: list[str] = []
        categorical: dict[str, tuple[str, ...]] = {}
        for name in columns:
            dtype = frame.schema[name]
            if dtype.is_numeric() or dtype == pl.Boolean:
                numeric.append(name)
            else:
                levels = frame[name].cast(pl.String).drop_nulls().unique().sort().to_list()
                categorical[name] = tuple(levels)
        return cls(numeric=tuple(numeric), categorical=categorical)

    @property
    def names(self) -> tuple[str, ...]:
        encoded = [f"{col}={level}" for col, levels in self.categorical.items() for level in levels]
        return self.numeric + tuple(encoded)

    def transform(self, frame: pl.DataFrame) -> np.ndarray:
        missing = [c for c in (*self.numeric, *self.categorical) if c not in frame.columns]
        if missing:
            raise KeyError(f"input is missing feature columns: {missing}")
        exprs = [pl.col(c).cast(pl.Float64).fill_null(float("nan")).alias(c) for c in self.numeric]
        for col, levels in self.categorical.items():
            exprs.extend(
                (pl.col(col).cast(pl.String) == level).fill_null(False).cast(pl.Float64).alias(f"{col}={level}")
                for level in levels
            )
        return np.ascontiguousarray(frame.select(exprs).to_numpy(), dtype=np.float64)

    def to_dict(self) -> dict[str, object]:
        return {"numeric": list(self.numeric), "categorical": {k: list(v) for k, v in self.categorical.items()}}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> FeatureSpec:
        return cls(
            numeric=tuple(payload["numeric"]),
            categorical={k: tuple(v) for k, v in payload["categorical"].items()},
        )


@dataclass(frozen=True)
class TableSchema:
    treatment: str
    outcome: str
    control: str
    features: tuple[str, ...] | None = None


def frame_to_data(
    frame: pl.DataFrame, schema: TableSchema, spec: FeatureSpec | None = None
) -> tuple[UpliftData, FeatureSpec]:
    """Turn a table into :class:`UpliftData`; columns named ``true_cate__<arm>`` carry known effects."""
    labels = frame[schema.treatment].cast(pl.String)
    observed = sorted(set(labels.to_list()))
    if schema.control not in observed:
        raise ValueError(f"control label {schema.control!r} not found in column {schema.treatment!r}")
    arm_names = (schema.control, *[a for a in observed if a != schema.control])
    code = {name: i for i, name in enumerate(arm_names)}
    arm = np.array([code[v] for v in labels.to_list()], dtype=np.int64)

    truth_cols = [c for c in frame.columns if c.startswith(TRUE_CATE_PREFIX)]
    excluded = {schema.treatment, schema.outcome, *truth_cols}
    if spec is None:
        columns = schema.features or tuple(c for c in frame.columns if c not in excluded)
        spec = FeatureSpec.infer(frame, columns)
    true_cate = None
    if truth_cols:
        true_cate = np.column_stack([frame[f"{TRUE_CATE_PREFIX}{a}"].to_numpy() for a in arm_names[1:]]).astype(
            np.float64
        )
    data = UpliftData(
        X=spec.transform(frame),
        arm=arm,
        y=frame[schema.outcome].cast(pl.Float64).to_numpy().copy(),
        feature_names=spec.names,
        arm_names=arm_names,
        outcome=schema.outcome,
        true_cate=true_cate,
    )
    return data, spec


def read_table(path: Path) -> pl.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pl.read_parquet(path)
    if suffix in {".csv", ".gz"}:
        return pl.read_csv(path, infer_schema_length=None)
    raise ValueError(f"unsupported table format: {path.name} (use .parquet or .csv)")


def write_table(frame: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".parquet":
        frame.write_parquet(path)
    else:
        frame.write_csv(path)


HILLSTROM_OUTCOMES = ("visit", "conversion", "spend")
HILLSTROM_FEATURES = ("recency", "history", "mens", "womens", "newbie", "zip_code", "channel")


def load_hillstrom(outcome: str = "visit", path: Path | None = None) -> tuple[UpliftData, FeatureSpec]:
    """MineThatData e-mail test: 64,000 customers randomised to Mens, Womens or no e-mail.

    ``history_segment`` is dropped because it is a bucketed copy of ``history``.
    """
    if outcome not in HILLSTROM_OUTCOMES:
        raise ValueError(f"outcome must be one of {HILLSTROM_OUTCOMES}")
    source = path or local_path("hillstrom")
    if not source.exists():
        raise FileNotFoundError(f"{source} not found; run `uplift fetch hillstrom` first")
    frame = pl.read_csv(source, infer_schema_length=None)
    schema = TableSchema(treatment="segment", outcome=outcome, control="No E-Mail", features=HILLSTROM_FEATURES)
    return frame_to_data(frame, schema)


CRITEO_FEATURES = tuple(f"f{i}" for i in range(12))


def load_criteo(
    outcome: str = "conversion", n_rows: int = 1_000_000, seed: int = 0, path: Path | None = None
) -> tuple[UpliftData, FeatureSpec]:
    """Uniform random subsample of the Criteo uplift v2.1 file (13.9M rows, binary treatment)."""
    if outcome not in {"conversion", "visit"}:
        raise ValueError("outcome must be 'conversion' or 'visit'")
    source = path or local_path("criteo")
    if not source.exists():
        raise FileNotFoundError(f"{source} not found; run `uplift fetch criteo` first")
    dtypes = dict.fromkeys(CRITEO_FEATURES, pl.Float32)
    frame = pl.read_csv(source, schema_overrides=dtypes, columns=[*CRITEO_FEATURES, "treatment", outcome])
    if n_rows < frame.height:
        frame = frame.sample(n=n_rows, seed=seed)
    frame = frame.with_columns(
        pl.when(pl.col("treatment") == 1).then(pl.lit("ad")).otherwise(pl.lit("no ad")).alias("treatment")
    )
    schema = TableSchema(treatment="treatment", outcome=outcome, control="no ad", features=CRITEO_FEATURES)
    return frame_to_data(frame, schema)
