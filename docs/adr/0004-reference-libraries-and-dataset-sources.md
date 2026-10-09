# ADR 0004: scikit-uplift as the only reference library; dataset sources

Status: accepted, 2026-10-09

## Context

The spec names scikit-uplift and causalml as parity references, and the Hillstrom and Criteo uplift
data sets. Every claim must be reproducible from a clean checkout on a machine with limited disk,
and CI must pass without the data sets.

## Decision

* scikit-uplift 0.5.1 is a development dependency only. It is the parity reference for the Qini and
  uplift curves, the perfect curves and the normalised scores (agreement to 1e-9 with and without
  tied scores). Through `sklift.models.TwoModels` it is also the parity test for the T-learner and
  the "two-model" benchmark baseline. The package never imports it.
* causalml is not installed. Its install pulls xgboost, shap, statsmodels and pygam among others; the
  forest is tested against a brute-force split search instead (ADR 0002).
* Hillstrom is fetched from the author's own URL and checked against a sha256 recorded from the
  first download on 2026-10-09.
* The Criteo uplift v2.1 link on go.criteo.net returned 404 on 2026-10-09. The fetch command uses
  Criteo's own Hugging Face repository (`criteo/criteo-uplift`), and the pinned sha256 is the Git LFS
  object id that repository publishes. The file was not downloaded while building this project, so
  that checksum has not been verified locally, and the Criteo scale run is reported as not run.
* Data live under `CAUSAL_UPLIFT_DATA_DIR` (default `~/.cache/causal-uplift-targeting`), never in
  git. Tests that need the real files skip when they are absent.

## Consequences

* The default test suite and CI run without network access.
* Parity is checked against one independent implementation, not two.
* If the Hillstrom URL disappears there is no committed mirror; the checksum still lets a user check
  a copy obtained elsewhere.

## Alternatives rejected

* Committing Hillstrom to the repository: the file carries no explicit licence, so it is not
  redistributed.
* `sklift.datasets.fetch_hillstrom`: it downloads a re-encoded copy from a third-party bucket rather
  than the author's file.
