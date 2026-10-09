"""Download the public datasets and verify them against pinned sha256 values."""

from __future__ import annotations

import hashlib
import os
import shutil
import urllib.request
from dataclasses import dataclass
from pathlib import Path

DATA_DIR_ENV = "CAUSAL_UPLIFT_DATA_DIR"
_CHUNK = 1 << 20


@dataclass(frozen=True)
class Source:
    name: str
    url: str
    filename: str
    sha256: str
    size_bytes: int
    licence: str
    sha256_origin: str


SOURCES: dict[str, Source] = {
    "hillstrom": Source(
        name="hillstrom",
        url=(
            "http://www.minethatdata.com/"
            "Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv"
        ),
        filename="hillstrom.csv",
        sha256="0e5893329d8b93cefecc571777672028290ab69865718020c78c7284f291aece",
        size_bytes=3_964_977,
        licence=(
            "No explicit licence. Published by Kevin Hillstrom for the 2008 MineThatData "
            "E-Mail Analytics and Data Mining Challenge; not redistributed by this repository."
        ),
        sha256_origin="computed from the first download on 2026-10-09",
    ),
    "criteo": Source(
        name="criteo",
        # The original go.criteo.net link returns 404; this is Criteo's own Hugging Face copy.
        url=("https://huggingface.co/datasets/criteo/criteo-uplift/resolve/main/criteo-research-uplift-v2.1.csv.gz"),
        filename="criteo-research-uplift-v2.1.csv.gz",
        sha256="2716e1bf0fd157a93b5bf86924d9088419dfbac2022c6cd90030220634f616dc",
        size_bytes=311_422_618,
        licence="CC BY-NC-SA 4.0 (non-commercial, share-alike); not redistributed here.",
        sha256_origin="Git LFS object id published by the Hugging Face repository criteo/criteo-uplift",
    ),
}


class ChecksumError(RuntimeError):
    pass


def data_dir() -> Path:
    """Directory holding downloaded data; overridable with CAUSAL_UPLIFT_DATA_DIR."""
    configured = os.environ.get(DATA_DIR_ENV)
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".cache" / "causal-uplift-targeting"


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def local_path(name: str, root: Path | None = None) -> Path:
    return (root or data_dir()) / SOURCES[name].filename


def fetch(name: str, root: Path | None = None, *, force: bool = False, url: str | None = None) -> Path:
    """Download ``name`` into the data directory unless a verified copy is already there.

    ``url`` overrides the source location (used by tests with ``file://`` URLs); the
    pinned checksum is still enforced.
    """
    source = SOURCES[name]
    target = local_path(name, root)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not force:
        if sha256_of(target) == source.sha256:
            return target
        raise ChecksumError(
            f"{target} exists but its sha256 does not match the pinned value; "
            "delete it or pass force=True to download again"
        )
    partial = target.with_suffix(target.suffix + ".part")
    digest = hashlib.sha256()
    request = urllib.request.Request(url or source.url, headers={"User-Agent": "causal-uplift-targeting"})
    with urllib.request.urlopen(request, timeout=60) as response, partial.open("wb") as out:
        while chunk := response.read(_CHUNK):
            digest.update(chunk)
            out.write(chunk)
    observed = digest.hexdigest()
    if observed != source.sha256:
        partial.unlink(missing_ok=True)
        raise ChecksumError(f"sha256 mismatch for {name}: expected {source.sha256}, got {observed}")
    shutil.move(partial, target)
    return target
