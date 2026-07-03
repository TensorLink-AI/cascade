"""Fetch the full benchmark datasets and wire them to the suites.

Each suite reads its data from a directory named by an env var (``GIFT_EVAL`` /
``BOOM`` / ``CASCADE_BENCH_TIME_DATASET``). Historically you had to download those
by hand; this module pulls each benchmark's HuggingFace dataset repo into a local
cache and returns the env-var → path mapping the suites expect.

The three repos are the upstream sources the suites are written against:

* GIFT-Eval → ``Salesforce/GiftEval``
* BOOM      → ``Datadog/BOOM``
* TIME      → ``Real-TSF/TIME``

Each is stored one ``datasets.save_to_disk`` dir per config (``<name>/<freq>/``),
which is exactly the layout gift-eval's / timebench's ``Dataset`` loads via
``load_from_disk(storage_path / name)``.

Set ``HF_TOKEN`` if a repo is gated. Downloads are resumable — ``huggingface_hub``
skips files already present, so re-running only fetches what's missing.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetSpec:
    """Where a suite's data lives upstream and how the suite finds it locally.

    ``revision`` pins the exact HF repo commit — benchmark numbers are only
    comparable across rounds/machines if the data is frozen, and upstream
    benchmark repos do get revised. Bump deliberately; the revision is recorded
    in every ``BenchmarkReport`` so historical numbers stay traceable.
    """

    suite: str
    hf_repo: str
    env_var: str
    revision: str


# suite name (matches suites.SUITES keys) → dataset source (revisions pinned
# 2026-07-02)
DATASETS: dict[str, DatasetSpec] = {
    "gift-eval": DatasetSpec(
        "gift-eval", "Salesforce/GiftEval", "GIFT_EVAL",
        revision="30841734ac5cfddbd0c3bad6d09d2b6b32becbb0",
    ),
    "boom": DatasetSpec(
        "boom", "Datadog/BOOM", "BOOM",
        revision="69325b544c45ff0d6c43c7a99c49a6601a01725b",
    ),
    "time": DatasetSpec(
        "time", "Real-TSF/TIME", "CASCADE_BENCH_TIME_DATASET",
        revision="83e3d0b3be28d11c7182bffcc1892d19b36c4da1",
    ),
}


def _looks_populated(path: Path) -> bool:
    """True if ``path`` already holds downloaded data (any dataset arrow file)."""
    return path.is_dir() and any(path.rglob("*.arrow"))


def download_suite(
    suite: str,
    dest: str | Path,
    *,
    allow_patterns: Sequence[str] | None = None,
) -> Path:
    """Download one benchmark's HF dataset repo into ``dest`` and return the path.

    ``allow_patterns`` restricts the pull (e.g. ``["ds-0-T/*"]`` for a quick
    subset); omit it to fetch the entire benchmark.
    """
    if suite not in DATASETS:
        raise KeyError(f"unknown benchmark {suite!r}; known: {', '.join(DATASETS)}")
    from huggingface_hub import snapshot_download

    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=DATASETS[suite].hf_repo,
        repo_type="dataset",
        revision=DATASETS[suite].revision,
        local_dir=str(dest),
        allow_patterns=list(allow_patterns) if allow_patterns else None,
    )
    return dest


def ensure_datasets(
    suites: Sequence[str],
    data_root: str | Path,
    *,
    download: bool = True,
    allow_patterns: Sequence[str] | None = None,
) -> dict[str, str]:
    """Ensure each suite's data is under ``data_root/<suite>`` and map env → path.

    With ``download=True`` (default) missing data is fetched; with ``download=False``
    only already-present dirs are wired up (missing ones are skipped, so the suite
    reports ``skipped`` rather than erroring). Returns ``{env_var: path}`` for the
    suites whose data is available — assign these into ``os.environ`` before running.
    """
    data_root = Path(data_root)
    env: dict[str, str] = {}
    for suite in suites:
        spec = DATASETS.get(suite)
        if spec is None:  # e.g. a suite with no downloadable dataset
            continue
        dest = data_root / suite
        if download and not _looks_populated(dest):
            download_suite(suite, dest, allow_patterns=allow_patterns)
        if _looks_populated(dest):
            env[spec.env_var] = str(dest)
    return env


def apply_env(env: dict[str, str]) -> None:
    """Point the suites at downloaded data by setting their env vars in-process."""
    for k, v in env.items():
        os.environ[k] = v
