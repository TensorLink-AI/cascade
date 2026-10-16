"""Scrape kings vs private (vault/direct) submissions.

The CI archiver holds no submission vault, so every private ref failed with
``generator_artifact_unreachable`` — classified TRANSIENT, so the run failed
daily from 2026-09-13 (415 private refs on 2026-10-14). A private submission
the caller cannot read is PRIVATE (tombstoned, parked), like a private Hub
repo; crowned ones resolve through the public champions objects.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from cascade.shared.hippius import (
    FETCH_PRIVATE,
    FETCH_TRANSIENT,
    StorageError,
    classify_fetch_failure,
)

DIGEST = "f33f40a427ba15e1fee088763b1e934342f047e7030c64f5919a735fbf9aef54"


def test_vault_miss_without_champion_base_is_private():
    # Verbatim from the 2026-10-14 04:48 CI run.
    exc = StorageError(
        f"generator_artifact_unreachable: vault digest {DIGEST} not present in "
        "$CASCADE_VAULT_DIR and no $CASCADE_CHAMPION_BASE configured — private "
        "submissions resolve only where their ZIP was stored or staged")
    assert classify_fetch_failure(exc) == FETCH_PRIVATE


@pytest.mark.parametrize("status", ["HTTP Error 404: Not Found", "HTTP Error 403: Forbidden"])
def test_uncrowned_champion_miss_is_private(status):
    exc = StorageError(
        f"generator_artifact_unreachable: vault digest {DIGEST} is not in "
        f"$CASCADE_VAULT_DIR and the champion fetch failed: {status}")
    assert classify_fetch_failure(exc) == FETCH_PRIVATE


@pytest.mark.parametrize("err", [
    "<urlopen error timed out>",
    "<urlopen error [Errno -3] Temporary failure in name resolution>",
    "[Errno 104] Connection reset by peer",
    "HTTP Error 503: Service Unavailable",
])
def test_champion_network_failure_stays_transient(err):
    exc = StorageError(
        f"generator_artifact_unreachable: vault digest {DIGEST} is not in "
        f"$CASCADE_VAULT_DIR and the champion fetch failed: {err}")
    assert classify_fetch_failure(exc) == FETCH_TRANSIENT


def test_unrelated_unreachable_stays_transient():
    assert classify_fetch_failure(
        StorageError("generator_artifact_unreachable: hub timeout")) == FETCH_TRANSIENT


def _load_scraper():
    path = Path(__file__).resolve().parents[2] / "scripts" / "scrape_kings.py"
    spec = importlib.util.spec_from_file_location("scrape_kings_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_scraper_defaults_champion_base_from_storage(monkeypatch):
    from cascade.funding.store import CHAMPION_BASE_ENV

    monkeypatch.delenv(CHAMPION_BASE_ENV, raising=False)
    mod = _load_scraper()
    mod._default_champion_base(SimpleNamespace(s3_endpoint="https://s3.example/",
                                               manifest_bucket="cascade-manifests"))
    import os
    assert os.environ[CHAMPION_BASE_ENV] == "https://s3.example/cascade-manifests"


def test_scraper_keeps_an_explicit_champion_base(monkeypatch):
    from cascade.funding.store import CHAMPION_BASE_ENV

    monkeypatch.setenv(CHAMPION_BASE_ENV, "https://override.example/b")
    mod = _load_scraper()
    mod._default_champion_base(SimpleNamespace(s3_endpoint="https://s3.example",
                                               manifest_bucket="cascade-manifests"))
    import os
    assert os.environ[CHAMPION_BASE_ENV] == "https://override.example/b"
