"""Dataset provenance — the download marker, resume-on-partial behaviour, and
true-revision reporting. snapshot_download is stubbed; what's under test is
when it is (re-)invoked and what the marker records."""

from __future__ import annotations

from pathlib import Path

import cascade_benchmark.datasets as ds


def _stub_snapshot(monkeypatch, calls: list, fail: bool = False):
    def fake(repo_id, repo_type, revision, local_dir, allow_patterns=None):
        if fail:
            raise OSError("offline")
        calls.append((revision, allow_patterns))
        sub = Path(local_dir) / "cfg"
        sub.mkdir(parents=True, exist_ok=True)
        (sub / "data-00000-of-00001.arrow").write_bytes(b"x")

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake)
    # The pinned revision's HF listing: exactly what the fake pull writes.
    monkeypatch.setattr(ds, "_expected_files", lambda spec, pats: [ARROW])


ARROW = "cfg/data-00000-of-00001.arrow"


def test_download_writes_marker_and_recorded_revision(tmp_path: Path, monkeypatch):
    calls: list = []
    _stub_snapshot(monkeypatch, calls)
    dest = ds.download_suite("gift-eval", tmp_path / "gift-eval")
    pinned = ds.DATASETS["gift-eval"].revision
    assert calls == [(pinned, None)]
    assert ds.recorded_revision(dest) == pinned


def test_partial_download_is_resumed_not_trusted(tmp_path: Path, monkeypatch):
    """Regression: arrow files without a completion marker (an interrupted
    pull) must trigger the resumable download again, not be scored as-is."""
    calls: list = []
    _stub_snapshot(monkeypatch, calls)
    dest = tmp_path / "gift-eval"
    (dest / "cfg").mkdir(parents=True)
    (dest / "cfg" / "data-00000-of-00001.arrow").write_bytes(b"partial")
    env = ds.ensure_datasets(["gift-eval"], tmp_path, download=True)
    assert len(calls) == 1  # re-invoked despite the existing arrow file
    assert env == {"GIFT_EVAL": str(dest)}
    # …and now that the marker matches the pin, a second call skips the pull
    ds.ensure_datasets(["gift-eval"], tmp_path, download=True)
    assert len(calls) == 1


def test_stale_pin_triggers_resync(tmp_path: Path, monkeypatch):
    calls: list = []
    _stub_snapshot(monkeypatch, calls)
    dest = ds.download_suite("boom", tmp_path / "boom")
    marker = dest / ds._MARKER
    marker.write_text(marker.read_text().replace(
        ds.DATASETS["boom"].revision, "0" * 40), encoding="utf-8")
    ds.ensure_datasets(["boom"], tmp_path, download=True)
    assert len(calls) == 2  # old-revision marker → re-sync to the pin


def test_partial_patterns_flagged_in_revision(tmp_path: Path, monkeypatch):
    calls: list = []
    _stub_snapshot(monkeypatch, calls)
    dest = ds.download_suite("time", tmp_path / "time", allow_patterns=["ds-0/*"])
    rev = ds.recorded_revision(dest)
    assert rev is not None and rev.endswith("(partial)")


def test_offline_with_existing_data_wires_it_anyway(tmp_path: Path, monkeypatch, capsys):
    _stub_snapshot(monkeypatch, [], fail=True)
    dest = tmp_path / "gift-eval"
    (dest / "cfg").mkdir(parents=True)
    (dest / "cfg" / "data-00000-of-00001.arrow").write_bytes(b"x")
    env = ds.ensure_datasets(["gift-eval"], tmp_path, download=True)
    assert env == {"GIFT_EVAL": str(dest)}          # usable data still wired
    assert "download failed" in capsys.readouterr().err
    assert ds.recorded_revision(dest) is None       # provenance honestly unknown


def test_hand_managed_dir_has_unknown_revision(tmp_path: Path):
    (tmp_path / "x.arrow").write_bytes(b"x")
    assert ds.recorded_revision(tmp_path) is None


def test_rate_limited_pull_missing_files_writes_no_marker(tmp_path: Path, monkeypatch):
    """Regression (miner report 2026-10-30): snapshot_download returned after
    HF rate limits with a few hundred of BOOM's 2,807 datasets and the old
    code still wrote the completion marker."""
    import pytest

    calls: list = []
    _stub_snapshot(monkeypatch, calls)
    monkeypatch.setattr(ds, "_expected_files", lambda spec, pats: [ARROW, "cfg2/data.arrow"])
    dest = tmp_path / "boom"
    with pytest.raises(RuntimeError, match="1 of 2 files"):
        ds.download_suite("boom", dest)
    assert not (dest / ds._MARKER).exists()
    assert ds.recorded_revision(dest) is None


def test_failed_repull_removes_the_previous_marker(tmp_path: Path, monkeypatch):
    calls: list = []
    _stub_snapshot(monkeypatch, calls)
    dest = ds.download_suite("boom", tmp_path / "boom")
    _stub_snapshot(monkeypatch, calls, fail=True)
    import pytest

    with pytest.raises(OSError):
        ds.download_suite("boom", dest)
    assert not (dest / ds._MARKER).exists()


def test_marker_records_listing_and_lost_files_read_as_partial(tmp_path: Path, monkeypatch):
    calls: list = []
    _stub_snapshot(monkeypatch, calls)
    dest = ds.download_suite("boom", tmp_path / "boom")
    import json

    marker = json.loads((dest / ds._MARKER).read_text())
    assert marker["n_files"] == 1 and len(marker["listing_sha256"]) == 64
    assert ds.recorded_revision(dest) == ds.DATASETS["boom"].revision
    (dest / ARROW).write_bytes(b"truncated-and-different")   # size changed
    assert "partial" in ds.recorded_revision(dest)
    ds.ensure_datasets(["boom"], tmp_path, download=True)   # resumes the pull
    assert len(calls) == 2


def test_verify_stamps_a_legacy_marker_without_downloading(tmp_path: Path, monkeypatch):
    import json

    import pytest

    calls: list = []
    _stub_snapshot(monkeypatch, calls)
    dest = tmp_path / "boom"
    (dest / "cfg").mkdir(parents=True)
    (dest / ARROW).write_bytes(b"x")
    spec = ds.DATASETS["boom"]
    (dest / ds._MARKER).write_text(json.dumps(
        {"repo": spec.hf_repo, "revision": spec.revision, "patterns": None}))
    r = ds.verify_suite("boom", dest)
    assert calls == [] and r["n_files"] == 1
    assert "listing_sha256" in json.loads((dest / ds._MARKER).read_text())
    monkeypatch.setattr(ds, "_expected_files", lambda spec, pats: [ARROW, "missing.arrow"])
    with pytest.raises(RuntimeError, match="1 of 2 files missing"):
        ds.verify_suite("boom", dest)
