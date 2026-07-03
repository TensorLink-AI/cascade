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
