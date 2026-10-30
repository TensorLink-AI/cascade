"""A benchmark data dir's marker is only trusted while its recorded download
listing still matches (miner report 2026-10-30: a rate-limited BOOM pull
wrote its completion marker with most datasets missing)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from cascade.validator.bench_identity import MARKER, current_revisions, listing_digest


def _data_dir(root: Path, *, listing: bool) -> Path:
    for suite in ("gift-eval", "boom", "time"):
        d = root / suite
        (d / "cfg").mkdir(parents=True)
        (d / "cfg" / "data.arrow").write_bytes(b"abc")
        marker = {"repo": "r", "revision": "a" * 40, "patterns": None}
        if listing:
            n, digest = listing_digest(d)
            marker.update(n_files=n, listing_sha256=digest)
        (d / MARKER).write_text(json.dumps(marker))
    return root


def test_identity_rejects_a_dir_that_lost_files_after_its_marker(tmp_path):
    root = _data_dir(tmp_path, listing=True)
    assert current_revisions(root, require_fev=False) == {s: "a" * 40 for s in ("gift-eval", "boom", "time")}
    (root / "boom" / "cfg" / "data.arrow").unlink()
    assert current_revisions(root, require_fev=False) is None


def test_identity_keeps_accepting_legacy_markers(tmp_path):
    root = _data_dir(tmp_path, listing=False)
    assert current_revisions(root, require_fev=False) is not None


def test_listing_digest_matches_the_sidecar_implementation(tmp_path):
    path = Path(__file__).resolve().parents[2] / "benchmarks/cascade_benchmark/datasets.py"
    spec = importlib.util.spec_from_file_location("_sidecar_datasets", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod                       # dataclasses resolve it there
    spec.loader.exec_module(mod)
    d = tmp_path / "x"
    (d / "a" / "b").mkdir(parents=True)
    (d / "a" / "b" / "f.arrow").write_bytes(b"12345")
    (d / "top.json").write_text("{}")
    (d / ".cache" / "huggingface").mkdir(parents=True)
    (d / ".cache" / "huggingface" / "lock").write_text("")
    (d / MARKER).write_text("{}")
    assert mod.listing_digest(d) == listing_digest(d)
    assert listing_digest(d)[0] == 2
