"""WB11 (RV6 B8): the intake's lookup fits the 7 GB production box.

``load_artefact`` returns the whole artefact as Python dicts. Measured on main
at 50 000 windows that is ~1 150 B of heap per window (57.6 MB peak, plus
~84 B/window of ``.json.gz`` on disk), so the review's "tens of millions" of
suite windows would OOM every service on the box. The intake therefore loads
the pinned artefact through ``load_lookup(path, *, expect_sha256)``, a
memory-bounded lookup structure that ``match_series`` / ``is_contaminated``
accept, with a per-window budget (README "Memory budget"). Measured on a
synthetic 50 000-window artefact, built and loaded in two fresh processes
(tracemalloc heap peak and ru_maxrss growth). Contract: README.md."""

from __future__ import annotations

import json

import pytest

from .helpers import (
    _BUILD_CHILD,
    _LOAD_CHILD,
    FILE_BYTES_PER_WINDOW,
    LOOKUP_PEAK_BYTES_PER_WINDOW,
    RSS_BYTES_PER_WINDOW,
    RSS_FIXED_BYTES,
    SYNTH_WINDOWS,
    build,
    file_budget,
    fixture_suites,
    lookup_budget,
    match,
    mod,
    run_child,
)


def _synthetic(tmp: str) -> tuple[dict, str]:
    path = f"{tmp}/hashset-v1.json.gz"
    return run_child(_BUILD_CHILD, path), path


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    return _synthetic(str(tmp_path_factory.mktemp("wb11-scale")))


def test_synthetic_artefact_has_the_stated_scale(synthetic):
    built, _ = synthetic
    assert built["n_exact"] == SYNTH_WINDOWS and built["n_normalised"] == SYNTH_WINDOWS, built


def test_artefact_file_is_within_its_per_window_budget(synthetic):
    import os

    built, path = synthetic
    size = os.path.getsize(path)
    assert size <= file_budget(SYNTH_WINDOWS), (
        f"hashset-v1.json.gz is {size} B = {size / SYNTH_WINDOWS:.1f} B/window at "
        f"{SYNTH_WINDOWS} windows; budget {FILE_BYTES_PER_WINDOW} B/window")


def test_lookup_loads_within_its_per_window_memory_budget(synthetic):
    built, path = synthetic
    got = run_child(_LOAD_CHILD, path, built["sha256"])
    assert "missing" not in got, (
        "cascade.eval.contamination has no load_lookup(path, *, expect_sha256): the "
        "intake would load the whole artefact as dicts (~1150 B/window measured on main)")
    peak, rss = got["peak"], got["rss_delta"]
    assert peak <= lookup_budget(SYNTH_WINDOWS), (
        f"load_lookup heap peak {peak} B = {peak / SYNTH_WINDOWS:.1f} B/window; budget "
        f"{LOOKUP_PEAK_BYTES_PER_WINDOW} B/window + 1 MiB")
    assert rss <= RSS_BYTES_PER_WINDOW * SYNTH_WINDOWS + RSS_FIXED_BYTES, (
        f"load_lookup RSS growth {rss} B = {rss / SYNTH_WINDOWS:.1f} B/window; budget "
        f"{RSS_BYTES_PER_WINDOW} B/window + 16 MiB")
    assert got["version"] == 1 and got["window"] == 32, got
    # a 40-point fragment of a suite item at offset 100: 9 windows, both sets
    r = got["match"]
    assert r["exact_hits"] == 9 and r["normalised_hits"] == 9, r
    assert r["first"] == {"kind": "exact", "suite": "gift", "window_id": "gift/synth/0@100",
                          "at": 0}, r
    assert got["contaminated"] is True


def _load_lookup():
    m = mod()
    load = getattr(m, "load_lookup", None)
    assert callable(load), "cascade.eval.contamination has no load_lookup (RV6 B8)"
    return m, load


@pytest.mark.parametrize("suffix", [".json", ".json.gz"])
def test_lookup_matches_exactly_like_the_full_artefact(tmp_path, suffix):
    """Same ``match_series`` / ``is_contaminated`` answers as the dict artefact,
    and the header keys scan_repo reads."""
    m, load = _load_lookup()
    a = build()
    p = tmp_path / f"hashset-v1{suffix}"
    sha = m.write_artefact(a, p)
    lk = load(p, expect_sha256=sha)
    for k in ("schema", "version", "window", "stride"):
        assert lk[k] == a[k], k
    assert lk["suites"] == a["suites"]
    s = fixture_suites()
    probes = [v for spec in s.values() for v in spec["series"].values()]
    probes += [[2.0 * x - 9.0 for x in probes[0]], [round(3.7 * x + 100, 3) for x in probes[1]],
               [0.0] * 40, list(range(40))]
    for vals in probes:
        assert m.match_series(lk, vals) == match(a, vals), vals[:3]
        assert m.is_contaminated(lk, vals) == m.is_contaminated(a, vals), vals[:3]


def test_lookup_refuses_a_wrong_pin_a_tampered_file_and_schema_1(tmp_path):
    m, load = _load_lookup()
    a = build()
    p = tmp_path / "hashset-v1.json"
    sha = m.write_artefact(a, p)
    with pytest.raises(ValueError):
        load(p, expect_sha256="sha256:" + "0123456789abcdef" * 4)
    art = json.loads(p.read_bytes())
    h = sorted(art["exact"])[0]
    del art["exact"][h]
    t = tmp_path / "hashset-v2.json"
    t.write_bytes(m.artefact_bytes(art))
    with pytest.raises(ValueError):
        load(t, expect_sha256=sha)
    v1 = dict(a, schema="cascade_contamination_hashset/1")
    p1 = tmp_path / "hashset-v3.json"
    sha1 = m.write_artefact(v1, p1)
    with pytest.raises(ValueError):
        load(p1, expect_sha256=sha1)


def test_cli_verify_records_scale_file_size_and_load_peak(tmp_path):
    """The owner's verify step reports what the pin evidence records:
    ``n_exact``, ``n_normalised``, ``file_bytes`` and ``lookup_peak_bytes``."""
    import os

    from .helpers import cli_json, run_cli

    m = mod()
    a = build()
    p = tmp_path / "hashset-v1.json.gz"
    sha = m.write_artefact(a, p)
    proc = run_cli("verify", "--artefact", str(p), "--sha256", sha)
    rep = cli_json(proc)
    assert proc.returncode == 0 and rep["ok"] is True, rep
    assert rep.get("n_exact") == len(a["exact"]), rep
    assert rep.get("n_normalised") == len(a["normalised"]), rep
    assert rep.get("file_bytes") == os.path.getsize(p), rep
    peak = rep.get("lookup_peak_bytes")
    assert isinstance(peak, int) and 0 < peak <= lookup_budget(len(a["exact"])), rep
