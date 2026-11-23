"""W1 acceptance: the lane D measurement answers its questions from the forge
mirror ("measured from forge mirror"), next to the snapshot-diff bound.

CPU only, on a tiny synthetic parquet mirror (helpers.build_mirror); nothing
reads a bucket. With LANED_EVIDENCE set (``accept W1 --evidence
evidence/w1.json``) the evidence file itself is verified.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from . import helpers as h

pytest.importorskip("pandas")
pytest.importorskip("pyarrow")
pytest.importorskip("yaml")


def _forge_mode():
    mod = h.load_script()
    for name in ("load_forge", "forge_measure"):
        assert hasattr(mod, name), f"laned_w1_measure.py has no forge-mirror mode ({name})"
    return mod


def _run(mod, tmp_path, name, *extra):
    forge = tmp_path / "forge"
    if not forge.exists():
        h.build_mirror(forge)
    out = tmp_path / name
    rc = mod.main(["--forge-dir", str(forge), "--out", str(out),
                   "--min-windows", "10", "--min-clusters", "1", *extra])
    assert rc == 0
    return json.loads(out.read_text(encoding="utf-8"))


def _check_schema(doc: dict) -> None:
    assert doc["schema"] == "cascade-laned-w1/1"
    for k in ("command_line", "host", "timestamp_utc"):
        assert doc[k]
    assert doc["labels"] == {"pairs": h.BOUND_LABEL, "forge_mirror": h.FORGE_LABEL}
    fm = doc["forge_mirror"]
    assert fm["label"] == h.FORGE_LABEL
    assert set(fm["per_freq"]) == set(h.FREQ_BUCKETS)
    for cell in fm["per_freq"].values():
        assert {"sources", "series"} <= set(cell)
    assert set(fm["windows"]["summary"]) == set(h.CLOCKS)
    rungs = fm["long_history"]["rungs"]
    assert [r["context"] for r in rungs] == list(h.CONTEXTS)
    for r in rungs:
        assert r["need_points"] == r["context"] + doc["params"]["horizon"]
        assert {"series", "sources", "gate_met"} <= set(r)
    gate = fm["long_history"]["gate"]
    assert gate["min_sources"] == h.GATE_MIN_SOURCES


def test_unit_suite_passes():
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(h.UNIT_TESTS)],
        cwd=str(h.REPO), capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-2000:]


def test_forge_report_schema_and_defaults(tmp_path):
    mod = _forge_mode()
    doc = _run(mod, tmp_path, "w1.json")
    _check_schema(doc)
    p = doc["forge_params"]
    assert p["embargo_hours"] == h.EMBARGO_HOURS
    assert p["min_fresh_share"] == h.MIN_FRESH_SHARE
    assert p["max_windows"] == h.MAX_WINDOWS
    assert p["forward_margin_hours"] == h.FORWARD_MARGIN_HOURS
    assert p["eval_delay_hours"] == h.EVAL_DELAY_HOURS
    from cascade.shared.config.lane_d import LaneDConfig
    assert doc["forge_mirror"]["long_history"]["gate"]["min_long_series"] == \
        LaneDConfig().min_long_series


def test_mirror_inventory_latency_and_long_history(tmp_path):
    mod = _forge_mode()
    fm = _run(mod, tmp_path, "w1.json")["forge_mirror"]
    pf = fm["per_freq"]
    assert (pf["1min"]["sources"], pf["1min"]["series"]) == (1, 1)
    assert (pf["hourly"]["sources"], pf["hourly"]["series"]) == (1, 2)
    assert pf["slower"]["reader_emits"] is False
    assert fm["sources"]["dis"]["skipped"] == "disabled"
    # first-seen proxy = the 23:30 partition write: mean wait ~11.5 h
    lat = pf["1min"]["first_seen_latency_hours"]["median_over_sources_upper"]
    assert 11.0 < lat < 12.5
    r = {x["context"]: x for x in fm["long_history"]["rungs"]}
    assert (r[4096]["series"], r[4096]["sources"]) == (3, 2)
    assert r[8192]["series"] == 0
    assert r[4096]["gate_met"] is False          # < 20 sources


def test_fresh_counted_from_commit_time_and_eval_delay(tmp_path):
    mod = _forge_mode()
    doc = _run(mod, tmp_path, "w1.json")
    for row in doc["forge_mirror"]["windows"]["sweep"]:
        assert row["boundary_b"] == row["commit_time"]                 # margin 0
        assert row["embargo_from"] == h.iso(h.parse(row["commit_time"]) - 48 * 3600)
        assert row["judging_from"] == h.iso(h.parse(row["commit_time"]) + 6 * 3600)
    assert doc["forge_mirror"]["windows"]["summary"]["T+24h"]["fresh"]["median"] > 0
    late = _run(mod, tmp_path, "w1_delay9.json", "--eval-delay-hours", "9")
    for row in late["forge_mirror"]["windows"]["sweep"]:
        assert isinstance(row["clocks"]["T+6h"], str) and row["clocks"]["T+6h"].startswith(
            "not run")
        assert isinstance(row["clocks"]["T+9h"], dict)
    m6 = _run(mod, tmp_path, "w1_m6.json", "--forward-margin-hours", "6")
    assert (m6["forge_mirror"]["windows"]["summary"]["T+24h"]["fresh"]["median"]
            < doc["forge_mirror"]["windows"]["summary"]["T+24h"]["fresh"]["median"])


def test_set_rule_holds_on_every_row(tmp_path):
    mod = _forge_mode()
    doc = _run(mod, tmp_path, "w1.json")
    s = h.MIN_FRESH_SHARE
    seen = 0
    for row in doc["forge_mirror"]["windows"]["sweep"]:
        for c in row["clocks"].values():
            if isinstance(c, str):
                continue
            seen += 1
            st, tot = c["set"], c["total"]
            assert st["fresh_kept"] <= tot["fresh"] and st["embargo_kept"] <= tot["embargo"]
            assert st["windows"] == st["fresh_kept"] + st["embargo_kept"] <= h.MAX_WINDOWS
            # fresh first; embargo never above the minimum fresh share
            assert st["embargo_kept"] * s <= st["fresh_kept"] * (1 - s) + 1e-9
            if st["fresh_kept"] == 0:
                assert st["windows"] == 0 and st["met"] is False   # embargo can't start a set
            if st["met"]:
                assert st["windows"] >= 10 and st["sources"] >= 1
                assert st["fresh_share"] >= s
    assert seen


def test_read_only_and_deterministic(tmp_path):
    mod = _forge_mode()
    a = _run(mod, tmp_path, "a.json")
    forge = tmp_path / "forge"
    before = sorted((p.relative_to(forge).as_posix(), p.stat().st_size, p.stat().st_mtime)
                    for p in forge.rglob("*") if p.is_file())
    b = _run(mod, tmp_path, "b.json")
    after = sorted((p.relative_to(forge).as_posix(), p.stat().st_size, p.stat().st_mtime)
                   for p in forge.rglob("*") if p.is_file())
    assert before == after
    assert a["forge_mirror"] == b["forge_mirror"]


def test_workflow_measures_forge_mirror_after_sync():
    import yaml

    wf = yaml.safe_load(h.WORKFLOW.read_text(encoding="utf-8"))
    steps = wf["jobs"]["measure"]["steps"]
    ids = [s.get("id") for s in steps]
    assert "forge_sync" in ids, "the forge sync step needs an id"
    sync_at = ids.index("forge_sync")
    guarded = [i for i, s in enumerate(steps)
               if str(s.get("if", "")).strip() == "steps.forge_sync.outcome == 'success'"]
    assert guarded and min(guarded) > sync_at
    run = steps[min(guarded)]["run"]
    assert "laned_w1_measure.py" in run and "--forge-dir" in run
    assert "--embargo-hours 48" in run
    text = h.WORKFLOW.read_text(encoding="utf-8")
    assert "--extra pool-forge" in text
    for s in steps:                    # read-only: every s3 sync is bucket -> runner
        r = str(s.get("run", ""))
        for line in r.splitlines():
            if "aws s3 sync" in line:
                assert line.split("aws s3 sync", 1)[1].split()[0].strip('"').startswith("s3://")
            assert "aws s3 cp" not in line and "put_object" not in line


def test_evidence_file():
    path = os.environ.get("LANED_EVIDENCE")
    if not path:
        pytest.skip("no LANED_EVIDENCE (accept W1 --evidence evidence/w1.json)")
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    _check_schema(doc)
    assert isinstance(doc["forge_mirror"]["windows"]["sweep"], list)
    assert doc["forge_mirror"]["windows"]["sweep"], "evidence must sweep >= 1 commit time"
