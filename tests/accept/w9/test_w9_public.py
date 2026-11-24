"""W9 (plan rev6.18 A2, A6, B3; F3): the audit re-derives the public draw
from the auditor's copy of each index and the settlement block hash,
re-stated in `cascade/audit/lane_d.py` (never imported from the producer,
invariant 12), and checks the pinned revision and the index / file-list
hashes. Both are ALERT class (`ld_draw`, `ld_public_index`): logged and
alerted, lane D keeps being paid. Contract: README.md "Public sources
(rev6.18 F3)".
"""

from __future__ import annotations

import copy
import gzip
import inspect
import json

from . import helpers as h
from .f3_fixture import pub


def _findings(report: dict, **kw):
    from cascade.audit.lane_d import lane_d_findings

    assert "public_indexes" in inspect.signature(lane_d_findings).parameters, (
        "F3: lane_d_findings / check_lane_d(..., public_indexes=) re-derive the public draw")
    h._check_lane_d()
    return lane_d_findings(copy.deepcopy(report), h.w7a.lane_d_cfg(),
                           signer_hotkey=h.signer().ss58_address,
                           max_channels=32, **kw)


def _codes(f) -> dict:
    return {c: d for c, d in f.findings}


def _resigned(report: dict, fn) -> dict:
    out = copy.deepcopy(report)
    fn(out["public"]["sources"][0])
    return h.resign(out)


def test_an_honest_public_block_passes_with_and_without_the_indexes(tmp_path):
    from cascade.audit.lane_d import check_lane_d

    run = pub.settle_pub(tmp_path)
    rep = run["s1"].report
    f = _findings(rep, public_indexes=run["indexes"])
    assert not f.findings, f.findings
    assert not _findings(rep).findings
    assert "public_indexes" in inspect.signature(check_lane_d).parameters
    res = check_lane_d(copy.deepcopy(rep), h.w7a.lane_d_cfg(),
                       signer_hotkey=h.signer().ss58_address, max_channels=32,
                       public_indexes=run["indexes"])
    assert res.status == "PASS", res.detail


def test_a_window_the_index_does_not_give_is_an_ld_draw_alert(tmp_path):
    from cascade.audit.lane_d import refusal_class

    run = pub.settle_pub(tmp_path)

    def move(s):
        w = s["windows"][0]
        if s["multi_window"]:
            w["cut"] -= 1
        else:
            w["cut"] = 128 if w["cut"] != 128 else 129
        w["window_id"] = w["window_id"].rsplit(":", 2)[0] + f":{w['cut']}:h64"
        s["windows"].sort(key=lambda x: x["window_id"])

    rep = _resigned(run["s1"].report, move)
    codes = _codes(_findings(rep, public_indexes=run["indexes"]))
    assert set(codes) == {"ld_draw"}, codes
    assert "re-derived draw" in codes["ld_draw"]
    assert refusal_class("ld_draw") == "alert"
    assert not _findings(rep).findings          # without the index: not re-derivable


def test_a_quota_or_seed_not_the_rule_is_an_ld_draw_alert(tmp_path):
    run = pub.settle_pub(tmp_path)

    def seed(s):
        s["seed"] = s["seed"].replace("lane_d_public", "lane_d_publik")

    def n(s):
        s["n_windows"] = 4000

    for fn in (seed, n):
        rep = _resigned(run["s1"].report, fn)
        f = _findings(rep)
        assert set(_codes(f)) == {"ld_draw"}, f.findings


def test_index_and_revision_mismatches_are_ld_public_index_alerts(tmp_path):
    from cascade.audit.lane_d import refusal_class

    run = pub.settle_pub(tmp_path)
    assert refusal_class("ld_public_index") == "alert"

    def rev(s):
        s["revision"] = "f" * 40

    def ih(s):
        s["index_sha256"] = "0" * 64

    def fh(s):
        s["files_sha256"] = "1" * 64

    rep = _resigned(run["s1"].report, rev)
    assert "ld_public_index" in _codes(_findings(rep))          # pinned, no index needed
    for fn in (ih, fh):
        rep = _resigned(run["s1"].report, fn)
        codes = _codes(_findings(rep, public_indexes=run["indexes"]))
        assert "ld_public_index" in codes, codes
        assert not {c for c in codes if refusal_class(c) == "burn"}


def test_a_tampered_auditor_index_is_named(tmp_path):
    run = pub.settle_pub(tmp_path)
    idx = run["indexes"]["gift_eval_pretrain"]
    raw = gzip.decompress((idx / "index.jsonl.gz").read_bytes())
    lines = raw.splitlines(keepends=True)
    row = json.loads(lines[0])
    row["length"] += 1
    lines[0] = json.dumps(row, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    (idx / "index.jsonl.gz").write_bytes(gzip.compress(b"".join(lines)))
    codes = _codes(_findings(run["s1"].report, public_indexes=run["indexes"]))
    assert "ld_public_index" in codes and "does not verify" in codes["ld_public_index"]


def test_the_audit_redraw_matches_the_report_on_every_source(tmp_path):
    from cascade.audit.lane_d import public_redraw

    run = pub.settle_pub(tmp_path)
    for s in run["s1"].report["public"]["sources"]:
        raw = gzip.decompress((run["indexes"][s["source"]] / "index.jsonl.gz").read_bytes())
        rows = [json.loads(x) for x in raw.splitlines()]
        got = public_redraw(rows, source=s["source"], block_hash=run["block_hash"], horizon=64)
        assert got["windows"] == s["windows"]
        assert (got["quota"], got["capacity"]) == (s["quota"], s["capacity"])
