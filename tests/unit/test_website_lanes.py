"""The dashboard's lane views: the lane-A king ladder panel, the lane-C page,
lane-B horizon slots / multi-target wins and the emission split with lane C
(rendered in node against fixtures built by the REAL writers), plus a static
check that the fields the page reads are fields the writers emit.
"""

from __future__ import annotations

import re
from dataclasses import replace
from types import SimpleNamespace

import pytest

from cascade.shared.config import LaneBConfig
from cascade.shared.config.lane_c import parse_lane_c
from cascade.shared.v2_status import (
    build_v2_status,
    king_check_fields,
    ladder_fields,
    lane_a_view,
    lane_c_state_from_receipt,
    lane_c_view,
    v2_index_fields,
)

from .test_lane_observability import CHECK, EMISSION, RECEIPT_BODY
from .test_website_render import INDEX_CALLS, NODE, WEB, _bucket, _html, _render

HTML = (WEB / "index.html").read_text()


def _with_lanes(fx: dict, *, lane_c: bool) -> dict:
    """Stamp the DEC-CA-0055 records (and lane C) on the fixture's receipt,
    index row and status doc via the real writers."""
    key = "receipts/5FcCsoB2R5M55RdrNrvPQv7xmLJRFzs7R7C73UUSxgUCyfsw/latest.json"
    body = fx[key]
    body.update(emission=EMISSION, king_check=CHECK)
    if lane_c:
        body.update(lane_c=RECEIPT_BODY["lane_c"], lane_c_pay=RECEIPT_BODY["lane_c_pay"])
    row = fx["receipts/index.json"]["rounds"][0]
    row.update(v2_index_fields(SimpleNamespace(
        weights=tuple(body["weights"]), reward_uids=tuple(body["reward_uids"]),
        lane_b=tuple(body.get("lane_b") or ()) or None, lane_b_pay=body.get("lane_b_pay"),
        lane_c=body.get("lane_c"), lane_c_pay=body.get("lane_c_pay"),
        emission=EMISSION, king_check=CHECK)))
    v2 = fx["status/v2.json"]
    v2["lane_a"] = lane_a_view(body)
    if lane_c:
        lc = replace(parse_lane_c({}), from_block=1000)
        v2["lane_c"] = lane_c_view(lc, lane_c_state_from_receipt(body), block=50_000, trainer={
            "queue": {"queued": 2}, "targets": {"mkt:returns": {"queued": 2, "open": 1,
                                                                   "finished": 0, "slots": 2}},
            "open": [], "finished": 0, "legs_started": 1, "settlements": [], "outcomes": []})
        fx["status/lane_c.json"] = {"lane": "c", "block": 50_000, "queue": {"queued": 2},
                                    "targets": {"mkt:returns": {"queued": 2, "open": 1,
                                                                  "finished": 0, "slots": 2}},
                                    "open": [{"hotkey": "hk-open", "uid": 30,
                                              "target": "mkt:returns", "ref": "r",
                                              "base_era": 6, "started_block": 49_000,
                                              "gpu": "H100"}],
                                    "finished": 0, "legs_started": 1, "settlements": [],
                                    "outcomes": []}
    return fx


CALLS = INDEX_CALLS + ["setPage('lanec')", "renderLaneC()", "setPage('main')",
                       "renderLaneALadder()"]


@pytest.mark.skipif(NODE is None, reason="node not installed")
def test_lane_c_page_ladder_and_split(tmp_path):
    fx = _with_lanes(_bucket(v2_from=1000, block=50_000, lane_b=True), lane_c=True)
    els = _render("index.html", fx, "testnet.casdesub.net", CALLS, tmp_path)
    lad = _html(els, "la-ladder")
    assert not els["la-ladder-wrap"]["hidden"]
    for want in ("60%", "pass", "5Former", "until block 50000", "none", "LCB 0.0123"):
        assert want in lad, want
    status = _html(els, "lc-status")
    assert "● live" in status
    assert "marketcast.json" in _html(els, "lc-title")
    targets = _html(els, "lc-targets")
    for want in ("mkt:vol", "uid 7", "100% of the slot", "Pending confirmation", "confirm",
                 "mkt:returns", "✓ confirmed"):
        assert want in targets, want
    q = _html(els, "lc-queue")
    assert "H100" in q and "queued <b>2</b>" in q
    split = _html(els, "v2-split")
    assert "lane C" in split                     # lane-C pay is never shown as burn


@pytest.mark.skipif(NODE is None, reason="node not installed")
def test_lane_c_off_reads_not_live(tmp_path):
    fx = _bucket(v2_from=1000, block=50_000, lane_b=True)
    els = _render("index.html", fx, "testnet.casdesub.net", CALLS, tmp_path)
    assert "not live" in _html(els, "lc-status") and "not live" in _html(els, "lc-targets")
    assert els["la-ladder-wrap"]["hidden"]       # no DEC-CA-0055 record ⇒ no panel
    assert "lane C" not in _html(els, "v2-split")
    els = _render("index.html", {}, "dashboard.casdesub.net", CALLS, tmp_path)
    assert "not published" in _html(els, "lc-status")


@pytest.mark.skipif(NODE is None, reason="node not installed")
def test_lane_b_horizon_slots_and_multi_target_wins(tmp_path, monkeypatch):
    from tests.unit import test_website_render as R

    def _cfg(from_block):
        lb = LaneBConfig(from_block=from_block, domains=R.DOMAINS, win_ttl_eras=28,
                         horizons=(64,), max_domains_per_leg=2)
        return SimpleNamespace(lane_b=lb, subnet=SimpleNamespace(netuid=91),
                               training=SimpleNamespace(v2_from_block=from_block))

    monkeypatch.setattr(R, "_cfg", _cfg)
    fx = R._bucket(v2_from=1000, block=50_000, lane_b=True)
    row = fx["receipts/index.json"]["rounds"][0]
    row["lane_b_verdicts"][0]["targets"] = ["energy", "horizon:64"]
    els = _render("index.html", fx, "testnet.casdesub.net", INDEX_CALLS, tmp_path)
    doms = _html(els, "lb-domains")
    assert "horizon 64" in doms and ">horizon<" in doms
    assert "Multi-target leg" in doms and "energy + horizon 64" in doms
    assert "+ 1 horizon slots" in _html(els, "lb-dom-when")


# ── the page reads only what the writers emit ────────────────────────────────


def _fn(name: str) -> str:
    m = re.search(r"function " + re.escape(name) + r"\(.*?\n(?=function |// ──|\(function)",
                  HTML, re.S)
    assert m, name
    return m.group(0)


def _reads(body: str, var: str) -> set[str]:
    return set(re.findall(r"(?<![\w.])" + re.escape(var) + r"\.([a-z_][a-z0-9_]*)", body))


def test_page_reads_only_written_fields(cfg):
    ladder = set(ladder_fields(EMISSION))
    check = set(king_check_fields(CHECK))
    lad_reads = _reads(_fn("renderLaneALadder"), "lad")
    assert lad_reads <= ladder, lad_reads - ladder
    kc_reads = _reads(_fn("renderLaneALadder"), "kc")
    assert kc_reads <= check, kc_reads - check

    lc = replace(parse_lane_c({}), from_block=1000)
    st = lane_c_state_from_receipt(RECEIPT_BODY)
    view = lane_c_view(lc, st, block=50_000, trainer={"queue": {}, "targets": {}})
    c_reads = _reads(_fn("renderLcStatus"), "c") | _reads(_fn("renderLcTargets"), "c")
    assert c_reads <= set(view), c_reads - set(view)
    for var, src in (("p", view["payout"]), ("v", view["verdict"]), ("legs", view["legs"])):
        r = _reads(_fn("renderLcStatus"), var)
        assert r <= set(src), (var, r - set(src))
    per = view["per_target"]["mkt:vol"]
    pt_reads = _reads(_fn("renderLcTargets"), "pt")
    assert pt_reads <= set(per), pt_reads - set(per)
    h_reads = _reads(_fn("renderLcTargets"), "h")
    assert h_reads <= set(per["holder"]), h_reads - set(per["holder"])
    pend = view["per_target"]["mkt:returns"]["pending"][0]
    assert _reads(_fn("renderLcTargets"), "x") <= set(pend)
    w_reads = _reads(_fn("renderLcTargets"), "w")
    win = RECEIPT_BODY["lane_c_pay"]["wins"][0]
    assert w_reads <= set(win), w_reads - set(win)

    rows = v2_index_fields(SimpleNamespace(
        weights=(1.0,), reward_uids=(0,), lane_b=None, lane_b_pay=None,
        lane_c=tuple(RECEIPT_BODY["lane_c"]), lane_c_pay=None, emission=None, king_check=None))
    v_reads = _reads(_fn("lcVerdicts"), "v")
    assert v_reads <= set(rows["lane_c_verdicts"][0]), v_reads - set(rows["lane_c_verdicts"][0])

    doc = build_v2_status(cfg, block=100, as_of="t", lane_a=lane_a_view(RECEIPT_BODY))
    assert {"ladder", "king_check"} <= set(doc["lane_a"])
    assert {"targets", "horizon_targets", "per_domain"} <= set(doc["lane_b"])
    # emission split keys the strip reads
    assert {"lane_a", "lane_b", "lane_c", "burn"} >= _reads(_fn("splitBar"), "s")
