"""W8b: the production lane D step calls `settle_lane_d` at each armed
boundary with the W8b row's inputs, then runs the W3b3 sentinel sweep,
publishes the signed report, its W7c receipt record and status/lane_d.json.
Contract: README.md."""

from __future__ import annotations

import json

import pytest

from . import helpers as h

BOUNDARIES = (h.B1, h.B2, h.B3)


def _settle_all(rig, boundaries=BOUNDARIES):
    out = {}
    for b in boundaries:
        out[b] = rig.step(b)
    return out


def _compare_to_oracle(tmp_path, *, sentinel: bool):
    cfg = h.chain_cfg(tmp_path / "v", sentinel=sentinel)
    fx = h.fixture(tmp_path)
    rig = h.Rig(tmp_path / "rig", cfg, fx)
    oracle = h.Oracle(tmp_path, cfg, fx)
    for b in BOUNDARIES:
        rig.step(b)
        want = oracle.settle(b)
        got = rig.published_report(b)
        assert h.body_sha256(got) == h.body_sha256(want.report), (
            f"boundary {b}: the published report is not settle_lane_d's with the row's inputs")
        assert rig.published_receipt(b) == want.receipt
        assert rig.state_path.read_bytes() == oracle.state_path.read_bytes()
        assert b in rig.factory.reader.blocks(), f"the boundary {b}'s block hash was never read"
    assert rig.bench.boundaries and h.B2 in rig.bench.boundaries
    assert rig.inputs.calls[:3] == list(BOUNDARIES)
    return rig, oracle


def test_step_settles_each_armed_boundary_with_the_row_inputs(tmp_path):
    rig, oracle = _compare_to_oracle(tmp_path, sentinel=False)
    s2 = rig.published_report(h.B2)
    assert s2["crowned"] == rig.fx["x"]
    assert s2["gate"]["block_hash"] == h.block_hash(h.B2)
    assert s2["gate"]["sentinel"] is None
    crowning = [v for v in s2["verdicts"] if v["digest"] == rig.fx["x"]]
    # rev6.18 A7 / A8 (F4): no A10 block; the crowning verdict carries its
    # per-half point gains (correction of the W3b5 lowfreq assertion).
    assert crowning and "lowfreq" not in crowning[0] \
        and (crowning[0].get("halves") or {}).get("live", {}).get("gain", 0) > 0, \
        "the crowning verdict carries a positive live half (and no A10 block)"


def test_step_passes_the_era_sentinel_drawn_from_the_era_start_hash(tmp_path):
    rig, oracle = _compare_to_oracle(tmp_path, sentinel=True)
    suite = h.era_sentinel(rig.cfg, h.B2)
    assert suite is not None
    s2 = rig.published_report(h.B2)
    assert "gate" in s2, "B2's first-verdict crown (rev6.11 (b)) runs the gate"
    assert s2["gate"]["sentinel"] == suite
    assert h.ERA_START in rig.factory.reader.blocks(), "the era-start hash was never read"


def test_new_king_gets_the_w3b3_sentinel_sweep_published(tmp_path):
    from cascade.validator.lane_d_sentinel import sentinel_key

    rig, oracle = _compare_to_oracle(tmp_path, sentinel=True)
    key = sentinel_key(h.ERA_START, rig.fx["x"])
    assert key in rig.store.objects, "no W3b3 sentinel record for the new king"
    assert rig.store.objects[key] == oracle.store.objects[key]
    rec = json.loads(rig.store.objects[key])
    assert rec["status"] == "ok" and rec["gating"] is False
    assert sum(k.startswith("lane-d/sentinel/") for k in rig.store.objects) == 1, \
        "only the new king (B2) is swept; B1 and B3 crown nobody"


def test_no_sentinel_work_while_ws1_is_unarmed(tmp_path):
    rig, oracle = _compare_to_oracle(tmp_path, sentinel=False)
    assert not [k for k in rig.store.objects if k.startswith("lane-d/sentinel/")]
    assert rig.host.calls == oracle.host.calls, "bench calls beyond the gate sweep"


def test_status_lane_d_json_is_published_with_the_ws3_board(tmp_path):
    from cascade.validator.lane_d_status import LANE_D_STATUS_KEY

    rig, oracle = _compare_to_oracle(tmp_path, sentinel=True)
    assert rig.store.objects[LANE_D_STATUS_KEY] == oracle.status_text()
    doc = json.loads(rig.store.objects[LANE_D_STATUS_KEY])
    assert doc["boundary"] == h.B3
    assert doc["sentinel"] is not None, "W-S1 armed: the status embeds the W-S3 board"
    assert [r["king_digest"] for r in doc["sentinel"]["kings"]] == [rig.fx["x"]]
    assert doc["sentinel"]["kings"][0]["state"] == "ok"


def test_report_receipt_and_latest_are_published_in_order(tmp_path):
    w = h.wiring()
    cfg = h.chain_cfg(tmp_path / "v")
    rig = h.Rig(tmp_path / "rig", cfg, h.fixture(tmp_path))
    got = _settle_all(rig)
    for b in BOUNDARIES:
        rk, ck = w.report_key(b), w.receipt_key(b)
        assert rk.startswith("lane-d/") and ck.startswith("lane-d/") and rk != ck
        assert rk.rsplit("/", 1)[-1] == f"lane_d_report_{b}.json"
        assert rig.store.objects[rk] == rig.report_file(b).read_text(encoding="ascii"), \
            "the published report must be the persisted signed report, byte for byte"
        assert rig.published_receipt(b) == got[b].receipt
        order = rig.store.puts
        assert order.index(rk) < order.index(ck), "a receipt never precedes its report"
    latest = rig.store.objects[w.RECEIPT_LATEST_KEY]
    assert latest == rig.store.objects[w.receipt_key(h.B3)]
    last_latest = max(i for i, k in enumerate(rig.store.puts) if k == w.RECEIPT_LATEST_KEY)
    assert rig.store.puts.index(w.receipt_key(h.B3)) < last_latest, \
        "the latest pointer moves only after the receipt it names is published"
    from cascade.validator.lane_d_state import verify_receipt, verify_receipt_chain

    receipts = [rig.published_receipt(b) for b in BOUNDARIES]
    verify_receipt_chain(receipts)
    for b, rec in zip(BOUNDARIES, receipts, strict=True):
        verify_receipt(rec, report=rig.published_report(b))


def test_court_for_is_the_published_courts_and_does_no_io(tmp_path):
    cfg = h.chain_cfg(tmp_path / "v")
    rig = h.Rig(tmp_path / "rig", cfg, h.fixture(tmp_path))
    assert rig.step.court_for(h.B1) is None
    rig.step(h.B1)
    assert rig.step.court_for(h.B1) == [None], "the house holds KING_D at B1: its slot burns"
    assert rig.step.court_for(h.B1 + 450) == [None]
    rig.step(h.B2)
    assert rig.step.court_for(h.B2) == [h.CHALLENGER_UID]
    assert rig.step.court_for(h.B2 + 899) == [h.CHALLENGER_UID]
    assert rig.step.court_for(h.B3) == [h.CHALLENGER_UID], \
        "the standing court is the latest published settlement's until the next one lands"
    assert rig.step.court_for(h.B1) == [None]
    assert rig.step.court_for(h.ARM_BLOCK) is None
    court = [s["uid"] for s in rig.published_report(h.B2)["pay"]["court"]]
    assert court == [h.CHALLENGER_UID]
    gets, puts, reads = len(rig.store.gets), len(rig.store.puts), len(rig.factory.reader.calls)
    rig.store.fail_get = lambda _k: True
    rig.factory.reader.fail = set(range(h.ARM_BLOCK, h.B3 + 1))
    assert rig.step.court_for(h.B2 + 5) == [h.CHALLENGER_UID]
    assert (len(rig.store.gets), len(rig.store.puts), len(rig.factory.reader.calls)) == \
        (gets, puts, reads), "court_for read the store or the chain"


def test_bench_hold_when_no_bench_session_is_configured(tmp_path):
    cfg = h.chain_cfg(tmp_path / "v")
    rig = h.Rig(tmp_path / "rig", cfg, h.fixture(tmp_path), bench=False)
    rig.step(h.B1)
    rig.step(h.B2)
    rep = rig.published_report(h.B2)
    assert rep["crowned"] is None, "an armed crown without a bench session must not crown"
    x = [e for e in rep["entries"] if e["digest"] == rig.fx["x"]][0]
    assert x["status"] not in ("crowned", "lost", "closed")
    assert rig.step.court_for(h.B2) == [None]


def test_sentinel_sweep_failure_never_gates_the_crown(tmp_path):
    from cascade.validator.lane_d_sentinel import sentinel_key

    cfg = h.chain_cfg(tmp_path / "v", sentinel=True)
    rig = h.Rig(tmp_path / "rig", cfg, h.fixture(tmp_path))
    rig.host.error_suites = {h.era_sentinel(cfg, h.B2)}
    rig.step(h.B1)
    rig.step(h.B2)
    assert rig.published_report(h.B2)["crowned"] == rig.fx["x"]
    assert rig.step.court_for(h.B2) == [h.CHALLENGER_UID]
    rec = json.loads(rig.store.objects[sentinel_key(h.ERA_START, rig.fx["x"])])
    assert rec["status"] in ("fault", "error"), rec["status"]


def test_two_validators_publish_the_same_settlements(tmp_path):
    """Determinism: two validators with the same inputs bind the same report
    bodies, receipts, state bytes and status (signatures are randomized)."""
    from cascade.validator.lane_d_status import LANE_D_STATUS_KEY

    cfg = h.chain_cfg(tmp_path / "v", sentinel=True)
    fx = h.fixture(tmp_path)
    a = h.Rig(tmp_path / "a", cfg, fx)
    b = h.Rig(tmp_path / "b", cfg, fx)
    _settle_all(a)
    _settle_all(b)
    for bd in BOUNDARIES:
        assert a.published_receipt(bd) == b.published_receipt(bd)
        assert h.body_sha256(a.published_report(bd)) == h.body_sha256(b.published_report(bd))
    assert a.state_path.read_bytes() == b.state_path.read_bytes()
    assert a.store.objects[LANE_D_STATUS_KEY] == b.store.objects[LANE_D_STATUS_KEY]


def test_satellite_replay_matches_the_published_chain(tmp_path):
    from cascade.audit.lane_d_chain import MATCH, replay_chain

    w = h.wiring()
    cfg = h.chain_cfg(tmp_path / "v")
    rig = h.Rig(tmp_path / "rig", cfg, h.fixture(tmp_path))
    _settle_all(rig)
    mirror = tmp_path / "satellite"
    mirror.mkdir()
    for b in BOUNDARIES:
        key = w.report_key(b)
        (mirror / key.rsplit("/", 1)[-1]).write_text(rig.store.objects[key], encoding="ascii")
    receipts = [json.loads(rig.store.objects[w.receipt_key(b)]) for b in BOUNDARIES]
    res = replay_chain(mirror, receipts, cfg.lane_d, signer_ss58=h.signer().ss58_address)
    assert [r.status for r in res] == [MATCH] * 3, res


# ── invariant 1: unarmed builds nothing ─────────────────────────────────────


def test_inv1_no_lane_d_step_unchanged_when_unarmed(tmp_path):
    cfg = h.shipped_cfg(tmp_path / "v")
    assert int(cfg.lane_d.from_block) == 0
    store, factory = h.Store(), h.ReaderFactory()
    state_dir = tmp_path / "lane_d"
    step = h.wiring().build_lane_d_step(cfg, state_dir=state_dir, signer=h.signer(),
                                        chain_factory=factory, store=store,
                                        inputs=lambda b: pytest.fail("inputs read unarmed"),
                                        bench=lambda b: pytest.fail("bench built unarmed"))
    assert step is None
    assert factory.threads == [] and store.puts == [] and store.gets == []
    assert h.dir_bytes(state_dir) == {}


def test_inv1_no_lane_d_step_armed_nonempty_lane_d_effect(tmp_path):
    cfg = h.chain_cfg(tmp_path / "v")
    rig = h.Rig(tmp_path / "rig", cfg, h.fixture(tmp_path))
    assert rig.step is not None and callable(rig.step)
    rig.step(h.B1)
    assert h.wiring().report_key(h.B1) in rig.store.objects
    assert rig.state_path.is_file()


def test_inactive_boundary_touches_nothing(tmp_path):
    """Armed from ARM_BLOCK: the settlement AT from_block belongs to the era
    before, so the step does nothing there."""
    cfg = h.chain_cfg(tmp_path / "v")
    rig = h.Rig(tmp_path / "rig", cfg, h.fixture(tmp_path))
    assert not cfg.lane_d.active(h.ARM_BLOCK)
    assert rig.step(h.ARM_BLOCK) is None
    assert rig.factory.threads == [] and rig.store.puts == [] and rig.host.calls == []
    assert rig.inputs.calls == []
    assert h.dir_bytes(rig.state_dir) == {}
