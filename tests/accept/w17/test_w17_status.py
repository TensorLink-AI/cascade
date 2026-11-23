"""W17: `status/lane_d.json` (queue, pending progress, king, board).
Contract: README.md. Every state and report is a real `settle_lane_d` run."""

from __future__ import annotations

import copy
import json

import pytest

from . import helpers as h


@pytest.fixture(scope="module")
def rich(tmp_path_factory):
    return h.rich_b1(tmp_path_factory.mktemp("w17-status-rich"))


@pytest.fixture(scope="module")
def crowned(tmp_path_factory):
    return h.crowned(tmp_path_factory.mktemp("w17-status-crowned"))


@pytest.fixture(scope="module")
def held(tmp_path_factory):
    return h.held(tmp_path_factory.mktemp("w17-status-held"))


@pytest.fixture(scope="module")
def progress(tmp_path_factory):
    return h.progress_b1(tmp_path_factory.mktemp("w17-status-progress"))


BLOCK_B1 = h.B1 + 17


def _doc(sc, *, cfg=None, block=BLOCK_B1, **kw):
    return h.status(sc["cfg"] if cfg is None else cfg, state=sc["state"], report=sc["report"],
                    block=block, **kw)


# ── the document ────────────────────────────────────────────────────────────


def test_module_constants():
    m = h.mod()
    assert m.LANE_D_STATUS_KEY == h.STATUS_KEY
    assert m.LEDGER_SCHEMA == h.LEDGER_SCHEMA
    assert isinstance(m.STATUS_SCHEMA, str) and m.STATUS_SCHEMA


def test_doc_is_schema_versioned_with_the_pinned_keys(rich):
    doc = _doc(rich)
    assert isinstance(doc, dict)
    assert set(doc) >= h.DOC_KEYS, f"missing: {sorted(h.DOC_KEYS - set(doc))}"
    assert doc["schema"] == h.mod().STATUS_SCHEMA
    assert doc["lane"] == "open"
    assert doc["block"] == BLOCK_B1
    assert doc["boundary"] == h.B1 == rich["state"]["boundary"]
    assert doc["era"] == h.ARM_ERA
    assert doc["floors"] == {"min_windows": 24, "min_clusters": 4}


def test_reason_enum_is_the_codes_enum():
    from cascade.validator.lane_d_state import ENTRY_REASONS

    enum = tuple(h.mod().REASON_ENUM)
    assert enum == tuple(ENTRY_REASONS)
    for code in ("coldkey_cap_settlement", "bench_hold", "pending_floor"):
        assert code in enum
    # rev6.11 (b): no confirmation duel, so no confirm_pending reason.
    assert "confirm_pending" not in enum and "lost_confirmation" not in enum


def test_doc_publishes_the_reason_enum(rich):
    from cascade.validator.lane_d_state import ENTRY_REASONS

    doc = _doc(rich)
    assert doc["reason_enum"] == list(ENTRY_REASONS)
    for row in doc["queue"]:
        assert row["reason"] in doc["reason_enum"]


def test_doc_is_strict_json_and_round_trips(rich, crowned, held):
    for sc, block in ((rich, BLOCK_B1), (crowned, h.B2_NEXT_ERA + 5), (held, h.B2 + 5)):
        doc = _doc(sc, block=block)
        text = json.dumps(doc, allow_nan=False)
        assert h.strict_loads(text) == doc


def test_doc_has_no_wall_clock_field(rich, crowned):
    for sc, block in ((rich, BLOCK_B1), (crowned, h.B2_NEXT_ERA + 5)):
        doc = _doc(sc, block=block)
        assert h.all_keys(doc).isdisjoint(h.WALL_CLOCK_KEYS)


# ── the queue: open entries, FIFO, the per-entry reason enum ─────────────────


def test_queue_is_the_open_entries_fifo_by_reveal(rich):
    fx, state = rich["fx"], rich["state"]
    doc = _doc(rich)
    open_ = sorted((dg for dg, r in state["entries"].items()
                    if r["status"] in h.OPEN_STATE_STATUSES),
                   key=lambda dg: (state["entries"][dg]["reveal_ts"], dg))
    assert [r["digest"] for r in doc["queue"]] == open_
    assert open_ == [fx.x["digest"], fx.y["digest"], fx.z["digest"]]
    closed = {fx.l["digest"], fx.e["digest"], fx.d["digest"]}
    assert closed.isdisjoint(r["digest"] for r in doc["queue"])


def test_queue_rows_carry_the_state_fields(rich):
    state = rich["state"]
    doc = _doc(rich)
    assert doc["queue"], "the rich fixture has three open entries"
    for row in doc["queue"]:
        assert set(row) >= h.QUEUE_ROW_KEYS, f"missing: {sorted(h.QUEUE_ROW_KEYS - set(row))}"
        rec = state["entries"][row["digest"]]
        assert row["hotkey"] == rec["hotkey"]
        assert row["coldkey"] == rec["coldkey_at_ready"]
        for k in ("context", "status", "reason", "ready_block", "reveal_block", "reveal_ts"):
            assert row[k] == rec[k], k


def test_coldkey_cap_settlement_entry_is_queued_with_its_reason(rich):
    row = h.row_of(_doc(rich), rich["fx"].y["digest"])
    assert row["status"] == "pending"
    assert row["reason"] == "coldkey_cap_settlement"
    assert row["confirm_stage"] == "first"


def test_held_first_clear_is_queued_at_the_first_stage(rich):
    """rev6.11 (b): a first clear the gate could not decide is held open
    (`hold` / `bench_hold`); there is no confirm stage."""
    row = h.row_of(_doc(rich), rich["fx"].x["digest"])
    assert row["status"] == "hold"
    assert row["reason"] == "bench_hold"
    assert row["confirm_stage"] == "first"


def test_pending_floor_entry_shows_the_first_stage(rich):
    row = h.row_of(_doc(rich), rich["fx"].z["digest"])
    assert (row["status"], row["reason"], row["confirm_stage"]) == ("pending", "pending_floor",
                                                                     "first")


def test_bench_hold_entry_is_queued_with_its_reason(held):
    doc = _doc(held, block=h.B2 + 5)
    row = h.row_of(doc, held["fx"].x["digest"])
    assert row["status"] == "hold"
    assert row["reason"] == "bench_hold"
    assert row["confirm_stage"] == "first"
    assert doc["king"]["house"] is True


def test_reveal_deadline_block_is_reveal_plus_max_reveal_delay(rich):
    doc = _doc(rich)
    days = rich["cfg"].lane_d.max_reveal_delay_days
    for row in doc["queue"]:
        assert row["reveal_deadline_block"] == row["reveal_block"] + days * h.BLOCKS_PER_DAY


def test_reveal_deadline_follows_the_config_not_a_constant(rich):
    cfg = h.chain_cfg(max_judged_per_coldkey_per_settlement=1, max_reveal_delay_days=3)
    doc = _doc(rich, cfg=cfg)
    for row in doc["queue"]:
        assert row["reveal_deadline_block"] == row["reveal_block"] + 3 * h.BLOCKS_PER_DAY


# ── pending progress against the floors (§11.3 Support) ─────────────────────


def test_pending_progress_counts_windows_and_clusters_against_the_floors(progress):
    fx = progress["fx"]
    doc = _doc(progress)
    assert doc["floors"] == {"min_windows": 30, "min_clusters": 4}
    x = h.row_of(doc, fx.x["digest"])
    assert set(x["progress"]) >= h.PROGRESS_KEYS
    assert {k: x["progress"][k] for k in h.PROGRESS_KEYS} == {
        "windows": 24, "clusters": 4, "min_windows": 30, "min_clusters": 4, "frozen": False}
    z = h.row_of(doc, fx.z["digest"])
    assert (z["progress"]["windows"], z["progress"]["clusters"], z["progress"]["frozen"]) == (
        0, 0, False)


def test_progress_is_derived_from_the_state_frozen_set(progress):
    fx, state = progress["fx"], progress["state"]
    fs = state["entries"][fx.x["digest"]]["frozen_set"]
    row = h.row_of(_doc(progress), fx.x["digest"])
    assert row["progress"]["windows"] == len(fs["window_ids"])
    assert row["progress"]["clusters"] == len(set(fs["sources"].values()))


def test_progress_of_a_held_first_clear(held):
    """The held clear's frozen set, judged at B2 over snapshots 1 and 2."""
    fs = held["state"]["entries"][held["fx"].x["digest"]]["frozen_set"]
    assert fs is not None and fs["frozen"] is True
    row = h.row_of(_doc(held, block=h.B2 + 5), held["fx"].x["digest"])
    p = row["progress"]
    assert (p["windows"], p["clusters"], p["frozen"]) == (
        len(fs["window_ids"]), len(set(fs["sources"].values())), True)
    assert p["windows"] >= 24 and p["clusters"] == 4


def test_progress_of_a_held_clear_is_its_judged_set(rich):
    # rev6.11 (b): no confirmation set accumulates; a held first clear keeps
    # the frozen set it was judged on (snapshot 1: 24 windows, 4 sources).
    fs = rich["state"]["entries"][rich["fx"].x["digest"]]["frozen_set"]
    assert fs is not None and fs["frozen"] is True
    p = h.row_of(_doc(rich), rich["fx"].x["digest"])["progress"]
    assert (p["windows"], p["clusters"], p["frozen"]) == (24, 4, True)


# ── the king and the house ──────────────────────────────────────────────────


def test_house_holds_king_d(rich):
    doc = _doc(rich)
    st = rich["state"]
    assert doc["king"] == {"digest": st["house"]["digest"], "hotkey": None, "house": True,
                           "since_boundary": None, "tenure_blocks": None}
    assert doc["house"] == {"digest": st["house"]["digest"],
                            "reveal_ts": st["house"]["reveal_ts"]}
    assert doc["prior_kings"] == []


def test_miner_king_with_tenure(crowned):
    fx = crowned["fx"]
    doc = _doc(crowned, block=h.B2_NEXT_ERA + 300)
    assert set(doc["king"]) >= h.KING_KEYS
    assert {k: doc["king"][k] for k in h.KING_KEYS} == {
        "digest": fx.x["digest"], "hotkey": fx.x["hotkey"], "house": False,
        "since_boundary": h.B2_NEXT_ERA, "tenure_blocks": 300}
    assert doc["house"]["digest"] == crowned["state"]["house"]["digest"]
    assert doc["prior_kings"] == [], "the house is never a prior king"
    assert doc["queue"] == [], "X is crowned and L lost: nothing is open"


# ── the board: the latest signed report ─────────────────────────────────────


def test_board_is_the_latest_reports_judgement(crowned):
    rep = crowned["report"]
    doc = _doc(crowned, block=h.B2_NEXT_ERA + 5)
    board = doc["board"]
    assert set(board) >= h.BOARD_KEYS
    assert board["boundary"] == rep["boundary"] == h.B2_NEXT_ERA
    assert board["crowned"] == crowned["fx"].x["digest"] == rep["crowned"]
    assert board["judged"] == rep["judged"]
    assert board["skipped"] == rep["skipped"]
    assert board["verdicts"] == [{k: v[k] for k in sorted(h.VERDICT_KEYS)}
                                 for v in rep["verdicts"]]
    assert board["gate"] == rep["gate"], "the gate record (suite, results, LCBs) verbatim"


def test_board_shows_losses_and_skips(rich):
    fx, rep = rich["fx"], rich["report"]
    board = _doc(rich)["board"]
    outcomes = {v["digest"]: v["outcome"] for v in board["verdicts"]}
    # rev6.11 (b): X's first clear is the crowning candidate, held by the
    # down bench ("cleared", never crowned); L lost its duel.
    assert outcomes == {fx.x["digest"]: "cleared", fx.l["digest"]: "lost"}
    assert {"digest": fx.y["digest"], "reason": "coldkey_cap_settlement"} in board["skipped"]
    assert board["crowned"] is None
    assert board["gate"] == rep.get("gate")


def test_board_numbers_are_the_report_floats_exactly(crowned):
    rep = crowned["report"]
    board = _doc(crowned, block=h.B2_NEXT_ERA + 5)["board"]
    for got, want in zip(board["verdicts"], rep["verdicts"], strict=True):
        assert got["lcb"] == want["lcb"] and got["gain"] == want["gain"]
        assert type(got["lcb"]) is float


def test_report_must_be_the_one_the_state_binds(crowned):
    with pytest.raises(ValueError):
        h.status(crowned["cfg"], state=crowned["state"], report=crowned["report_b1"],
                 block=h.B2_NEXT_ERA + 5)
    tampered = copy.deepcopy(crowned["report"])
    tampered["verdicts"][0]["lcb"] = tampered["verdicts"][0]["lcb"] + 1.0
    with pytest.raises(ValueError):
        h.status(crowned["cfg"], state=crowned["state"], report=tampered,
                 block=h.B2_NEXT_ERA + 5)


# ── purity and determinism ──────────────────────────────────────────────────


def test_build_is_pure(rich):
    state, report = copy.deepcopy(rich["state"]), copy.deepcopy(rich["report"])
    before_state, before_report = copy.deepcopy(state), copy.deepcopy(report)
    file_before = rich["state_path"].read_bytes()
    store = h.MemoryStore()
    h.mod().build_lane_d_status(rich["cfg"], state=state, report=report, block=BLOCK_B1,
                                era_block=None, era_block_hash=None, store=store)
    assert state == before_state and report == before_report
    assert rich["state_path"].read_bytes() == file_before
    assert store.calls() == 0, "W-S1 unarmed: the build reads and writes nothing"


def test_equal_inputs_give_equal_documents(rich):
    a = _doc(rich)
    shuffled = copy.deepcopy(rich["state"])
    shuffled["entries"] = dict(reversed(list(shuffled["entries"].items())))
    shuffled = dict(reversed(list(shuffled.items())))
    b = h.status(rich["cfg"], state=shuffled, report=rich["report"], block=BLOCK_B1)
    assert h.canonical(a) == h.canonical(b)
    assert h.canonical(_doc(rich)) == h.canonical(a)


# ── the lane closed, and unarmed (invariants 1 and 10) ──────────────────────


def test_lane_closed_after_until_block(rich):
    until = h.ARM_BLOCK + 2 * h.ERA
    cfg = h.chain_cfg(max_judged_per_coldkey_per_settlement=1, until_block=until)
    doc = _doc(rich, cfg=cfg, block=until + 5)
    assert doc is not None
    assert doc["schema"] == h.mod().STATUS_SCHEMA
    assert doc["lane"] == "closed"
    assert _doc(rich, cfg=cfg, block=BLOCK_B1)["lane"] == "open"


def test_inv1_status_unchanged_when_unarmed(rich):
    store = h.MemoryStore()
    m = h.mod()
    unarmed = h.chain_cfg(armed=False)
    assert unarmed.lane_d.from_block == 0
    assert h.status(unarmed, state=rich["state"], report=rich["report"], block=BLOCK_B1,
                    store=store) is None
    # Armed later than the block: not yet active, same as unarmed.
    later = h.chain_cfg(from_block=h.ARM_BLOCK + 10 * h.ERA)
    assert h.status(later, state=rich["state"], report=rich["report"], block=BLOCK_B1,
                    store=store) is None
    assert m.publish_lane_d_status(store, None) is None
    assert store.calls() == 0, "unarmed: no status file, no store call"


def test_inv1_status_armed_nonempty_lane_d_effect(rich):
    store = h.MemoryStore()
    doc = _doc(rich, store=store)
    assert doc is not None and doc["queue"]
    key = h.mod().publish_lane_d_status(store, doc)
    assert key == h.STATUS_KEY
    assert [p[0] for p in store.puts] == [h.STATUS_KEY]
