"""Lane-C slot-holder registrations recorded at read time (owner 2026-11-24):
eviction never needs chain history.

* the validator judges from its boundary snapshot (read live at B − 1), else
  — catching up late — the MAIN validator's signed receipt record, and reads
  the chain only without either;
* the receipt records the registrations it judged (``lane_c_registrations``);
* the audit / satellite row replays that record without a chain client: an
  honest one passes, a forged one fails (and is lane C's row only)."""
from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from cascade.audit.checks import check_lane_c_registrations
from cascade.shared.hippius import receipt_round_key
from cascade.shared.lane_c import TargetWin
from cascade.shared.lane_c_registrations import (
    RegistrationSnapshots,
    recorded_uid,
    registration_record,
    registration_record_problem,
)
from cascade.shared.receipt import dump_receipt, load_receipt, sign_receipt

from . import test_marketcast_validator as T

TARGET = "mkt:returns"
B = 9 * 900
ACTIVE = SimpleNamespace(lane_c=SimpleNamespace(active=lambda b: True))


def _post(**slots):
    """state_post lane-C slots ``{target: [hotkey, uid, win_block, round, …]}``."""
    return {"lane_c": {"slots": {t: [hk, uid, wb, "r", 0.1, 0.05, 10]
                                 for t, (hk, uid, wb) in slots.items()}, "pending": []}}


def _receipt(rec, post):
    return SimpleNamespace(epoch_start_block=B, lane_c_registrations=rec, state_post=post)


# ── the record and its replay (no chain) ─────────────────────────────────────


def test_honest_record_replays_without_any_chain_history():
    rec = registration_record(B - 1, [
        {"target": TARGET, "hotkey": "kept", "uid": 4},
        {"target": "mkt:vol", "hotkey": "gone", "uid": None}])
    post = _post(**{TARGET: ("kept", 4, B - 900)})        # "gone" was evicted
    row = check_lane_c_registrations(_receipt(rec, post), ACTIVE)
    assert row.status == "PASS", row.detail
    assert recorded_uid(rec, "gone", B - 1) == (True, None)
    assert recorded_uid(rec, "kept", B - 1) == (True, 4)
    assert recorded_uid(rec, "kept", B) == (False, None)  # another block: not recorded


@pytest.mark.parametrize("forge,why", [
    # a row flipped after the digest was taken
    (lambda rec: {**rec, "rows": [{**rec["rows"][0], "uid": 9}]}, "sha256"),
    # a deregistered holder that still holds its slot
    (lambda rec: registration_record(B - 1, [{"target": TARGET, "hotkey": "kept",
                                              "uid": None}]), "still holds"),
    # a holder judged but omitted from the record
    (lambda rec: registration_record(B - 1, [{"target": TARGET, "hotkey": "other",
                                              "uid": 2}]), "no row"),
    # recorded at another block than the boundary − 1
    (lambda rec: registration_record(B - 300, rec["rows"]), "boundary"),
    # rows out of canonical order / duplicated
    (lambda rec: {**rec, "rows": rec["rows"] * 2}, "canonical"),
])
def test_forged_record_fails_the_replay(forge, why):
    rec = registration_record(B - 1, [{"target": TARGET, "hotkey": "kept", "uid": 4}])
    post = _post(**{TARGET: ("kept", 4, B - 900)})
    row = check_lane_c_registrations(_receipt(forge(rec), post), ACTIVE)
    assert row.status == "FAIL" and why in row.detail, row.detail


def test_a_slot_won_at_this_settlement_needs_no_row():
    rec = registration_record(B - 1, [{"target": TARGET, "hotkey": "old", "uid": None}])
    post = _post(**{TARGET: ("new", 3, B)})               # re-won this settlement
    assert registration_record_problem(rec, boundary=B, state_post=post) is None


def test_record_outside_lane_c_fails_and_is_lane_c_scoped():
    from cascade.validator.satellite import _lane_of_row

    rec = registration_record(B - 1, [{"target": TARGET, "hotkey": "x", "uid": 1}])
    inactive = SimpleNamespace(lane_c=SimpleNamespace(active=lambda b: False))
    assert check_lane_c_registrations(_receipt(rec, _post()), inactive).status == "FAIL"
    assert _lane_of_row("lane_c_registrations") == "c"     # containment: lane C only


def test_receipt_round_trip_keeps_and_signs_the_record(cfg):
    from cascade.shared.receipt import verify_receipt_signature

    from .receipt_fixture import make_rejected_receipt

    plain = make_rejected_receipt(cfg)
    assert "lane_c_registrations" not in plain.canonical_body().decode()  # bytes unchanged
    rec = registration_record(B - 1, [{"target": TARGET, "hotkey": "kept", "uid": 4}])
    signed = sign_receipt(replace(plain, lane_c_registrations=rec), T.MAIN_KP)
    back = load_receipt(dump_receipt(signed))
    assert back.lane_c_registrations == rec
    assert verify_receipt_signature(back, T.MAIN_KP.ss58_address)
    forged = replace(back, lane_c_registrations=registration_record(
        B - 1, [{"target": TARGET, "hotkey": "kept", "uid": None}]))
    assert not verify_receipt_signature(forged, T.MAIN_KP.ss58_address)


def test_snapshots_persist_and_keep_the_newest(tmp_path):
    s = RegistrationSnapshots(tmp_path / "snaps.json")
    for b in range(40):
        s.put(b, {"hk": b})
    again = RegistrationSnapshots(tmp_path / "snaps.json")
    assert again.get(39) == {"hk": 39} and again.get(0) is None


# ── the validator: snapshot → main's signed record → chain ───────────────────

SETTLE = T.BLOCK + 2 * T.EB          # round 1003's boundary in _three_rounds


def _holder(r, reg_calls):
    def reg(hotkey, block):
        reg_calls.append((hotkey, block))
        return 5
    r.registration_at_block_fn = reg
    r._lane_c().slots[TARGET] = TargetWin(
        TARGET, "former", 5, "old", T.BLOCK, base_pointer=T.MC.base_pointer, gain_lcb=0.99,
        tier=0.05, full_eras=10, trained_pointer=T._ptr("former"))


def test_boundary_snapshot_decides_without_a_chain_read(cfg, tmp_path):
    calls: list = []

    def hook(r):
        _holder(r, calls)
        r._lane_c_reg_snapshots().put(SETTLE - 1, {"alice": 7})   # "former" deregistered

    f = T._three_rounds(cfg, tmp_path, before_confirm=hook)
    assert calls == []                                            # no chain read at all
    assert f.r._lane_c().slots[TARGET].hotkey == "alice"         # former evicted
    rec = f.r._lane_c_receipt_kw([])["lane_c_registrations"]
    assert rec == registration_record(SETTLE - 1, [
        {"target": TARGET, "hotkey": "former", "uid": None}])


def _main_receipt(cfg, store, rec, *, signer):
    from .receipt_fixture import make_rejected_receipt

    rc = replace(make_rejected_receipt(cfg), round_id="1003", lane_c_registrations=rec)
    store.objs[receipt_round_key("1003", T.MAIN_KP.ss58_address)] = dump_receipt(
        sign_receipt(rc, signer))


def test_late_node_uses_the_main_validators_signed_record(cfg, tmp_path):
    calls: list = []
    rec = registration_record(SETTLE - 1, [{"target": TARGET, "hotkey": "former",
                                            "uid": None}])

    def hook(r):
        _holder(r, calls)
        _main_receipt(cfg, r._lane_c_store, rec, signer=T.MAIN_KP)

    f = T._three_rounds(cfg, tmp_path, before_confirm=hook)
    assert calls == []                                            # the record, not history
    assert f.r._lane_c().slots[TARGET].hotkey == "alice"
    assert f.r._lane_c_receipt_kw([])["lane_c_registrations"] == rec


def test_a_record_not_signed_by_the_main_is_ignored_for_the_chain(cfg, tmp_path):
    calls: list = []
    rec = registration_record(SETTLE - 1, [{"target": TARGET, "hotkey": "former",
                                            "uid": None}])

    def hook(r):
        _holder(r, calls)
        _main_receipt(cfg, r._lane_c_store, rec, signer=T.TRAINER_KP)  # wrong key

    f = T._three_rounds(cfg, tmp_path, before_confirm=hook)
    assert calls == [("former", SETTLE - 1)]                     # fell back to the chain
    assert f.r._lane_c().slots[TARGET].hotkey == "former"        # registered: kept
    assert f.r._lane_c_receipt_kw([])["lane_c_registrations"]["rows"] == [
        {"target": TARGET, "hotkey": "former", "uid": 5}]


def test_registration_tick_snapshots_once_while_fresh(cfg, tmp_path):
    f = T._three_rounds(cfg, tmp_path)
    r = f.r
    grid = r._lane_b_epoch_blocks(SETTLE)
    b = (SETTLE + 3 * grid) // grid * grid
    reads: list = []
    client = SimpleNamespace(current_block=lambda: b + 30,
                             hotkey_uids_at_block=lambda blk: reads.append(blk) or {"a": 1})
    r._lane_c_registration_tick(client)
    r._lane_c_registration_tick(client)
    assert reads == [b - 1] and r._lane_c_reg_snapshots().get(b - 1) == {"a": 1}
    late = SimpleNamespace(current_block=lambda: b + grid - 1,
                           hotkey_uids_at_block=lambda blk: reads.append(blk) or {})
    r._lane_c_registration_tick(late)                             # stale: not read
    assert reads == [b - 1]
