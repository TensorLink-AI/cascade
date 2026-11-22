"""W-S3: the era's sentinel board (plan §16 rule 4, §1.3b "Progress board").

``cascade.shared.sentinel_board.build_sentinel_board`` reads the W3b3
``lane-d-sentinel/1`` records of the era's kings from the public store and
shows the era's sentinel suite and each king's published numbers (with the
previous KING_D's and the house's), labelled non-gating. A record that is
absent, unreadable or not a valid record is shown as such, never as a number.
README.md states the contract."""

from __future__ import annotations

import json
import math

import pytest

from . import helpers as h


def _floats(obj) -> list[float]:
    """Every float anywhere in ``obj``."""
    if isinstance(obj, bool):
        return []
    if isinstance(obj, float):
        return [obj]
    if isinstance(obj, dict):
        return [x for v in obj.values() for x in _floats(v)]
    if isinstance(obj, (list, tuple)):
        return [x for v in obj for x in _floats(v)]
    return []


def _assert_no_numbers(row: dict) -> None:
    assert row["models"] is None, f"{row['state']} row shows models: {row['models']!r}"
    assert _floats(row) == [], f"{row['state']} row shows numbers: {_floats(row)}"
    assert isinstance(row["reason"], str) and row["reason"], (
        f"{row['state']} row has no reason: {row['reason']!r}")


@pytest.fixture()
def era():
    return h.three_kings_store()


def test_the_board_shows_the_era_suite_and_every_king_in_order(era):
    store, _ = era
    b = h.board(h.cfg(), kings=[h.K1, h.K2, h.K3], store=store)
    assert isinstance(b, dict)
    assert set(b) >= h.BOARD_KEYS, f"board keys {sorted(b)} lack {sorted(h.BOARD_KEYS - set(b))}"
    assert b["schema"] == h.BOARD_SCHEMA
    assert b["era"] == {"block": h.ERA_BLOCK, "block_hash": h.era_hash("armed")}
    assert b["suite"] == h.draw(h.era_hash("armed"))
    assert list(b["sentinel_suites"]) == list(h.SUITES)
    assert [r["king_digest"] for r in b["kings"]] == [h.K1, h.K2, h.K3]
    for r in b["kings"]:
        assert set(r) >= h.ROW_KEYS, f"row keys {sorted(r)} lack {sorted(h.ROW_KEYS - set(r))}"
        assert r["key"] == h.record_key(h.ERA_BLOCK, r["king_digest"])
        assert r["state"] in h.STATES


def test_the_board_order_is_the_given_order(era):
    store, _ = era
    b = h.board(h.cfg(), kings=[h.K3, h.K1, h.K2], store=store)
    assert [r["king_digest"] for r in b["kings"]] == [h.K3, h.K1, h.K2]


def test_ok_rows_carry_the_published_numbers_exactly(era):
    store, res = era
    b = h.board(h.cfg(), kings=[h.K1, h.K2, h.K3], store=store)
    for king, row in h.rows_by_king(b).items():
        rec = res[king]["record"]
        assert rec["status"] == "ok", "fixture: the W3b3 record is not ok"
        assert row["state"] == "ok"
        assert row["reason"] is None
        assert row["suite"] == rec["suite"] == b["suite"]
        assert row["boundary"] == rec["boundary"]
        assert row["gate_suite"] == rec["gate_suite"]
        assert row["session"] == rec["session"]
        # the published numbers, not re-derived, rounded or re-scored
        assert row["models"] == rec["models"]
        assert set(row["models"]) == set(h.ROLES)
        for role in h.ROLES:
            for m in ("crps", "mase"):
                assert row["models"][role][m] == rec["models"][role][m]
                assert isinstance(row["models"][role][m], float)


def test_the_house_and_previous_king_numbers_are_on_the_board(era):
    store, res = era
    rows = h.rows_by_king(h.board(h.cfg(), kings=[h.K1, h.K2, h.K3], store=store))
    assert rows[h.K1]["models"]["house"]["digest"] == h.HOUSE
    assert rows[h.K1]["models"]["previous_king"]["digest"] == h.HOUSE
    assert rows[h.K2]["models"]["previous_king"]["digest"] == h.K1
    assert rows[h.K3]["models"]["previous_king"]["digest"] == h.K2
    for king in (h.K1, h.K2, h.K3):
        house = rows[king]["models"]["house"]
        assert house["status"] == "ok"
        assert house["crps"] == res[king]["record"]["models"]["house"]["crps"]


def test_a_regressed_king_is_shown_with_its_numbers(era):
    """K3's sentinel numbers are worse than K2's and the house's: the board
    shows them as published; it never hides, flags or ranks a king away."""
    store, _ = era
    rows = h.rows_by_king(h.board(h.cfg(), kings=[h.K1, h.K2, h.K3], store=store))
    k3 = rows[h.K3]["models"]
    assert k3["king"]["crps"] > k3["previous_king"]["crps"]
    assert k3["king"]["crps"] > k3["house"]["crps"]
    assert rows[h.K3]["state"] == "ok"


def test_the_board_is_labelled_non_gating(era):
    store, _ = era
    b = h.board(h.cfg(), kings=[h.K1, h.K2, h.K3], store=store)
    assert b["gating"] is False
    assert isinstance(b["label"], str) and "non-gating" in b["label"].lower(), (
        f"the board label {b['label']!r} does not say it is non-gating")


def test_no_kings_still_shows_the_era_suite():
    store = h.MemoryStore()
    b = h.board(h.cfg(), kings=[], store=store)
    assert b["kings"] == []
    assert b["suite"] == h.draw(h.era_hash("armed"))
    assert store.gets == [] and store.puts == []


def test_the_draw_follows_the_configured_sentinel_suites():
    suites = ("boom", "time", "fev")
    hh = h.hash_drawing("fev", suites)
    store = h.MemoryStore()
    h.publish_record(store, king=h.K1, block_hash=hh, suites=suites)
    b = h.board(h.cfg(suites=suites), era_block_hash=hh, kings=[h.K1], store=store)
    assert list(b["sentinel_suites"]) == list(suites)
    assert b["suite"] == "fev"
    assert h.rows_by_king(b)[h.K1]["state"] == "ok"


@pytest.mark.parametrize("suite", h.SUITES)
def test_the_board_suite_is_the_era_draw(suite):
    hh = h.hash_drawing(suite)
    store = h.MemoryStore()
    h.publish_record(store, king=h.K1, block_hash=hh)
    b = h.board(h.cfg(), era_block_hash=hh, kings=[h.K1], store=store)
    assert b["suite"] == suite
    assert h.rows_by_king(b)[h.K1]["suite"] == suite


# ── records that are not ok ──────────────────────────────────────────────────


def test_a_missing_record_is_shown_as_missing():
    store = h.MemoryStore()
    h.publish_record(store, king=h.K1)
    rows = h.rows_by_king(h.board(h.cfg(), kings=[h.K1, h.K2], store=store))
    assert rows[h.K1]["state"] == "ok"
    assert rows[h.K2]["state"] == "missing"
    _assert_no_numbers(rows[h.K2])


def test_a_record_of_another_era_is_not_shown():
    store = h.MemoryStore()
    h.publish_record(store, king=h.K1, era_block=h.ERA_BLOCK + h.ERA)
    rows = h.rows_by_king(h.board(h.cfg(), kings=[h.K1], store=store))
    assert rows[h.K1]["state"] == "missing"
    _assert_no_numbers(rows[h.K1])


def test_an_unreadable_store_is_shown_as_unavailable_and_never_raises():
    from cascade.shared.hippius import StorageError

    store = h.MemoryStore()
    h.publish_record(store, king=h.K1)
    h.publish_record(store, king=h.K2, previous=h.K1)
    store.fail_get[h.record_key(h.ERA_BLOCK, h.K2)] = StorageError("s3_get_failed: 503")
    rows = h.rows_by_king(h.board(h.cfg(), kings=[h.K1, h.K2], store=store))
    assert rows[h.K1]["state"] == "ok"
    assert rows[h.K2]["state"] == "unavailable"
    _assert_no_numbers(rows[h.K2])


def test_a_store_outage_shows_every_row_unavailable():
    from cascade.shared.hippius import StorageError

    store, _ = h.three_kings_store()
    store.fail_get["*"] = StorageError("s3_get_failed: connection reset")
    b = h.board(h.cfg(), kings=[h.K1, h.K2, h.K3], store=store)
    assert [r["state"] for r in b["kings"]] == ["unavailable"] * 3
    for r in b["kings"]:
        _assert_no_numbers(r)
    assert b["suite"] == h.draw(h.era_hash("armed"))


def test_a_fault_record_is_shown_as_a_fault():
    store = h.MemoryStore()
    rec = h.publish_record(store, king=h.K1, nan_role="house")["record"]
    assert rec["status"] == "fault", "fixture: the NaN did not make a fault record"
    row = h.rows_by_king(h.board(h.cfg(), kings=[h.K1], store=store))[h.K1]
    assert row["state"] == "fault"
    assert row["reason"] == rec["reason"]
    assert row["models"] == rec["models"]
    assert row["models"]["house"]["status"] == "fault"
    assert row["models"]["house"]["crps"] is None and row["models"]["house"]["mase"] is None
    assert row["models"]["king"]["crps"] == rec["models"]["king"]["crps"]
    assert all(math.isfinite(x) for x in _floats(row))


def test_an_error_record_is_shown_as_an_error():
    store = h.MemoryStore()
    rec = h.publish_record(store, king=h.K1,
                           host_error=ConnectionError("bench pod unreachable"))["record"]
    assert rec["status"] == "error", "fixture: the host error did not make an error record"
    row = h.rows_by_king(h.board(h.cfg(), kings=[h.K1], store=store))[h.K1]
    assert row["state"] == "error"
    assert row["reason"] == rec["reason"]
    assert row["models"] == rec["models"]
    for role in h.ROLES:
        assert row["models"][role]["status"] == "error"
        assert row["models"][role]["crps"] is None and row["models"][role]["mase"] is None
    assert _floats(row) == []


def _tamper(kind: str, rec: dict) -> str:
    r = json.loads(json.dumps(rec))
    if kind == "not_json":
        return "{this is not json"
    if kind == "a_json_array":
        return json.dumps([r])
    if kind == "nan_number":
        r["models"]["king"]["crps"] = float("nan")
        return json.dumps(r, sort_keys=True)
    if kind == "infinite_number":
        r["models"]["house"]["mase"] = float("inf")
        return json.dumps(r, sort_keys=True)
    if kind == "wrong_schema":
        r["schema"] = "lane-d-sentinel/2"
    elif kind == "another_kings_record":
        r["king_digest"] = h.K2
        r["models"]["king"]["digest"] = h.K2
    elif kind == "another_era":
        r["era"]["block"] = h.ERA_BLOCK + h.ERA
    elif kind == "another_era_hash":
        r["era"]["block_hash"] = h.era_hash("other")
    elif kind == "not_the_era_draw":
        r["suite"] = next(s for s in h.SUITES if s != rec["suite"])
    elif kind == "claims_to_gate":
        r["gating"] = True
    elif kind == "unknown_status":
        r["status"] = "great"
    elif kind == "ok_model_without_a_number":
        r["models"]["king"]["crps"] = None
    elif kind == "ok_model_with_a_string_number":
        r["models"]["previous_king"]["mase"] = "0.5"
    elif kind == "missing_models":
        del r["models"]
    else:
        raise AssertionError(f"fixture: unknown tamper {kind!r}")
    return json.dumps(r, sort_keys=True)


TAMPERS = ("not_json", "a_json_array", "nan_number", "infinite_number", "wrong_schema",
           "another_kings_record", "another_era", "another_era_hash", "not_the_era_draw",
           "claims_to_gate", "unknown_status", "ok_model_without_a_number",
           "ok_model_with_a_string_number", "missing_models")


@pytest.mark.parametrize("kind", TAMPERS)
def test_an_invalid_record_is_shown_as_invalid_never_as_numbers(kind):
    store = h.MemoryStore()
    res = h.publish_record(store, king=h.K1)
    store.objects[res["key"]] = _tamper(kind, res["record"])
    row = h.rows_by_king(h.board(h.cfg(), kings=[h.K1], store=store))[h.K1]
    assert row["state"] == "invalid", f"{kind}: state {row['state']!r}"
    _assert_no_numbers(row)


# ── what the board reads and writes ──────────────────────────────────────────


def test_the_board_reads_only_the_kings_keys_and_writes_nothing(era):
    store, _ = era
    store.puts.clear()
    store.gets.clear()
    before = dict(store.objects)
    h.board(h.cfg(), kings=[h.K1, h.K3], store=store)
    assert set(store.gets) == {h.record_key(h.ERA_BLOCK, h.K1), h.record_key(h.ERA_BLOCK, h.K3)}
    assert store.puts == []
    assert store.objects == before


@pytest.mark.parametrize("kings, err", [
    ([h.K1, h.K1], ValueError),
    ([h.K1, ""], ValueError),
    ([h.K1, None], (TypeError, ValueError)),
    (h.K1, TypeError),
], ids=["duplicate", "empty_digest", "none_digest", "bare_string"])
def test_malformed_kings_are_refused_before_any_read(kings, err):
    from cascade.shared import sentinel_board

    store, _ = h.three_kings_store()
    store.gets.clear()
    with pytest.raises(err):
        sentinel_board.build_sentinel_board(h.cfg(), era_block=h.ERA_BLOCK,
                                            era_block_hash=h.era_hash("armed"),
                                            kings=kings, store=store)
    assert store.gets == []


@pytest.mark.parametrize("bad", ["", "0x", "0x" + "zz" * 32, "ab" * 32, None],
                         ids=["empty", "bare_0x", "non_hex", "no_prefix", "none"])
def test_a_malformed_era_hash_is_refused_before_any_read(bad):
    from cascade.shared import sentinel_board

    store, _ = h.three_kings_store()
    store.gets.clear()
    with pytest.raises((ValueError, TypeError)):
        sentinel_board.build_sentinel_board(h.cfg(), era_block=h.ERA_BLOCK,
                                            era_block_hash=bad, kings=[h.K1], store=store)
    assert store.gets == []


# ── invariant 1 / 10 ─────────────────────────────────────────────────────────


def test_board_unchanged_when_unarmed():
    """W-S1 unarmed (``sentinel_from_block = 0``, or an era before it): no
    board and no store read, whatever records exist."""
    store, _ = h.three_kings_store()
    store.gets.clear()
    store.puts.clear()
    assert h.board(h.cfg(armed=False), kings=[h.K1, h.K2, h.K3], store=store) is None
    assert h.board(h.cfg(armed=True), era_block=h.UNARMED_ERA_BLOCK,
                   kings=[h.K1, h.K2, h.K3], store=store) is None
    assert store.gets == [] and store.puts == []


def test_board_armed_nonempty_lane_d_effect():
    store, _ = h.three_kings_store()
    b = h.board(h.cfg(armed=True), kings=[h.K1, h.K2, h.K3], store=store)
    assert isinstance(b, dict) and len(b["kings"]) == 3
    assert all(r["state"] == "ok" and r["models"] for r in b["kings"])
    # armed exactly from the arming block
    store2, _ = h.three_kings_store(era_block=h.ARM_BLOCK)
    b2 = h.board(h.cfg(armed=True), era_block=h.ARM_BLOCK, kings=[h.K1], store=store2)
    assert isinstance(b2, dict) and b2["kings"][0]["state"] == "ok"
