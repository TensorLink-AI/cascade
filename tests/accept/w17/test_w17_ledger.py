"""W17: the W18 vault ledger `laned_vault_ledger/1` (plan rev6.2 item (e)):
the current era, every vault entry with its status and `closed_era`, and the
open disputes. It must round-trip through W18's own parser and drive W18's
sweeper (both on main). Contract: README.md."""

from __future__ import annotations

import copy
import json
from datetime import UTC, datetime

import pytest

from . import helpers as h

BLOCK_SAME_ERA = h.B1 + 17                       # era 2700
BLOCK_DUE = h.ARM_BLOCK + 4 * h.ERA + 10         # era 2704: 4 eras after 2700
NOW = datetime(2026, 11, 22, 12, 0, tzinfo=UTC)


@pytest.fixture(scope="module")
def rich(tmp_path_factory):
    return h.rich_b1(tmp_path_factory.mktemp("w17-ledger-rich"))


@pytest.fixture(scope="module")
def crowned(tmp_path_factory):
    return h.crowned(tmp_path_factory.mktemp("w17-ledger-crowned"))


def _vs():
    from cascade.laned import vault_sweeper

    return vault_sweeper


def _led(sc, block, **kw):
    return h.ledger(sc["cfg"], state=sc["state"], reports=sc["reports"], block=block, **kw)


# ── shape ───────────────────────────────────────────────────────────────────


def test_ledger_shape(rich):
    led = _led(rich, BLOCK_SAME_ERA)
    assert set(led) == h.LEDGER_KEYS
    assert led["schema"] == h.LEDGER_SCHEMA
    assert led["era"] == h.ARM_ERA and type(led["era"]) is int
    assert led["disputes"] == []
    for row in led["entries"]:
        assert set(row) == h.LEDGER_ENTRY_KEYS
    digests = [r["digest"] for r in led["entries"]]
    assert digests == sorted(digests)
    assert h.strict_loads(json.dumps(led, allow_nan=False)) == led


def test_ledger_era_is_the_chain_blocks_era(rich):
    assert _led(rich, BLOCK_DUE)["era"] == h.ARM_ERA + 4
    assert _led(rich, h.ARM_BLOCK + 2 * h.ERA)["era"] == h.ARM_ERA + 2


def test_ledger_lists_every_state_entry_and_never_the_house(rich):
    state = rich["state"]
    led = _led(rich, BLOCK_SAME_ERA)
    assert {(r["hotkey"], r["digest"]) for r in led["entries"]} == {
        (rec["hotkey"], dg) for dg, rec in state["entries"].items()}
    assert state["house"]["digest"] not in {r["digest"] for r in led["entries"]}


# ── statuses and closed_era ─────────────────────────────────────────────────


def test_ledger_statuses_and_closed_eras(rich):
    fx = rich["fx"]
    led = _led(rich, BLOCK_SAME_ERA)
    want = {"x": ("pending", None), "y": ("pending", None), "z": ("pending", None),
            "l": ("lost", h.ARM_ERA), "e": ("ready_expired", h.ARM_ERA),
            "d": ("voided", h.ARM_ERA)}
    names = fx.names()
    got = {names[r["digest"]]: (r["status"], r["closed_era"]) for r in led["entries"]}
    assert got == want


def test_king_is_kept_and_closed_era_is_the_closing_settlements_era(crowned):
    fx = crowned["fx"]
    led = _led(crowned, h.B2_NEXT_ERA + 5)
    assert led["era"] == h.ARM_ERA + 1
    x = h.ledger_entry(led, fx.x["digest"])
    assert (x["status"], x["closed_era"]) == ("king", None)
    lo = h.ledger_entry(led, fx.l["digest"])
    # L lost at B1 (era 2700), although the state is at B2_NEXT_ERA (era 2701).
    assert (lo["status"], lo["closed_era"]) == ("lost", h.ARM_ERA)


@pytest.mark.parametrize("reason,status", sorted(h.CLOSED_TO_LEDGER.items()))
def test_closed_reason_maps_to_a_ledger_status(rich, reason, status):
    fx = rich["fx"]
    sc = dict(rich)
    sc["state"] = copy.deepcopy(rich["state"])
    rec = sc["state"]["entries"][fx.d["digest"]]
    assert rec["status"] == "closed"
    rec["reason"] = reason
    row = h.ledger_entry(_led(sc, BLOCK_SAME_ERA), fx.d["digest"])
    assert (row["status"], row["closed_era"]) == (status, h.ARM_ERA)


def test_a_closure_no_report_records_is_refused(crowned):
    with pytest.raises(ValueError):
        h.ledger(crowned["cfg"], state=crowned["state"], reports=[], block=h.B2_NEXT_ERA + 5)


# ── disputes ────────────────────────────────────────────────────────────────


def test_disputes_are_sorted_unique_digests(rich):
    fx = rich["fx"]
    dl, de = fx.l["digest"], fx.e["digest"]
    led = _led(rich, BLOCK_SAME_ERA, disputes=[dl, de, dl])
    assert led["disputes"] == sorted({dl, de})


@pytest.mark.parametrize("bad", ["", "abc", "0x" + "a" * 64, "A" * 64, "g" * 64])
def test_a_malformed_dispute_is_refused(rich, bad):
    with pytest.raises(ValueError):
        _led(rich, BLOCK_SAME_ERA, disputes=[bad])


# ── W18 round trip ──────────────────────────────────────────────────────────


def test_ledger_parses_with_the_w18_parser(rich, crowned):
    vs = _vs()
    for sc, block in ((rich, BLOCK_DUE), (crowned, BLOCK_DUE)):
        led = _led(sc, block)
        parsed = vs.parse_ledger(h.strict_loads(h.canonical(led)))
        assert parsed.era == led["era"]
        assert set(parsed.entries) == {(r["hotkey"], r["digest"]) for r in led["entries"]}
        for r in led["entries"]:
            e = parsed.entries[(r["hotkey"], r["digest"])]
            assert (e.status, e.closed_era) == (r["status"], r["closed_era"])


def _seed_vault(root, sc) -> dict[str, str]:
    """One object per state entry at `<hotkey>/<digest>/model.safetensors`."""
    keys = {}
    for dg, rec in sc["state"]["entries"].items():
        key = f"{rec['hotkey']}/{dg}/model.safetensors"
        p = root / key
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(f"w17 weights {dg}".encode())
        keys[dg] = key
    return keys


def _dry_run(tmp_path, sc, led):
    vs = _vs()
    vroot, aroot = tmp_path / "vault", tmp_path / "archive"
    keys = _seed_vault(vroot, sc)
    log = vs.sweep(vault=vs.FsStore(vroot), archive=vs.FsStore(aroot), ledger=led,
                   policy=vs.policy_from_config(sc["cfg"]), now=NOW, live=False,
                   plan_path=None, log_dir=tmp_path / "logs")
    by = {}
    for a in log["actions"]:
        if a["store"] == "vault":
            by.setdefault(a["key"], []).append((a["op"], a["outcome"], a["reason"]))
    return {dg: sorted(by.get(k, [])) for dg, k in keys.items()}


PLANNED = sorted([("archive", "planned", None), ("delete", "planned", None)])


def test_w18_sweeper_keeps_open_entries_and_retires_due_losers(rich, tmp_path):
    names = rich["fx"].names()
    acts = _dry_run(tmp_path, rich, _led(rich, BLOCK_DUE))
    got = {names[dg]: a for dg, a in acts.items()}
    for n in ("x", "y", "z"):
        assert got[n] == [("keep", "kept", "open_entry")], n
    for n in ("l", "e", "d"):
        assert got[n] == PLANNED, n


def test_w18_sweeper_keeps_losers_inside_the_ttl(rich, tmp_path):
    names = rich["fx"].names()
    acts = _dry_run(tmp_path, rich, _led(rich, BLOCK_SAME_ERA))
    for dg, a in acts.items():
        if names[dg] in ("l", "e", "d"):
            assert a == [("keep", "kept", "ttl_not_expired")], names[dg]


def test_w18_sweeper_keeps_a_disputed_loser(rich, tmp_path):
    fx = rich["fx"]
    acts = _dry_run(tmp_path, rich, _led(rich, BLOCK_DUE, disputes=[fx.l["digest"]]))
    assert acts[fx.l["digest"]] == [("keep", "kept", "dispute_open")]
    assert acts[fx.e["digest"]] == PLANNED


def test_w18_sweeper_keeps_the_king(crowned, tmp_path):
    fx = crowned["fx"]
    acts = _dry_run(tmp_path, crowned, _led(crowned, BLOCK_DUE))
    assert acts[fx.x["digest"]] == [("keep", "kept", "king")]
    assert acts[fx.l["digest"]] == PLANNED
