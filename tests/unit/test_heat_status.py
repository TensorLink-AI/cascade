"""The public heat mirror — ``status/heat.json``, ``heats/round-<id>.json`` and
``heats/index.json``: the standings a miner reads while the duel is still
running, instead of waiting for the round's receipt."""

from __future__ import annotations

import json
from datetime import UTC, datetime

from cascade.shared.heat_status import (
    HEAT_INDEX_KEY,
    HEAT_STATUS_KEY,
    build_heat_status,
    heat_round_key,
    heat_summary,
    live_heat,
    publish_heat_status,
    read_heat_index,
    update_heat_index,
)
from cascade.shared.hippius import StorageError
from cascade.shared.manifest import HeatEntrant, HeatResult

AS_OF = "2026-07-30T09:00:00+00:00"


def _heat() -> HeatResult:
    return HeatResult(
        screen_size="s",
        finalists=1,
        entrants=(
            HeatEntrant(uid=2, hotkey="c", gen_ref="carol/gen@sha256:" + "c" * 64,
                        status="advanced", rank=1, rel_score=1.0, p_best=0.7,
                        crps=0.4123, mase=1.021),
            HeatEntrant(uid=3, hotkey="d", gen_ref="dave/gen@sha256:" + "d" * 64,
                        status="screened", rank=2, rel_score=1.25, p_best=0.3,
                        crps=0.5, mase=1.2),
            HeatEntrant(uid=4, hotkey="e", gen_ref="eve/gen@sha256:" + "e" * 64,
                        status="failed_train"),
        ),
        leader_lcb=0.031,
        n_windows=120,
        n_clusters=9,
    )


def _doc(**over):
    doc = build_heat_status(_heat(), round_id="1", epoch_start_block=14_400,
                            as_of=AS_OF, screened=3, netuid=91)
    doc.update(over)
    return doc


class _Store:
    """Minimal S3Store stand-in: records writes, serves them back for the
    index's read-modify-write."""

    def __init__(self, fail_acl=False):
        self.fail_acl = fail_acl
        self.writes: list[tuple[str, str, str | None]] = []
        self.objects: dict[str, str] = {}

    def put_text(self, key, text, *, content_type="", acl=None):
        if acl and self.fail_acl:
            raise StorageError("acl unsupported")
        self.writes.append((key, text, acl))
        self.objects[key] = text

    def get_text(self, key):
        if key not in self.objects:
            raise StorageError(f"no such key: {key}")
        return self.objects[key]


def test_build_heat_status_mirrors_the_manifest_heat_shape():
    from cascade.shared.manifest import heat_to_json

    doc = _doc()
    assert doc["schema"] == 1
    assert (doc["round_id"], doc["epoch_start_block"]) == ("1", 14_400)
    assert (doc["as_of"], doc["screened"], doc["netuid"]) == (AS_OF, 3, 91)
    # every field of the manifest's heat block, unchanged — one shape means the
    # dashboards render a live heat and a settled round's heat identically
    body = heat_to_json(_heat())
    assert {k: doc[k] for k in body} == body
    assert "no_screen" not in doc


def test_build_heat_status_publishes_a_no_screen_round_with_its_reason():
    # Without this the live pointer would keep serving the PREVIOUS round's
    # standings as if they were this round's.
    doc = build_heat_status(None, round_id="7", epoch_start_block=21_600,
                            as_of=AS_OF, screened=1, finalists=1,
                            no_screen_reason="the field fit within the 1 finalist slot(s)")
    assert doc["no_screen"] is True
    assert doc["finalists"] == 1  # the configured slot count, not a bare 0
    assert "1 finalist slot" in doc["no_screen_reason"]
    assert doc["entrants"] == [] and doc["screen_size"] == ""
    assert doc["epoch_start_block"] == 21_600


def test_publish_writes_the_round_key_then_the_live_pointer():
    store = _Store()
    keys = publish_heat_status(store, _doc())
    assert keys == [heat_round_key("1"), HEAT_STATUS_KEY]
    # round key FIRST: a half-failed publish leaves the standings readable,
    # where a live pointer alone would vanish at the next round
    assert [k for k, _, _ in store.writes] == [heat_round_key("1"), HEAT_STATUS_KEY]
    assert all(acl == "public-read" for _, _, acl in store.writes)
    assert json.loads(store.objects[HEAT_STATUS_KEY])["round_id"] == "1"


def test_publish_falls_back_to_private_when_the_backend_rejects_acls():
    store = _Store(fail_acl=True)
    publish_heat_status(store, _doc())
    assert [acl for _, _, acl in store.writes] == [None, None]


def test_heat_summary_is_a_compact_listing_entry():
    entry = heat_summary(_doc())
    assert entry["round_id"] == "1"
    assert (entry["n_entrants"], entry["n_advanced"]) == (3, 1)
    assert (entry["leader_uid"], entry["leader_hotkey"]) == (2, "c")
    assert (entry["leader_lcb"], entry["n_windows"], entry["n_clusters"]) == (0.031, 120, 9)
    assert entry["heat_key"] == heat_round_key("1")
    assert entry["no_screen"] is False


def test_index_is_idempotent_per_round_and_chronological():
    store = _Store()
    update_heat_index(store, _doc())
    update_heat_index(store, _doc(round_id="2", epoch_start_block=7_200))
    update_heat_index(store, _doc(screened=4))  # re-run of round 1 replaces itself

    index = read_heat_index(store)
    assert index["schema"] == 1
    assert index["updated_at"] == AS_OF
    # sorted by epoch_start_block (round ids are block-hash seeds, not monotonic)
    assert [(h["round_id"], h["epoch_start_block"]) for h in index["heats"]] == [
        ("2", 7_200), ("1", 14_400)]
    assert store.writes[-1][2] == "public-read"


def test_index_caps_at_max_keep_keeping_the_newest():
    store = _Store()
    for i in range(5):
        update_heat_index(store, _doc(round_id=str(i), epoch_start_block=1_000 * i),
                          max_keep=3)
    assert [h["round_id"] for h in read_heat_index(store)["heats"]] == ["2", "3", "4"]


def test_read_index_tolerates_absent_and_malformed():
    store = _Store()
    assert read_heat_index(store) == {"schema": 1, "heats": []}
    store.objects[HEAT_INDEX_KEY] = "not json"
    assert read_heat_index(store)["heats"] == []
    store.objects[HEAT_INDEX_KEY] = json.dumps({"heats": "nope"})
    assert read_heat_index(store)["heats"] == []


def test_live_heat_accepts_this_rounds_fresh_standings():
    now = 1_784_000_000.0
    as_of = datetime.fromtimestamp(now - 1800, tz=UTC).isoformat()
    doc = live_heat(_doc(as_of=as_of), epoch_start_block=14_400, now_s=now)
    assert doc is not None and doc["round_id"] == "1"
    # a no-screen doc is still THIS round's answer (an empty field, not "unknown")
    empty = build_heat_status(None, round_id="1", epoch_start_block=14_400, as_of=as_of)
    assert live_heat(empty, epoch_start_block=14_400, now_s=now) is not None


def test_live_heat_rejects_other_rounds_stale_and_malformed():
    now = 1_784_000_000.0
    old = datetime.fromtimestamp(now - 200_000, tz=UTC).isoformat()
    future = datetime.fromtimestamp(now + 4_000, tz=UTC).isoformat()
    ok = {"epoch_start_block": 14_400, "now_s": now}
    fresh = datetime.fromtimestamp(now - 60, tz=UTC).isoformat()
    assert live_heat(_doc(as_of=old), **ok) is None                      # stale
    assert live_heat(_doc(as_of=future), **ok) is None                   # skewed
    assert live_heat(_doc(as_of=fresh), epoch_start_block=21_600, now_s=now) is None
    assert live_heat(_doc(as_of="2026-07-30T09:00:00"), **ok) is None    # naive ts
    assert live_heat(_doc(as_of="nope"), **ok) is None
    assert live_heat({"epoch_start_block": 14_400, "as_of": fresh}, **ok) is None  # no entrants
    assert live_heat(None, **ok) is None
    assert live_heat("garbage", **ok) is None
