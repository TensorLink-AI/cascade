"""The lane-C slot-holder registration record (owner 2026-11-24).

MarketCast evicts a slot holder whose hotkey is deregistered AS OF the
settlement boundary − 1 (``_mkt_settle``). That is a read at a past block,
which a lite node prunes within the hour, so it is recorded instead of
re-read:

* every validator snapshots ``hotkey → uid`` of the whole netuid at ``B − 1``
  while that block is still fresh on any node (the boundary snapshot,
  :class:`RegistrationSnapshots`), and judges from it;
* the receipt records, for the settlement's current round, one row per slot
  holder whose registration was consulted — ``{target, hotkey, uid}`` (uid
  ``None`` = deregistered) — under ``block`` (= ``B − 1``) and the rows'
  ``sha256`` (:func:`registration_record`);
* satellites and the audit replay that record at the same trust level as the
  rows (:func:`registration_record_problem`), the way the lane-D ``coldkey``
  is replayed — no chain history needed;
* a validator catching up late, with no boundary snapshot, takes a holder's
  registration from the MAIN validator's signed receipt record
  (:func:`recorded_uid`), and only without one reads the chain (whose client
  falls back to an archive endpoint for a pruned block).

Nothing here touches lane A: the record exists only on a lane-C settlement
that consulted a slot holder's registration.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path

log = logging.getLogger("cascade.lane_c")

# How far past ``B − 1`` the boundary snapshot is still read live (blocks).
# Lite nodes keep ~256 blocks of state; stay inside that.
SNAPSHOT_FRESH_BLOCKS = 240
SNAPSHOT_KEEP = 32


def _canonical_rows(rows) -> list[dict]:
    out = [{"target": str(r["target"]), "hotkey": str(r["hotkey"]),
            "uid": None if r.get("uid") is None else int(r["uid"])} for r in rows]
    # one row per (target, hotkey); deterministic order
    uniq = {(r["target"], r["hotkey"]): r for r in out}
    return [uniq[k] for k in sorted(uniq)]


def registration_rows_sha256(rows) -> str:
    text = json.dumps(_canonical_rows(rows), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def registration_record(block: int, rows) -> dict | None:
    """The receipt's ``lane_c_registrations`` record, or None without rows
    (the receipt then carries no field — its bytes unchanged)."""
    canon = _canonical_rows(rows or ())
    if not canon:
        return None
    return {"block": int(block), "rows": canon, "sha256": registration_rows_sha256(canon)}


def recorded_uid(record: dict | None, hotkey: str, block: int) -> tuple[bool, int | None]:
    """``(found, uid)`` of ``hotkey`` in a verified record at ``block``."""
    if not isinstance(record, dict) or int(record.get("block", -1)) != int(block):
        return False, None
    rows = record.get("rows") or ()
    if registration_rows_sha256(rows) != str(record.get("sha256", "")):
        return False, None
    for r in rows:
        if str(r.get("hotkey")) == str(hotkey):
            return True, (None if r.get("uid") is None else int(r["uid"]))
    return False, None


def _kind(target: str) -> str:
    return str(target or "").strip().partition(":")[0]


def _slot_hotkey_block(w) -> tuple[str, int] | None:
    """``(hotkey, win_block)`` of a ``state_post`` lane-C slot (the list
    shape of :func:`cascade.shared.settlement_state.settlement_state`)."""
    if isinstance(w, (list, tuple)) and len(w) >= 3:
        return str(w[0]), int(w[2])
    if isinstance(w, dict):
        return str(w.get("hotkey", "")), int(w.get("win_block", 0))
    return None


def registration_record_problem(record: dict | None, *, boundary: int,
                                state_post: dict | None) -> str | None:
    """Why a receipt's registration record is inconsistent, or None.

    Checked without the chain: the rows' digest; the block is ``boundary − 1``;
    rows are canonical (sorted, unique, uid int ≥ 0 or null); a row recording
    a DEREGISTERED holder must not still hold that target with a win from
    before this settlement (``state_post``: a deregistered holder is evicted);
    and, per competition kind the record covers, every holder still holding
    a pre-settlement win has a row (every holder judged is recorded)."""
    if record is None:
        return None
    if not isinstance(record, dict):
        return "lane_c_registrations is not an object"
    rows = record.get("rows")
    if not isinstance(rows, list) or not rows:
        return "lane_c_registrations has no rows"
    try:
        canon = _canonical_rows(rows)
    except Exception as e:  # noqa: BLE001
        return f"lane_c_registrations rows malformed: {e}"
    if json.loads(json.dumps(rows)) != canon:
        return "lane_c_registrations rows are not canonical (sorted, unique, typed)"
    if any(r["uid"] is not None and r["uid"] < 0 for r in canon):
        return "lane_c_registrations row with a negative uid"
    if registration_rows_sha256(canon) != str(record.get("sha256", "")):
        return "lane_c_registrations sha256 does not match its rows"
    if int(record.get("block", -1)) != int(boundary) - 1:
        return (f"lane_c_registrations block {record.get('block')} != settlement "
                f"boundary − 1 ({int(boundary) - 1})")
    slots = {}
    if isinstance(state_post, dict) and isinstance(state_post.get("lane_c"), dict):
        slots = state_post["lane_c"].get("slots") or {}
    kinds = {_kind(r["target"]) for r in canon}
    held = {}
    for t, w in (slots.items() if isinstance(slots, dict) else ()):
        hb = _slot_hotkey_block(w)
        if hb is None or _kind(str(t)) not in kinds:
            continue
        if hb[1] < int(boundary):
            held[str(t)] = hb[0]
    rowmap = {(r["target"], r["hotkey"]): r["uid"] for r in canon}
    for (t, hk), uid in rowmap.items():
        if uid is None and held.get(t) == hk:
            return (f"lane_c_registrations records holder {hk[:12]} of {t} deregistered at "
                    f"block {int(boundary) - 1}, but it still holds the slot")
    for t, hk in sorted(held.items()):
        if (t, hk) not in rowmap:
            return (f"lane_c_registrations has no row for slot holder {hk[:12]} of {t} "
                    "(every holder judged is recorded)")
        if rowmap[(t, hk)] is None:
            return f"lane_c_registrations: deregistered holder {hk[:12]} kept {t}"
    return None


class RegistrationSnapshots:
    """``{block: {hotkey: uid}}`` read live at each settlement boundary − 1,
    persisted beside the validator state (atomic, last
    :data:`SNAPSHOT_KEEP` blocks). A local cache, never consensus: it holds
    exactly the chain state at that block."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._data: dict[int, dict[str, int]] | None = None

    def _load(self) -> dict[int, dict[str, int]]:
        if self._data is None:
            data: dict[int, dict[str, int]] = {}
            try:
                raw = json.loads(self.path.read_text(encoding="utf-8"))
                for b, m in (raw.get("blocks") or {}).items():
                    data[int(b)] = {str(k): int(v) for k, v in dict(m).items()}
            except FileNotFoundError:
                pass
            except Exception as e:  # noqa: BLE001 — a torn cache is rebuilt / falls back
                log.warning("lane C: registration snapshots %s unreadable (%s); ignored",
                            self.path, e)
            self._data = data
        return self._data

    def get(self, block: int) -> dict[str, int] | None:
        return self._load().get(int(block))

    def put(self, block: int, uids: dict[str, int]) -> None:
        data = self._load()
        data[int(block)] = {str(k): int(v) for k, v in uids.items()}
        for b in sorted(data)[:-SNAPSHOT_KEEP]:
            del data[b]
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps({"blocks": {str(b): data[b] for b in sorted(data)}},
                                  sort_keys=True), encoding="utf-8")
        os.replace(tmp, self.path)


__all__ = ["RegistrationSnapshots", "SNAPSHOT_FRESH_BLOCKS", "recorded_uid",
           "registration_record", "registration_record_problem", "registration_rows_sha256"]
