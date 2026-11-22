"""LANE_D_PLAN §16 rule 4, §1.3b "Progress board" (W-S3): the era's sentinel
numbers on the status boards.

W-S1 (:mod:`cascade.shared.sentinel`) draws the era's sentinel suite from the
era-start block hash; W3b3 (:mod:`cascade.validator.lane_d_sentinel`) sweeps it
for every new KING_D, the previous KING_D and the house, and publishes a
``lane-d-sentinel/1`` record under ``sentinel_key(era_block, king)``.
:func:`build_sentinel_board` reads those records back and lays them out as a
``sentinel-board/1`` document for ``status/v2.json`` (and W17's
``status/lane_d.json``).

PRESENTATIONAL and NON-GATING: the board computes no number (every float on it
is the published one, verbatim), never ranks or hides a king, and nothing a
verdict depends on reads it. A record that is absent, unreadable or not a
valid record of this king and era is shown as such (``missing`` /
``unavailable`` / ``invalid``, ``models: None``), never as a number.

Unarmed (``[v2] sentinel_from_block`` = 0, or an era before it) there is no
board and no store read. Deterministic: no wall-clock or process-local field;
the same records give the same board on every host.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from typing import Any

from .hippius import ObjectNotFound, StorageError
from .sentinel import sentinel_armed, sentinel_for_era

BOARD_SCHEMA = "sentinel-board/1"
RECORD_SCHEMA = "lane-d-sentinel/1"
ROLES = ("king", "previous_king", "house")
RECORD_STATES = ("ok", "fault", "error")
STATES = (*RECORD_STATES, "missing", "unavailable", "invalid")
METRICS = ("crps", "mase")
LABEL = ("Sentinel (plan §16 rule 4) - non-gating: the era's held-out suite, swept for "
         "each new lane D king, the previous king and the house. Shown for comparison "
         "only; it never decides a crowning or a payout.")


class InvalidRecord(ValueError):
    """The object at a king's key is not a valid record of this king and era."""


# ── argument checks ──────────────────────────────────────────────────────────


def _check_era_block(era_block: object) -> int:
    if isinstance(era_block, bool) or not isinstance(era_block, int):
        raise TypeError(f"era_block must be an int; got {era_block!r}")
    if era_block < 0:
        raise ValueError(f"era_block must be >= 0; got {era_block!r}")
    return era_block


def _check_kings(kings: object) -> tuple[str, ...]:
    if isinstance(kings, (str, bytes)) or not isinstance(kings, Iterable):
        raise TypeError(f"kings must be an iterable of digests; got {kings!r}")
    out = tuple(kings)
    for k in out:
        if not isinstance(k, str):
            raise TypeError(f"a king digest must be a str; got {k!r}")
        if not k:
            raise ValueError("a king digest is empty")
    if len(set(out)) != len(out):
        raise ValueError(f"kings has duplicates: {list(out)}")
    return out


# ── record validation ────────────────────────────────────────────────────────


def _refuse_constant(name: str) -> float:
    raise InvalidRecord(f"non-strict JSON constant {name}")


def _parse(text: object) -> dict:
    if not isinstance(text, str):
        raise InvalidRecord(f"the object is not text ({type(text).__name__})")
    try:
        rec = json.loads(text, parse_constant=_refuse_constant)
    except ValueError as e:
        raise InvalidRecord(f"not strict JSON: {e}") from None
    if not isinstance(rec, dict):
        raise InvalidRecord(f"not a JSON object ({type(rec).__name__})")
    return rec


def _is_int(v: object) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _is_number(v: object) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _check_model(role: str, m: object, digest: str | None) -> str:
    """The model entry's status after its shape checks."""
    if not isinstance(m, dict):
        raise InvalidRecord(f"models[{role!r}] is not an object")
    d, status = m.get("digest"), m.get("status")
    if not isinstance(d, str) or not d:
        raise InvalidRecord(f"models[{role!r}] has no digest")
    if digest is not None and d != digest:
        raise InvalidRecord(f"models[{role!r}] digest {d!r} is not {digest!r}")
    if status not in RECORD_STATES:
        raise InvalidRecord(f"models[{role!r}] has an unknown status {status!r}")
    if status == "ok":
        items = m.get("items")
        if not _is_int(items) or items < 1:
            raise InvalidRecord(f"ok models[{role!r}] items {items!r} is not a positive int")
        for k in METRICS:
            if not _is_number(m.get(k)):
                raise InvalidRecord(f"ok models[{role!r}] {k} {m.get(k)!r} is not a finite number")
    else:
        for k in ("items", *METRICS):
            if m.get(k) is not None:
                raise InvalidRecord(f"{status} models[{role!r}] carries {k} {m.get(k)!r}")
    return status


def _check_record(rec: dict, *, king: str, era_block: int, era_block_hash: str,
                  suite: str, suites: tuple[str, ...]) -> dict:
    """``rec`` when it is a valid ``lane-d-sentinel/1`` record of ``king`` for
    this era and draw; raises :class:`InvalidRecord` otherwise."""
    if rec.get("schema") != RECORD_SCHEMA:
        raise InvalidRecord(f"schema {rec.get('schema')!r} is not {RECORD_SCHEMA!r}")
    if rec.get("king_digest") != king:
        raise InvalidRecord(f"king_digest {rec.get('king_digest')!r} is not this king")
    era = rec.get("era")
    if not isinstance(era, dict):
        raise InvalidRecord("era is not an object")
    if not _is_int(era.get("block")) or era["block"] != era_block:
        raise InvalidRecord(f"era block {era.get('block')!r} is not {era_block}")
    if era.get("block_hash") != era_block_hash:
        raise InvalidRecord(f"era block_hash {era.get('block_hash')!r} is not the era's")
    if rec.get("suite") != suite:
        raise InvalidRecord(f"suite {rec.get('suite')!r} is not the era's draw {suite!r}")
    if rec.get("sentinel_suites") != list(suites):
        raise InvalidRecord(f"sentinel_suites {rec.get('sentinel_suites')!r} are not "
                            f"{list(suites)}")
    if rec.get("gating") is not False:
        raise InvalidRecord(f"gating {rec.get('gating')!r} is not false")
    status = rec.get("status")
    if status not in RECORD_STATES:
        raise InvalidRecord(f"unknown status {status!r}")
    reason = rec.get("reason")
    if status == "ok" and reason is not None:
        raise InvalidRecord(f"an ok record carries a reason {reason!r}")
    if status != "ok" and (not isinstance(reason, str) or not reason):
        raise InvalidRecord(f"a {status} record has no reason")
    boundary = rec.get("boundary")
    if boundary is not None and not _is_int(boundary):
        raise InvalidRecord(f"boundary {boundary!r} is not an int")
    gate = rec.get("gate_suite")
    if not isinstance(gate, str) or not gate or gate == suite:
        raise InvalidRecord(f"gate_suite {gate!r} is not a suite other than the sentinel")
    session = rec.get("session")
    if not isinstance(session, str) or not session:
        raise InvalidRecord(f"session {session!r} is not a non-empty str")
    models = rec.get("models")
    if not isinstance(models, dict) or set(models) != set(ROLES):
        raise InvalidRecord(f"models must name exactly {list(ROLES)}")
    per = [_check_model(role, models[role], king if role == "king" else None)
           for role in ROLES]
    consistent = {"ok": all(s == "ok" for s in per),
                  "fault": "fault" in per and "error" not in per,
                  "error": all(s == "error" for s in per)}[status]
    if not consistent:
        raise InvalidRecord(f"a {status} record has model statuses {per}")
    return rec


# ── rows ─────────────────────────────────────────────────────────────────────


def _no_record_row(king: str, key: str, state: str, reason: str) -> dict:
    return {"king_digest": king, "key": key, "state": state, "reason": reason,
            "boundary": None, "suite": None, "gate_suite": None, "session": None,
            "models": None}


def _row(store: Any, king: str, key: str, *, era_block: int, era_block_hash: str,
         suite: str, suites: tuple[str, ...]) -> dict:
    try:
        text = store.get_text(key)
    except ObjectNotFound:
        return _no_record_row(king, key, "missing", "no sentinel record is published at "
                                                    "this key")
    except StorageError as e:
        return _no_record_row(king, key, "unavailable", f"store read failed: {e}")
    except Exception as e:  # noqa: BLE001 — a dashboard read never raises
        return _no_record_row(king, key, "unavailable",
                              f"store read failed: {type(e).__name__}: {e}")
    try:
        rec = _check_record(_parse(text), king=king, era_block=era_block,
                            era_block_hash=era_block_hash, suite=suite, suites=suites)
    except InvalidRecord as e:
        return _no_record_row(king, key, "invalid", f"not a valid {RECORD_SCHEMA} record "
                                                    f"of this king and era: {e}")
    return {"king_digest": king, "key": key, "state": rec["status"],
            "reason": rec["reason"], "boundary": rec["boundary"], "suite": rec["suite"],
            "gate_suite": rec["gate_suite"], "session": rec["session"],
            "models": {role: dict(rec["models"][role]) for role in ROLES}}


# ── the board ────────────────────────────────────────────────────────────────


def build_sentinel_board(cfg: Any, *, era_block: int, era_block_hash: str,
                         kings: Iterable[str], store: Any) -> dict | None:
    """The era's ``sentinel-board/1`` document, or ``None`` (no store read)
    when the sentinel is not armed at ``era_block``.

    One row per entry of ``kings`` (the era's new KING_D digests), in that
    order, keyed by :func:`cascade.validator.lane_d_sentinel.sentinel_key`.
    Raises ``ValueError`` / ``TypeError`` before any store read on a
    malformed ``era_block``, ``era_block_hash`` or ``kings``; never raises
    on a store failure (the row is ``unavailable``). Reads only the listed
    kings' keys and writes nothing."""
    if not sentinel_armed(cfg, era_block):
        return None
    block = _check_era_block(era_block)
    suites = tuple(cfg.v2.sentinel_suites)
    suite = sentinel_for_era(era_block_hash, suites)
    names = _check_kings(kings)

    from cascade.validator.lane_d_sentinel import sentinel_key

    rows = [_row(store, k, sentinel_key(block, k), era_block=block,
                 era_block_hash=era_block_hash, suite=suite, suites=suites)
            for k in names]
    return {"schema": BOARD_SCHEMA, "gating": False, "label": LABEL,
            "era": {"block": block, "block_hash": era_block_hash},
            "suite": suite, "sentinel_suites": list(suites), "kings": rows}


__all__ = ["BOARD_SCHEMA", "LABEL", "RECORD_SCHEMA", "ROLES", "STATES",
           "build_sentinel_board"]
