"""``status/lane_d.json`` and the W18 vault ledger (LANE_D_PLAN W17).

Two pure, presentational exports of the lane D state (``lane_d_state.json``)
and its signed ``lane-d-report/1`` chain:

- :func:`build_lane_d_status` is the public status document (§11.2, §11.3
  "Support", §1.3b "Progress board"): the FIFO queue of open entries with
  the per-entry typed reason (:data:`REASON_ENUM`, the state's own enum),
  the confirmation stage, the reveal deadline and the eligible-window
  progress against the floors; KING_D, the house and the prior kings; the
  board of the latest signed report; and, when W-S1 is armed, the era's
  W-S3 sentinel board (:func:`cascade.shared.sentinel_board.build_sentinel_board`).
  :func:`publish_lane_d_status` writes it at :data:`LANE_D_STATUS_KEY`.
- :func:`build_vault_ledger` is the ``laned_vault_ledger/1`` object W18's
  sweeper (:func:`cascade.laned.vault_sweeper.parse_ledger`) reads: the
  current era, every vault entry with its ledger status and ``closed_era``,
  and the open disputes.

Neither function verdicts anything: the status board and the ledger only
restate what the state and the signed reports already hold. Neither reads a
clock, writes a file or mutates its inputs; equal inputs give equal outputs
whatever the key order of the state. Unarmed (``[lane_d] from_block = 0`` or
a block before it), :func:`build_lane_d_status` returns ``None`` and nothing
is published (invariant 1). Credentials, URLs and the report signature are
never part of either document (§1.2). Wiring the publish into the validator
loop is W8b's.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterable
from typing import Any

from ..shared.config import BLOCK_SECONDS
from ..shared.era import era_index_of, settlement_era
from . import lane_d_state

LANE_D_STATUS_KEY = "status/lane_d.json"
STATUS_SCHEMA = "lane-d-status/1"
LEDGER_SCHEMA = "laned_vault_ledger/1"
# The per-entry reason enum is the state's own, never a copy that can drift.
REASON_ENUM = tuple(lane_d_state.ENTRY_REASONS)

# A9 "maximum reveal delay" is configured in days; one day of 12 s blocks.
BLOCKS_PER_DAY = int(86400 // BLOCK_SECONDS)

# State statuses of an entry still in the queue (queue selection walks these).
OPEN_STATUSES = ("pending", "hold")
# The README's ledger mapping for a ``closed`` entry's reason; any other
# closed reason is ``closed_unjudged``.
_CLOSED_LEDGER = {"deregistered": "voided", "ready_expired": "ready_expired"}
_VERDICT_FIELDS = ("digest", "king_digest", "stage", "outcome", "lcb", "gain")
_HEX64 = re.compile(r"[0-9a-f]{64}")
# §1.2: nothing URL- or credential-shaped is ever published.
_FORBIDDEN = ("://", "X-Amz-", "x-amz-", "Signature=", "Credential=", "presigned")
_SECRET_ENV = ("HIPPIUS_S3_ACCESS_KEY", "HIPPIUS_S3_SECRET_KEY", "HF_TOKEN")


# ── helpers ─────────────────────────────────────────────────────────────────


def _armed(cfg: Any, block: int) -> bool:
    start = int(cfg.lane_d.from_block or 0)
    return start > 0 and int(block) >= start


def _check_binding(state: dict, report: dict | None) -> None:
    """``report`` must be the report ``state["last_report"]`` binds."""
    bound = state.get("last_report")
    if report is None:
        if bound is not None:
            raise ValueError("lane_d_status: the state binds a report but none was given")
        return
    if bound is None:
        raise ValueError("lane_d_status: a report was given but the state binds none")
    if int(report["boundary"]) != int(state["boundary"]):
        raise ValueError(f"lane_d_status: report boundary {report['boundary']} is not the "
                         f"state's {state['boundary']}")
    if lane_d_state.report_digest(report) != bound["report_sha256"]:
        raise ValueError("lane_d_status: the report's body digest is not the one the state "
                         "binds")


def _progress(cfg: Any, rec: dict) -> dict:
    fs = rec.get("frozen_set")
    if fs is None:
        windows, clusters, frozen = 0, 0, False
    else:
        windows = len(fs["window_ids"])
        clusters = len(set(fs["sources"].values()))
        frozen = bool(fs["frozen"])
    return {"windows": windows, "clusters": clusters,
            "min_windows": int(cfg.lane_d.min_windows),
            "min_clusters": int(cfg.lane_d.min_clusters), "frozen": frozen}


def _queue(cfg: Any, state: dict) -> list[dict]:
    delay = int(cfg.lane_d.max_reveal_delay_days) * BLOCKS_PER_DAY
    open_ = sorted(((dg, rec) for dg, rec in state["entries"].items()
                    if rec["status"] in OPEN_STATUSES),
                   key=lambda item: (int(item[1]["reveal_ts"]), item[0]))
    rows = []
    for dg, rec in open_:
        reason = rec["reason"]
        if reason not in REASON_ENUM:
            raise ValueError(f"lane_d_status: entry {dg[:12]} has reason {reason!r} outside "
                             "the enum")
        rows.append({
            "digest": dg, "hotkey": rec["hotkey"], "coldkey": rec["coldkey_at_ready"],
            "context": int(rec["context"]), "status": rec["status"], "reason": reason,
            "ready_block": int(rec["ready_block"]), "reveal_block": int(rec["reveal_block"]),
            "reveal_ts": int(rec["reveal_ts"]),
            "reveal_deadline_block": int(rec["reveal_block"]) + delay,
            "confirm_stage": "confirm" if rec.get("confirm") is not None else "first",
            "progress": _progress(cfg, rec)})
    return rows


def _king(state: dict, block: int) -> dict:
    k = state["king"]
    if k.get("house"):
        return {"digest": k["digest"], "hotkey": None, "house": True,
                "since_boundary": None, "tenure_blocks": None}
    since = int(k["since_boundary"])
    return {"digest": k["digest"], "hotkey": k["hotkey"], "house": False,
            "since_boundary": since, "tenure_blocks": int(block) - since}


def _board(report: dict | None) -> dict | None:
    if report is None:
        return None
    gate = report.get("gate")
    return {"boundary": int(report["boundary"]), "crowned": report.get("crowned"),
            "judged": json.loads(json.dumps(report.get("judged", []))),
            "skipped": json.loads(json.dumps(report.get("skipped", []))),
            "verdicts": [{f: v[f] for f in _VERDICT_FIELDS} for v in report["verdicts"]],
            "gate": None if gate is None else json.loads(json.dumps(gate))}


def _sentinel(cfg: Any, king: dict, *, era_block: int | None, era_block_hash: str | None,
              store: Any) -> dict | None:
    from ..shared.sentinel import sentinel_armed, settlement_era_start

    if era_block is None or not sentinel_armed(cfg, era_block):
        return None
    from ..shared.sentinel_board import build_sentinel_board

    kings = []
    if not king["house"] and settlement_era_start(cfg, king["since_boundary"]) == int(era_block):
        kings.append(king["digest"])
    return build_sentinel_board(cfg, era_block=int(era_block), era_block_hash=era_block_hash,
                                kings=kings, store=store)


# ── status/lane_d.json ──────────────────────────────────────────────────────


def build_lane_d_status(cfg: Any, *, state: dict, report: dict | None, block: int,
                        era_block: int | None = None, era_block_hash: str | None = None,
                        store: Any = None) -> dict | None:
    """The ``status/lane_d.json`` document, or ``None`` while lane D is not
    armed at ``block`` (nothing is read or published). ``report`` must be the
    report ``state["last_report"]`` binds (else ``ValueError``). The only
    store reads are the embedded sentinel board's, and only when
    ``era_block`` is given and W-S1 is armed there."""
    if not _armed(cfg, block):
        return None
    _check_binding(state, report)
    until = int(cfg.lane_d.until_block or 0)
    king = _king(state, block)
    house = state["house"]
    doc = {
        "schema": STATUS_SCHEMA,
        "lane": "closed" if until > 0 and int(block) > until else "open",
        "block": int(block),
        "era": era_index_of(cfg.round, int(block)),
        "boundary": int(state["boundary"]),
        "reason_enum": list(REASON_ENUM),
        "floors": {"min_windows": int(cfg.lane_d.min_windows),
                   "min_clusters": int(cfg.lane_d.min_clusters)},
        "king": king,
        "house": {"digest": house["digest"], "reveal_ts": int(house["reveal_ts"])},
        "prior_kings": [{"digest": p["digest"], "hotkey": p.get("hotkey"),
                         "reveal_ts": int(p["reveal_ts"])} for p in state["prior_kings"]],
        "queue": _queue(cfg, state),
        "board": _board(report),
        "sentinel": _sentinel(cfg, king, era_block=era_block, era_block_hash=era_block_hash,
                              store=store),
    }
    return doc


def _render(doc: dict) -> str:
    return json.dumps(doc, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False)


def _refuse_secrets(text: str) -> None:
    for needle in _FORBIDDEN:
        if needle in text:
            raise ValueError(f"lane_d_status: refusing to publish a {needle!r}-shaped value")
    for name in _SECRET_ENV:
        value = os.environ.get(name)
        if value and value in text:
            raise ValueError(f"lane_d_status: refusing to publish the value of {name}")


def publish_lane_d_status(store: Any, doc: dict | None) -> str | None:
    """Write ``doc`` at :data:`LANE_D_STATUS_KEY` (one ``put_text``,
    ``application/json``, public-read like the other status files) and
    return the key; ``None`` (no store call) when ``doc`` is ``None``. The
    bytes are canonical (sorted keys, compact, ASCII, strict JSON), so the
    same document always gives the same bytes."""
    if doc is None:
        return None
    text = _render(doc)
    _refuse_secrets(text)
    store.put_text(LANE_D_STATUS_KEY, text, content_type="application/json",
                   acl="public-read")
    return LANE_D_STATUS_KEY


# ── laned_vault_ledger/1 (W18) ──────────────────────────────────────────────


def _ledger_status(dg: str, rec: dict) -> str:
    status = rec["status"]
    if status in OPEN_STATUSES:
        return "pending"
    if status == "crowned":
        return "king"
    if status == "lost":
        return "lost"
    if status == "closed":
        return _CLOSED_LEDGER.get(rec["reason"], "closed_unjudged")
    raise ValueError(f"lane_d_status: entry {dg[:12]} has unknown status {status!r}")


def _closing_boundaries(reports: Iterable[dict]) -> dict[str, int]:
    """digest -> the lowest report boundary listing it ``lost`` or ``closed``."""
    out: dict[str, int] = {}
    for rep in reports:
        boundary = int(rep["boundary"])
        for e in rep.get("entries") or ():
            if e.get("status") in ("lost", "closed"):
                dg = e["digest"]
                if dg not in out or boundary < out[dg]:
                    out[dg] = boundary
    return out


def _disputes(disputes: Iterable[str]) -> list[str]:
    out = set()
    for d in disputes:
        if not isinstance(d, str) or not _HEX64.fullmatch(d):
            raise ValueError(f"lane_d_status: dispute {d!r} is not a 64 lowercase hex digest")
        out.add(d)
    return sorted(out)


def build_vault_ledger(cfg: Any, *, state: dict, reports: Iterable[dict], block: int,
                       disputes: Iterable[str] = ()) -> dict:
    """The ``laned_vault_ledger/1`` object for W18's sweeper: ``era`` at
    ``block``, one row per state entry (never the house) sorted by digest,
    and the deduplicated, sorted ``disputes``. A closed or lost entry's
    ``closed_era`` is the settlement era of the earliest report in
    ``reports`` that records its closure; one no report records is a
    ``ValueError``. Statuses always come from the state."""
    closing = _closing_boundaries(reports)
    rows = []
    for dg in sorted(state["entries"]):
        rec = state["entries"][dg]
        status = _ledger_status(dg, rec)
        closed_era = None
        if rec["status"] in ("lost", "closed"):
            if dg not in closing:
                raise ValueError(f"lane_d_status: no report records the closure of "
                                 f"{dg[:12]}")
            closed_era = int(settlement_era(cfg.round, closing[dg]).index)
        rows.append({"hotkey": rec["hotkey"], "digest": dg, "status": status,
                     "closed_era": closed_era})
    return {"schema": LEDGER_SCHEMA, "era": int(era_index_of(cfg.round, int(block))),
            "entries": rows, "disputes": _disputes(disputes)}


__all__ = ["BLOCKS_PER_DAY", "LANE_D_STATUS_KEY", "LEDGER_SCHEMA", "REASON_ENUM",
           "STATUS_SCHEMA", "build_lane_d_status", "build_vault_ledger",
           "publish_lane_d_status"]
