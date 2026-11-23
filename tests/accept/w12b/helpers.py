"""Shared spec and fixtures for the W12b acceptance tests.

Copied from docs/plans/LANE_D_PLAN.md (rev6):

- W12b row: "`cascade status --json` and `discovery-check` lane D fields";
  Depends "W2, W17"; Runs in "ci"; Accept "`scripts/laned/accept W12b`".
- W17 row: "`status/lane_d.json` producer (queue, pending progress, king,
  board; rev3: per-entry reason enum including `coldkey_cap_settlement`,
  `confirm_pending` and reveal deadline)".
- §11.3 Support: "`status/lane_d.json` shows each pending entry's
  eligible-window progress against the floors."
- §3.2 invariant 1: "Unarmed ⇒ byte-identical."

Every lane D document is REAL W17 output: ``build_lane_d_status`` over a real
``settle_lane_d`` run on the W17/W7c synthetic chain (numpy only, no torch).
The public bucket is the only external service: the CLI's
``fetch_public_json`` is replaced by a dict of published objects.
"""

from __future__ import annotations

import copy
from datetime import UTC, datetime, timedelta
from functools import cache
from pathlib import Path

from tests.accept.w17 import helpers as w17

LANE_D_KEY = "status/lane_d.json"
LANE_D_SCHEMA = "lane-d-status/1"
ERA_BLOCKS = 3600
NOW = datetime(2026, 11, 23, 12, 0, tzinfo=UTC)
PTR = "metro-v1:trained:hippius:cascade/init@sha256:" + "a" * 64
TODAY_SOURCES = {"chain": "status/chain.json", "competition": "status/v2.json",
                 "lane_a_queue": "funded/latest.json", "lane_b_queue": "status/lane_b.json",
                 "lane_c_queue": "status/lane_c.json"}
LANE_D_FRESH_KEYS = {"available", "block", "age_blocks", "fresh"}
LANE_D_SUMMARY_KEYS = {"active", "lane", "schema", "block", "era", "boundary", "king",
                       "house", "floors"}

DOC_BLOCK = w17.B1 + 17


@cache
def _rich(root: str) -> dict:
    return w17.rich_b1(Path(root))


def lane_d_doc(root: Path, *, block: int = DOC_BLOCK, cfg=None) -> dict:
    """The status/lane_d.json document W17 builds for the rich B1 chain at
    ``block`` (a deep copy: tests may edit it)."""
    sc = _rich(str(root))
    doc = w17.status(sc["cfg"] if cfg is None else cfg, state=sc["state"], report=sc["report"],
                     block=block)
    assert doc is not None and doc["schema"] == LANE_D_SCHEMA
    return copy.deepcopy(doc)


def closed_doc(root: Path) -> dict:
    """The same chain after the overlay closed the lane (``until_block``
    before the document's block): ``lane == "closed"``."""
    until = w17.ARM_BLOCK + ERA_BLOCKS           # an era boundary (W2 requires one)
    cfg = w17.chain_cfg(max_judged_per_coldkey_per_settlement=1, until_block=until)
    doc = lane_d_doc(root, cfg=cfg, block=until + 17)
    assert doc["lane"] == "closed"
    return doc


# ── reading the W12b fields (a missing field fails as an assertion) ─────────


def queue_d(snap: dict):
    assert "d" in snap.get("queues", {}), "status --json needs queues.d (W12b)"
    return snap["queues"]["d"]


def lane_d(snap: dict):
    assert "d" in snap.get("lanes", {}), "status --json needs lanes.d (W12b)"
    return snap["lanes"]["d"]


def lane_d_freshness(snap: dict) -> dict:
    fr = snap.get("freshness", {}).get("lane_d_queue")
    assert isinstance(fr, dict), "status --json needs freshness.lane_d_queue (W12b)"
    return fr


def docs(*, current_block: int, now: datetime = NOW) -> dict:
    """status/chain.json + status/v2.json, both fresh at ``now``, with the
    current era's warm-start init (so ``ready_for_training`` holds)."""
    era_start = current_block // ERA_BLOCKS * ERA_BLOCKS
    return {"chain": {"as_of": now.isoformat(), "current_block": current_block,
                      "era_blocks": ERA_BLOCKS},
            "competition": {"as_of": now.isoformat(), "block": current_block,
                            "warm_start": {"pointer": PTR, "mode": "checkpoint",
                                           "era_start_block": era_start},
                            "lane_b": {"active": False}, "lane_c": {"active": False}}}


def snapshot(all_docs: dict, *, now: datetime = NOW, max_age: float = 180.0) -> dict:
    from cascade.miner.live_status import status_snapshot

    return status_snapshot(copy.deepcopy(all_docs), now=now, max_age=max_age)


def stale(now: datetime, seconds: float) -> datetime:
    return now - timedelta(seconds=seconds)
