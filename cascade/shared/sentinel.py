"""LANE_D_PLAN §16 (D33): the cross-lane rotating held-out sentinel.

At each era boundary one suite of ``[v2] sentinel_suites`` is drawn from the
era-boundary block hash — the same on every validator, unpredictable at commit
time — and held out of lane A's bench gate and king check for that era (they
judge on the remaining suites; with ``[v2] sentinel_ratchet`` the candidate
may still not regress on it by more than ``sentinel_tau``, D34). The main
validator records the draw on the settlement receipt (:func:`sentinel_record`);
satellites verify it from the block hash, or replay the recorded value when
the hash is unreadable (:func:`cascade.validator.satellite.sentinel_replay`).

CONSENSUS, block-gated by ``[v2] sentinel_from_block`` (0 = off): unarmed,
nothing here is consulted and every receipt, verdict and vector keeps its
bytes. Pure: no I/O, no process-local state (no ``hash()``).
"""

from __future__ import annotations

import hashlib
import string
from collections.abc import Sequence
from typing import Any

# Domain separation: the round base seed hashes the same block hash
# (:func:`cascade.shared.chain.seed_from_block_hash`, blake2b); the draw uses
# its own tagged sha256 so the two are independent.
_DRAW_TAG = b"cascade|sentinel|v1|"
_HEX = frozenset(string.hexdigits.lower())


def _hash_hex(block_hash: str) -> str:
    """The 64 lowercase hex digits of a ``0x``-prefixed block hash, or raise."""
    if not isinstance(block_hash, str):
        raise TypeError(f"block hash must be a string; got {type(block_hash).__name__}")
    h = block_hash.strip().lower()
    if not h.startswith("0x"):
        raise ValueError(f"block hash {block_hash!r} is not 0x-prefixed")
    h = h[2:]
    if len(h) != 64 or not set(h) <= _HEX:
        raise ValueError(f"block hash {block_hash!r} is not 0x + 64 hex digits")
    return h


def sentinel_for_era(block_hash: str, suites: Sequence[str]) -> str:
    """The era's sentinel suite: ``suites[i]`` with ``i`` = the tagged sha256
    of the era-boundary block hash, as a big-endian integer, modulo
    ``len(suites)`` (uniform up to a 2**-250 bias; independent between eras
    because each era's hash is). Raises ``ValueError`` / ``TypeError`` on a
    malformed hash or empty ``suites``."""
    names = tuple(suites)
    if not names:
        raise ValueError("sentinel_suites is empty")
    digest = hashlib.sha256(_DRAW_TAG + _hash_hex(block_hash).encode("ascii")).digest()
    return str(names[int.from_bytes(digest, "big") % len(names)])


def sentinel_from_block(cfg: Any) -> int:
    """``[v2] sentinel_from_block`` (0 = off)."""
    return int(getattr(getattr(cfg, "v2", None), "sentinel_from_block", 0) or 0)


def sentinel_armed(cfg: Any, block: int | None) -> bool:
    """CONSENSUS: whether the era starting at ``block`` holds a sentinel out —
    ``sentinel_from_block > 0`` and ``block >= sentinel_from_block``."""
    n = sentinel_from_block(cfg)
    return n > 0 and block is not None and int(block) >= n


def settlement_era_start(cfg: Any, boundary: int) -> int:
    """The era start a settlement at epoch boundary ``boundary`` belongs to
    (:func:`cascade.shared.era.settlement_contract_block`): its sentinel is
    drawn from that block's hash."""
    from .era import settlement_contract_block

    return int(settlement_contract_block(cfg.round, int(boundary)))


def sentinel_record(cfg: Any, era_start_block: int, block_hash: str) -> dict | None:
    """The receipt record of the era's draw — ``{"suite", "block",
    "block_hash"}`` — or ``None`` when the sentinel is not armed at
    ``era_start_block`` (the receipt then carries no field)."""
    if not sentinel_armed(cfg, era_start_block):
        return None
    return {"suite": sentinel_for_era(block_hash, cfg.v2.sentinel_suites),
            "block": int(era_start_block), "block_hash": str(block_hash)}


def sentinel_gate_kw(cfg: Any, suite: str | None) -> dict:
    """The keywords the lane-A bench gate / king check take for an armed
    era's ``suite`` (``{}`` when ``suite`` is None — the call is then the
    unarmed one, byte-identical)."""
    if suite is None:
        return {}
    return {"sentinel": str(suite), "sentinel_ratchet": bool(cfg.v2.sentinel_ratchet),
            "sentinel_tau": float(cfg.v2.sentinel_tau)}


def check_record(record: Any, cfg: Any) -> str:
    """The recorded suite of a receipt's ``sentinel`` field after its shape
    checks (a dict naming a suite of ``cfg.v2.sentinel_suites``, an int block
    and a string hash); raises ``ValueError`` otherwise."""
    if not isinstance(record, dict):
        raise ValueError(f"sentinel record must be an object; got {record!r}")
    suite, block, h = record.get("suite"), record.get("block"), record.get("block_hash")
    suites = tuple(getattr(cfg.v2, "sentinel_suites", ()) or ())
    if not isinstance(suite, str) or suite not in suites:
        raise ValueError(f"recorded sentinel {suite!r} is not one of sentinel_suites "
                         f"{list(suites)}")
    if isinstance(block, bool) or not isinstance(block, int) or block <= 0:
        raise ValueError(f"recorded sentinel block {block!r} is not a positive integer")
    if not isinstance(h, str):
        raise ValueError(f"recorded sentinel block_hash {h!r} is not a string")
    return suite


def verify_record(record: Any, cfg: Any, block_hash: str | None) -> str:
    """Verify a recorded draw. With the chain's ``block_hash`` for
    ``record["block"]`` (``None`` = unreadable) the hash must equal the
    recorded one and draw the recorded suite; without it the recorded suite
    is replayed (it must still be a configured suite). Returns the suite;
    raises ``ValueError`` on any contradiction."""
    suite = check_record(record, cfg)
    if block_hash is None:
        return suite
    if _hash_hex(str(block_hash)) != _hash_hex(record["block_hash"]):
        raise ValueError(f"block {record['block']} hash {block_hash} != the recorded "
                         f"{record['block_hash']}")
    drawn = sentinel_for_era(str(block_hash), cfg.v2.sentinel_suites)
    if drawn != suite:
        raise ValueError(f"block {record['block']} hash draws sentinel {drawn!r}, the "
                         f"receipt records {suite!r}")
    return suite
