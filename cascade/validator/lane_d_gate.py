"""LANE_D_PLAN §1.3b "The draw" / §16 rule 2 (lane D): the rotating gate suite.

At each lane D crowning one suite is drawn from the **settlement boundary's
block hash** — the same on every validator, unpredictable at commit time:

- over ``[lane_d] gate_suites`` minus the era's §16 sentinel
  (:func:`cascade.shared.sentinel.sentinel_for_era`; ``None`` = W-S1 unarmed,
  nothing removed);
- weighted by ``[lane_d] gate_weights`` (start equal; a suite without a weight
  entry has weight 0, the sentinel's weight is dropped and the rest are
  renormalised).

The draw is domain-separated from the sentinel draw and the round base seed
(:func:`cascade.shared.chain.seed_from_block_hash`) by its own tagged sha256,
so the gate suite is independent of the era's sentinel even though both come
from block hashes. The 256-bit digest is mapped onto the cumulative weights in
exact rational arithmetic (no float accumulation), so the result depends only
on the hash and the weight ratios.

Pure: no I/O, no process-local state (no ``hash()``). Wiring into
``settle_lane_d`` and the paired sweep is W3b2's job; :func:`gate_record` is
the ``gate`` field of the lane D report (W7a).
"""

from __future__ import annotations

import hashlib
import math
import string
from collections.abc import Iterable, Mapping, Sequence
from fractions import Fraction
from typing import Any

# Its own domain tag: distinct from W-S1's ``cascade|sentinel|v1|`` and from the
# untagged / blake2b mappings of the same hash.
_DRAW_TAG = b"cascade|lane_d|gate_suite|v1|"
_HEX = frozenset(string.hexdigits.lower())
_SPACE = 1 << 256


def _hash_hex(block_hash: object) -> str:
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


def _suites(gate_suites: Iterable[str]) -> tuple[str, ...]:
    if isinstance(gate_suites, (str, bytes)):
        raise TypeError(f"gate_suites must be a sequence of names; got {gate_suites!r}")
    names = tuple(gate_suites)
    for s in names:
        if not isinstance(s, str) or not s:
            raise ValueError(f"gate suite {s!r} is not a non-empty string")
    if len(set(names)) != len(names):
        raise ValueError(f"gate_suites has duplicates: {list(names)}")
    return names


def _weight(suite: str, raw: object) -> Fraction:
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise ValueError(f"gate weight for {suite!r} must be a number; got {raw!r}")
    if not math.isfinite(raw):
        raise ValueError(f"gate weight for {suite!r} must be finite; got {raw!r}")
    if raw < 0:
        raise ValueError(f"gate weight for {suite!r} must be >= 0; got {raw!r}")
    return Fraction(raw)


def _weights(gate_weights: Any, suites: tuple[str, ...]) -> dict[str, Fraction]:
    """``{suite: weight}`` over ``suites`` — equal when ``gate_weights`` is
    None, else from a mapping or ``(suite, weight)`` pairs (absent = 0)."""
    if gate_weights is None:
        return {s: Fraction(1) for s in suites}
    if isinstance(gate_weights, Mapping):
        pairs = list(gate_weights.items())
    elif isinstance(gate_weights, (str, bytes)):
        raise ValueError(f"gate_weights must be a mapping or (suite, weight) pairs; "
                         f"got {gate_weights!r}")
    else:
        pairs = []
        for item in gate_weights:
            if not isinstance(item, Sequence) or isinstance(item, (str, bytes)) \
                    or len(item) != 2:
                raise ValueError(f"gate_weights entry {item!r} is not a (suite, weight) pair")
            pairs.append((item[0], item[1]))
    out: dict[str, Fraction] = {}
    for s, raw in pairs:
        if s not in suites:
            raise ValueError(f"gate_weights names {s!r}, not in gate_suites {list(suites)}")
        if s in out:
            raise ValueError(f"gate_weights names {s!r} twice")
        out[s] = _weight(s, raw)
    return out


def gate_suite_for(boundary_block_hash: str, *, gate_suites: Iterable[str],
                   gate_weights: Any = None, sentinel: str | None = None) -> str:
    """CONSENSUS: the lane D gate suite for the settlement boundary whose block
    hash is ``boundary_block_hash`` (``0x`` + 64 hex digits).

    Eligible suites are ``gate_suites`` in order, minus ``sentinel`` (a
    sentinel outside ``gate_suites`` removes nothing), with positive weight.
    The tagged sha256 of the hash, as a big-endian integer ``d``, selects the
    first eligible suite whose cumulative weight ``C`` satisfies
    ``d * total < C * 2**256`` — exact, so scaling every weight by the same
    factor leaves every draw unchanged. Raises ``ValueError`` / ``TypeError``
    on a malformed hash, a negative / non-finite weight, a weight naming a
    suite outside ``gate_suites``, or an empty eligible set."""
    h = _hash_hex(boundary_block_hash)
    suites = _suites(gate_suites)
    weights = _weights(gate_weights, suites)
    eligible = [(s, weights[s]) for s in suites
                if s != sentinel and weights.get(s, Fraction(0)) > 0]
    if not eligible:
        raise ValueError(f"no eligible gate suite: gate_suites={list(suites)}, "
                         f"sentinel={sentinel!r}, positive weights="
                         f"{[s for s in suites if weights.get(s, Fraction(0)) > 0]}")
    total = sum((w for _, w in eligible), Fraction(0))
    d = int.from_bytes(hashlib.sha256(_DRAW_TAG + h.encode("ascii")).digest(), "big")
    point = Fraction(d, _SPACE) * total
    acc = Fraction(0)
    for s, w in eligible:
        acc += w
        if point < acc:
            return s
    # Unreachable: d < 2**256 ⇒ point < total == acc after the last suite.
    raise AssertionError("gate draw fell past the cumulative total")


def gate_record(cfg: Any, block: int, block_hash: str, *,
                sentinel: str | None = None) -> dict:
    """The ``gate`` record of a lane D settlement: ``{"suite", "block",
    "block_hash", "sentinel", "eligible"}`` for the boundary ``block`` (a
    non-negative ``int``) with hash ``block_hash``, drawn over ``cfg``'s
    (a ``LaneDConfig``) ``gate_suites`` / ``gate_weights``. JSON
    round-trippable. Raises ``TypeError`` / ``ValueError`` on a bad block,
    a malformed hash or an empty eligible set."""
    if isinstance(block, bool) or not isinstance(block, int):
        raise TypeError(f"boundary block must be an int; got {block!r}")
    if block < 0:
        raise ValueError(f"boundary block must be >= 0; got {block!r}")
    suites = _suites(cfg.gate_suites)
    suite = gate_suite_for(block_hash, gate_suites=suites, gate_weights=cfg.gate_weights,
                           sentinel=sentinel)
    weights = _weights(cfg.gate_weights, suites)
    eligible = [s for s in suites if s != sentinel and weights.get(s, Fraction(0)) > 0]
    return {"suite": suite, "block": int(block), "block_hash": str(block_hash),
            "sentinel": None if sentinel is None else str(sentinel), "eligible": eligible}
