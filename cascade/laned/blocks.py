"""``cascade-laned blocks``: block arithmetic from the live grid, cross-checked.

Every activation, effective and era-start block the lane D rollout quotes is
computed here, never by hand (plan §5 "Block arithmetic"; the past off-by-ones
are in §13). Each answer is computed twice:

1. :class:`SharedComputation` runs the real consensus helpers
   (``cascade.shared.era`` / ``netconfig`` / ``activation`` / ``chain``) on the
   grid's ``[round]`` keys, deriving the grid in force AT the queried block
   (``effective_epoch_blocks``), never at the head.
2. :class:`SegmentWalk` ignores ``[round]`` and those helpers entirely: it finds
   the observed chain segment containing the block and does absolute grid
   arithmetic (``block // length``) with that segment's lengths.

The answers must be identical, field for field, or the tool refuses (exit 3,
``mismatch``). ``--config chain.toml`` additionally refuses a ``[round]`` that
drifted from the live grid (exit 3, ``drift``).

Exit codes: 0 answered (one JSON object on stdout); 2 usage error or invalid
input; 3 cross-check refusal. Nothing is printed on stdout unless exit is 0.
"""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from cascade.shared import activation as shared_activation
from cascade.shared import era as shared_era
from cascade.shared import netconfig as shared_netconfig
from cascade.shared.chain import blocks_until_boundary_reveal
from cascade.shared.config import RoundConfig, effective_epoch_blocks

GRID_SCHEMA = "laned_grid/1"
ROUND_KEYS_REQUIRED = ("epoch_blocks", "epoch_blocks_prev", "epoch_activation_block",
                       "era_settlements")
ROUND_KEYS_OPTIONAL = ("reveal_margin_blocks",)
ROUND_KEYS = ROUND_KEYS_REQUIRED + ROUND_KEYS_OPTIONAL
SEGMENT_KEYS = ("from_block", "epoch_blocks", "era_blocks")
ALIGNS = ("boundary", "era", "epoch")

EXIT_OK = 0
EXIT_INVALID = 2
EXIT_REFUSED = 3


class InvalidInput(Exception):
    """Bad fixture, bad argument, or a query the grid cannot answer (exit 2)."""


class CrossCheckRefused(Exception):
    """The two computations, or chain.toml and the grid, disagree (exit 3)."""


# ── the live grid ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Segment:
    from_block: int
    epoch_blocks: int
    era_blocks: int


@dataclass(frozen=True)
class Grid:
    round: dict[str, int]
    segments: tuple[Segment, ...]


def _strict_int(value: Any, where: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidInput(f"{where} must be an integer, got {value!r}")
    if value < minimum:
        raise InvalidInput(f"{where} must be >= {minimum}, got {value}")
    return value


def parse_grid(obj: Any) -> Grid:
    """Validate a ``laned_grid/1`` object (README "The live grid")."""
    if not isinstance(obj, dict):
        raise InvalidInput("grid must be a JSON object")
    if obj.get("schema") != GRID_SCHEMA:
        raise InvalidInput(f"grid schema must be {GRID_SCHEMA!r}, got {obj.get('schema')!r}")
    if "round" not in obj or not isinstance(obj["round"], dict):
        raise InvalidInput("grid has no 'round' object")
    if "segments" not in obj or not isinstance(obj["segments"], list) or not obj["segments"]:
        raise InvalidInput("grid has no non-empty 'segments' list")
    raw_round = obj["round"]
    rnd: dict[str, int] = {}
    for key in ROUND_KEYS_REQUIRED:
        if key not in raw_round:
            raise InvalidInput(f"grid round is missing {key!r}")
        rnd[key] = _strict_int(raw_round[key], f"round.{key}",
                               minimum=1 if key == "epoch_blocks" else 0)
    for key in ROUND_KEYS_OPTIONAL:
        if key in raw_round:
            rnd[key] = _strict_int(raw_round[key], f"round.{key}", minimum=0)
    segments: list[Segment] = []
    for i, raw_seg in enumerate(obj["segments"]):
        if not isinstance(raw_seg, dict):
            raise InvalidInput(f"segments[{i}] must be an object")
        for key in SEGMENT_KEYS:
            if key not in raw_seg:
                raise InvalidInput(f"segments[{i}] is missing {key!r}")
        seg = Segment(
            from_block=_strict_int(raw_seg["from_block"], f"segments[{i}].from_block", minimum=0),
            epoch_blocks=_strict_int(raw_seg["epoch_blocks"], f"segments[{i}].epoch_blocks",
                                     minimum=1),
            era_blocks=_strict_int(raw_seg["era_blocks"], f"segments[{i}].era_blocks", minimum=1),
        )
        if seg.era_blocks % seg.epoch_blocks:
            raise InvalidInput(f"segments[{i}].era_blocks {seg.era_blocks} is not a whole "
                               f"number of {seg.epoch_blocks}-block settlements")
        if i == 0 and seg.from_block != 0:
            raise InvalidInput(f"segments[0].from_block must be 0, got {seg.from_block}")
        if segments and seg.from_block <= segments[-1].from_block:
            raise InvalidInput(f"segments[{i}].from_block {seg.from_block} is not after "
                               f"segments[{i - 1}].from_block {segments[-1].from_block}")
        segments.append(seg)
    return Grid(round=rnd, segments=tuple(segments))


def load_grid(path: Path) -> Grid:
    try:
        obj = json.loads(path.read_text())
    except OSError as e:
        raise InvalidInput(f"cannot read grid {path}: {e}") from e
    except json.JSONDecodeError as e:
        raise InvalidInput(f"grid {path} is not JSON: {e}") from e
    return parse_grid(obj)


def check_config_drift(grid: Grid, config_path: Path) -> None:
    """chain.toml ``[round]`` must equal the live grid's ``round`` for every
    grid key present in both; any difference is drift (exit 3)."""
    try:
        raw = tomllib.loads(config_path.read_text())
    except OSError as e:
        raise InvalidInput(f"cannot read config {config_path}: {e}") from e
    except tomllib.TOMLDecodeError as e:
        raise InvalidInput(f"config {config_path} is not TOML: {e}") from e
    table = raw.get("round")
    if not isinstance(table, dict):
        raise InvalidInput(f"config {config_path} has no [round] table")
    drift = [f"[round] {key}: chain.toml {table[key]!r} != live grid {grid.round[key]!r}"
             for key in ROUND_KEYS
             if key in table and key in grid.round and table[key] != grid.round[key]]
    if drift:
        raise CrossCheckRefused(f"drift between {config_path} and the live grid: "
                                + "; ".join(drift))


# ── computation 1: the shared consensus code on [round] ──────────────────────


class SharedComputation:
    """Answers from ``cascade.shared`` on the grid's ``round`` keys."""

    def __init__(self, rnd: dict[str, int]) -> None:
        self.rc = RoundConfig(
            epoch_blocks=rnd["epoch_blocks"],
            epoch_blocks_prev=rnd["epoch_blocks_prev"],
            epoch_activation_block=rnd["epoch_activation_block"],
            era_settlements=rnd["era_settlements"],
        )
        # netconfig.is_era_start reads only ``cfg.round``.
        self.cfg = SimpleNamespace(round=self.rc)

    def is_boundary(self, block: int) -> bool:
        return block % int(effective_epoch_blocks(self.rc, block)) == 0

    def is_era_start(self, block: int) -> bool:
        return bool(shared_netconfig.is_era_start(self.cfg, block))

    def era_start(self, block: int) -> dict[str, Any]:
        return {
            "epoch_blocks": int(effective_epoch_blocks(self.rc, block)),
            "era_blocks": int(shared_era.era_length_blocks(self.rc, block)),
            "era_index": int(shared_era.era_index_of(self.rc, block)),
            "era_start": int(shared_era.era_start_block(self.rc, block)),
            "is_era_start": self.is_era_start(block),
        }

    def next_era_start(self, block: int) -> dict[str, Any]:
        nxt = int(shared_era.next_era_start(self.rc, block))
        return {"next_era_start": nxt,
                "next_era_index": int(shared_era.era_index_of(self.rc, nxt))}

    def settlement(self, boundary: int) -> dict[str, Any]:
        spec = shared_era.settlement_era(self.rc, boundary)
        return {
            "era_index": int(shared_era.era_index_of(self.rc, boundary)),
            "settlement_era": int(spec.index),
            "settlement_era_start": int(spec.start_block),
            "is_era_start": self.is_era_start(boundary),
        }

    def activation(self, lock_block: int, align: str) -> dict[str, Any]:
        return {"activation_block":
                int(shared_activation.activation_block_for(self.rc, lock_block, align))}

    def effective_block(self, effective_block: int, lock_block: int) -> dict[str, Any]:
        record = shared_activation.ActivationRecord(lock_block=lock_block)
        apply = shared_netconfig.apply_block_for({"effective_block": effective_block},
                                                 record, self.rc)
        return {"apply_block": int(apply)}

    def min_effective_era(self, block: int) -> dict[str, Any]:
        e = int(shared_era.min_effective_era(self.rc, block))
        return {"min_effective_era": e,
                "effective_block": e * int(shared_era.era_length_blocks(self.rc, block))}

    def promotion_fire(self, check_boundary: int, fire_block: int) -> dict[str, Any]:
        eff = int(shared_era.min_effective_era(self.rc, check_boundary))
        at_fire = int(shared_era.min_effective_era(self.rc, fire_block))
        return {"effective_era": eff, "min_effective_era_at_fire": at_fire,
                "fires": eff >= at_fire}

    def reveal(self, block: int, margin: int, next_epoch: bool) -> dict[str, Any]:
        eb = int(effective_epoch_blocks(self.rc, block))
        try:
            delay = int(blocks_until_boundary_reveal(block, eb, margin, next_epoch=next_epoch))
        except ValueError as e:
            raise InvalidInput(f"reveal: {e}") from e
        reveal_block = block + delay
        return {"blocks_until_reveal": delay, "reveal_block": reveal_block,
                "boundary": (reveal_block // eb + 1) * eb}


# ── computation 2: an independent walk over the observed segments ────────────


class SegmentWalk:
    """Answers from the observed grid segments alone (no ``cascade.shared``)."""

    def __init__(self, segments: tuple[Segment, ...]) -> None:
        self.segments = segments

    def segment_at(self, block: int) -> Segment:
        found = self.segments[0]
        for seg in self.segments:
            if seg.from_block > block:
                break
            found = seg
        return found

    def _era(self, block: int) -> int:
        return self.segment_at(block).era_blocks

    def _epoch(self, block: int) -> int:
        return self.segment_at(block).epoch_blocks

    def is_boundary(self, block: int) -> bool:
        return block % self._epoch(block) == 0

    def is_era_start(self, block: int) -> bool:
        return block > 0 and block % self._era(block) == 0

    def _next_era_start(self, block: int) -> int:
        length = self._era(block)
        return (block // length + 1) * length

    def _min_effective_era(self, block: int) -> int:
        # created inside era e ⇒ the first era with a whole era of notice is e + 2
        return block // self._era(block) + 2

    def era_start(self, block: int) -> dict[str, Any]:
        seg = self.segment_at(block)
        return {
            "epoch_blocks": seg.epoch_blocks,
            "era_blocks": seg.era_blocks,
            "era_index": block // seg.era_blocks,
            "era_start": (block // seg.era_blocks) * seg.era_blocks,
            "is_era_start": self.is_era_start(block),
        }

    def next_era_start(self, block: int) -> dict[str, Any]:
        nxt = self._next_era_start(block)
        return {"next_era_start": nxt, "next_era_index": nxt // self._era(nxt)}

    def settlement(self, boundary: int) -> dict[str, Any]:
        judged = max(0, boundary - 1)
        judged_era = self._era(judged)
        return {
            "era_index": boundary // self._era(boundary),
            "settlement_era": judged // judged_era,
            "settlement_era_start": (judged // judged_era) * judged_era,
            "is_era_start": self.is_era_start(boundary),
        }

    def activation(self, lock_block: int, align: str) -> dict[str, Any]:
        if align == "boundary":
            eb = self._epoch(lock_block)
            act = (lock_block // eb + 1) * eb
        elif align == "era":
            act = self._next_era_start(lock_block)
        else:
            act = self._min_effective_era(lock_block) * self._era(lock_block)
        return {"activation_block": act}

    def effective_block(self, effective_block: int, lock_block: int) -> dict[str, Any]:
        if lock_block < effective_block:
            return {"apply_block": effective_block}
        return {"apply_block": self._next_era_start(lock_block)}

    def min_effective_era(self, block: int) -> dict[str, Any]:
        e = self._min_effective_era(block)
        return {"min_effective_era": e, "effective_block": e * self._era(block)}

    def promotion_fire(self, check_boundary: int, fire_block: int) -> dict[str, Any]:
        eff = self._min_effective_era(check_boundary)
        at_fire = self._min_effective_era(fire_block)
        return {"effective_era": eff, "min_effective_era_at_fire": at_fire,
                "fires": eff >= at_fire}

    def reveal(self, block: int, margin: int, next_epoch: bool) -> dict[str, Any]:
        eb = self._epoch(block)
        if not 0 <= margin < eb:
            raise InvalidInput(f"reveal: margin {margin} must be in [0, {eb})")
        target = (block // eb + 1) * eb + (eb if next_epoch else 0)
        reveal_block = block + max(1, target - margin - block)
        return {"blocks_until_reveal": reveal_block - block, "reveal_block": reveal_block,
                "boundary": (reveal_block // eb + 1) * eb}


# ── cross-check and queries ──────────────────────────────────────────────────


def cross_check(what: str, compute: Callable[[Any], Any],
                shared: SharedComputation, walk: SegmentWalk) -> Any:
    """Run ``compute`` on both computations; refuse unless they agree."""
    a = compute(shared)
    b = compute(walk)
    if a != b:
        raise CrossCheckRefused(
            f"mismatch in {what}: shared code on [round] says {a!r}, "
            f"the segment walk says {b!r}")
    return a


def answer(args: argparse.Namespace, grid: Grid) -> dict[str, Any]:
    shared = SharedComputation(grid.round)
    walk = SegmentWalk(grid.segments)

    def check(what: str, compute: Callable[[Any], Any]) -> Any:
        return cross_check(what, compute, shared, walk)

    q = args.query
    out: dict[str, Any] = {"query": q}
    if q == "era-start":
        out["block"] = args.block
        out.update(check("era-start", lambda c: c.era_start(args.block)))
    elif q == "next-era-start":
        out["block"] = args.block
        out.update(check("next-era-start", lambda c: c.next_era_start(args.block)))
    elif q == "settlement":
        if not check("boundary test", lambda c: c.is_boundary(args.boundary)):
            raise InvalidInput(f"{args.boundary} is not a settlement boundary of the grid "
                               f"in force there")
        out["boundary"] = args.boundary
        out.update(check("settlement", lambda c: c.settlement(args.boundary)))
    elif q == "activation":
        out["lock_block"] = args.lock_block
        out["align"] = args.align
        out.update(check("activation", lambda c: c.activation(args.lock_block, args.align)))
    elif q == "effective-block":
        if not check("era-start test", lambda c: c.is_era_start(args.effective_block)):
            raise InvalidInput(f"effective_block {args.effective_block} is not an era start")
        out["effective_block"] = args.effective_block
        out["lock_block"] = args.lock_block
        out.update(check("effective-block",
                         lambda c: c.effective_block(args.effective_block, args.lock_block)))
    elif q == "min-effective-era":
        out["block"] = args.block
        out.update(check("min-effective-era", lambda c: c.min_effective_era(args.block)))
    elif q == "promotion-fire":
        out["check_boundary"] = args.check_boundary
        out["fire_block"] = args.fire_block
        out.update(check("promotion-fire",
                         lambda c: c.promotion_fire(args.check_boundary, args.fire_block)))
    elif q == "reveal":
        margin = args.margin if args.margin is not None else grid.round.get("reveal_margin_blocks")
        if margin is None:
            raise InvalidInput("reveal: no --margin and the grid has no "
                               "round.reveal_margin_blocks")
        out["block"] = args.block
        out["margin_blocks"] = margin
        out.update(check("reveal", lambda c: c.reveal(args.block, margin, args.next_epoch)))
    else:
        raise InvalidInput(f"unknown query {q!r}")
    out["cross_checked"] = True
    return out


def run(args: argparse.Namespace) -> int:
    try:
        grid = load_grid(Path(args.grid))
        if args.config:
            check_config_drift(grid, Path(args.config))
        out = answer(args, grid)
    except InvalidInput as e:
        print(f"cascade-laned blocks: invalid input: {e}", file=sys.stderr)
        return EXIT_INVALID
    except CrossCheckRefused as e:
        print(f"cascade-laned blocks: refused: {e}", file=sys.stderr)
        return EXIT_REFUSED
    print(json.dumps(out, sort_keys=True))
    return EXIT_OK


# ── argument parsing ─────────────────────────────────────────────────────────


def _block(text: str) -> int:
    try:
        value = int(text)
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"not an integer block: {text!r}") from e
    if value < 0:
        raise argparse.ArgumentTypeError(f"block must be >= 0, got {value}")
    return value


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--grid", required=True, help="live grid JSON (schema laned_grid/1)")
    p.add_argument("--config", help="chain.toml whose [round] must match the grid")


def add_parser(subparsers: Any) -> argparse.ArgumentParser:
    """Register ``blocks`` and its queries on the ``cascade-laned`` parser."""
    blocks = subparsers.add_parser(
        "blocks", help="block arithmetic from the live grid, cross-checked twice")
    queries = blocks.add_subparsers(dest="query", required=True)

    p = queries.add_parser("era-start", help="the era containing a block")
    p.add_argument("block", type=_block)
    _common(p)

    p = queries.add_parser("next-era-start", help="first era start strictly after a block")
    p.add_argument("block", type=_block)
    _common(p)

    p = queries.add_parser("settlement", help="the era a settlement boundary is judged in")
    p.add_argument("boundary", type=_block)
    _common(p)

    p = queries.add_parser("activation", help="rollover block for a lock-in")
    p.add_argument("--lock-block", type=_block, required=True)
    p.add_argument("--align", choices=ALIGNS, required=True)
    _common(p)

    p = queries.add_parser("effective-block", help="apply block of an overlay")
    p.add_argument("--effective-block", type=_block, required=True)
    p.add_argument("--lock-block", type=_block, required=True)
    _common(p)

    p = queries.add_parser("min-effective-era", help="earliest era a record may take effect")
    p.add_argument("block", type=_block)
    _common(p)

    p = queries.add_parser("promotion-fire", help="whether a promotion check still fires")
    p.add_argument("--check-boundary", type=_block, required=True)
    p.add_argument("--fire-block", type=_block, required=True)
    _common(p)

    p = queries.add_parser("reveal", help="timelock reveal delay before the next boundary")
    p.add_argument("block", type=_block)
    p.add_argument("--margin", type=_block)
    p.add_argument("--next-epoch", action="store_true")
    _common(p)

    blocks.set_defaults(func=run)
    return blocks
