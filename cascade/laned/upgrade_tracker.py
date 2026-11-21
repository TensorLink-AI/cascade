"""Stake-weighted upgrade tracker for a ``cfg-`` overlay (lane D R4, plan §5
S6 and X6a).

``python -P -m cascade.laned.upgrade_tracker --metagraph F --notes G --cfg X
[--threshold T]`` reads a metagraph (``{hotkey: {"stake", "validator_permit"}}``)
and the readiness notes read elsewhere (``{hotkey: note | null}``), and reports
per hotkey the release (primary segment), the ``cfg-`` segment and whether it
is on the target overlay; then the exact stake-weighted share of eligible
validator stake on the target, and the laggards by stake.

The note grammar is ``activation.parse_signals``; the eligible and signed sets
come from the real ``activation.tally``; the target name is
``netconfig.cfg_feature_name``. The tracker only adds exact (``Fraction``)
arithmetic over decimal stakes taken as written.

Stdout is one JSON object. Exit 0 = share >= threshold, 1 = below, 2 = bad
input (message on stderr). It reads no chain and no network.
Contract: ``tests/accept/r4/README.md``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from cascade.shared import activation as A
from cascade.shared import netconfig as N

EXIT_MEETS = 0
EXIT_BELOW = 1
EXIT_BAD_INPUT = 2

DEFAULT_THRESHOLD = "0.6667"
PCT_DECIMALS = 4

_HEX8 = re.compile(r"[0-9a-f]{8}")
_HEX64 = re.compile(r"[0-9a-f]{64}")

NOTE_OK = "ok"
NOTE_MALFORMED = "malformed"
NOTE_MISSING = "missing"


class InputError(Exception):
    """Bad input: an unreadable file, invalid JSON, a wrong shape or type, a bad
    flag value, or no eligible stake."""


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        self.print_usage(sys.stderr)
        raise InputError(f"usage: {message}")


def _parser() -> argparse.ArgumentParser:
    p = _Parser(prog="python -P -m cascade.laned.upgrade_tracker",
                description="Stake-weighted adoption of a cfg- overlay among eligible validators.")
    p.add_argument("--metagraph", required=True,
                   help='JSON {"<hotkey>": {"stake": <number>, "validator_permit": <bool>}}')
    p.add_argument("--notes", required=True, help='JSON {"<hotkey>": "<note>" | null}')
    p.add_argument("--cfg", required=True,
                   help="cfg-<8 hex>, <8 hex>, or the 64-hex overlay body digest")
    p.add_argument("--threshold", default=DEFAULT_THRESHOLD,
                   help=f"decimal in (0, 1], default {DEFAULT_THRESHOLD}")
    return p


# ── inputs ───────────────────────────────────────────────────────────────────


def _reject_constant(name: str):
    raise InputError(f"non-finite JSON number {name} is not allowed")


def _load_json(path: str, what: str):
    try:
        text = Path(path).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        raise InputError(f"{what}: cannot read {path}: {e}") from None
    try:
        return json.loads(text, parse_float=Fraction, parse_constant=_reject_constant)
    except json.JSONDecodeError as e:
        raise InputError(f"{what}: invalid JSON in {path}: {e}") from None


@dataclass(frozen=True)
class Row:
    hotkey: str
    stake: Fraction
    permit: bool


def parse_metagraph(raw: object) -> list[Row]:
    if not isinstance(raw, dict):
        raise InputError("metagraph: top level must be a JSON object keyed by hotkey")
    rows = []
    for hk, row in raw.items():
        if not isinstance(row, dict):
            raise InputError(f"metagraph: row for {hk!r} must be an object")
        if "stake" not in row:
            raise InputError(f"metagraph: row for {hk!r} has no stake")
        if "validator_permit" not in row:
            raise InputError(f"metagraph: row for {hk!r} has no validator_permit")
        stake = row["stake"]
        if isinstance(stake, bool) or not isinstance(stake, (int, Fraction)):
            raise InputError(f"metagraph: stake for {hk!r} must be a JSON number, got {stake!r}")
        stake = Fraction(stake)
        if stake < 0:
            raise InputError(f"metagraph: stake for {hk!r} is negative ({stake})")
        permit = row["validator_permit"]
        if not isinstance(permit, bool):
            raise InputError(f"metagraph: validator_permit for {hk!r} must be a JSON bool")
        rows.append(Row(hotkey=hk, stake=stake, permit=permit))
    return rows


def parse_notes(raw: object) -> dict[str, str | None]:
    if not isinstance(raw, dict):
        raise InputError("notes: top level must be a JSON object keyed by hotkey")
    for hk, note in raw.items():
        if note is not None and not isinstance(note, str):
            raise InputError(f"notes: note for {hk!r} must be a string or null")
    return dict(raw)


def parse_target(value: str) -> str:
    v = str(value)
    if v.startswith(N.FEATURE_PREFIX) and _HEX8.fullmatch(v[len(N.FEATURE_PREFIX):]):
        return v
    if _HEX8.fullmatch(v):
        return N.FEATURE_PREFIX + v
    if _HEX64.fullmatch(v):
        return N.cfg_feature_name(v)
    raise InputError(f"--cfg: want cfg-<8 lowercase hex>, <8 lowercase hex> or a 64-hex "
                     f"body digest; got {v!r}")


def parse_threshold(value: str) -> Fraction:
    try:
        t = Fraction(str(value).strip())
    except (ValueError, ZeroDivisionError):
        raise InputError(f"--threshold: not a decimal number: {value!r}") from None
    if not (0 < t <= 1):
        raise InputError(f"--threshold: must be in (0, 1]; got {value!r}")
    return t


# ── per-hotkey report ────────────────────────────────────────────────────────


def _number(x: Fraction) -> int | float:
    return int(x) if x.denominator == 1 else float(x)


def note_view(note: str | None, target: str) -> dict:
    """``note_state``, ``release``, ``cfg`` and ``on_target`` for one note."""
    if note is None:
        return {"note_state": NOTE_MISSING, "release": None, "cfg": None, "on_target": False}
    segments = A.parse_signals(note)
    if not segments:
        return {"note_state": NOTE_MALFORMED, "release": None, "cfg": None, "on_target": False}
    on_target = A.signal_for(note, target) is not None
    cfg = target if on_target else next(
        (s.feature for s in segments if s.feature.startswith(N.FEATURE_PREFIX)), None)
    return {"note_state": NOTE_OK, "release": segments[0].feature, "cfg": cfg,
            "on_target": on_target}


def track(rows: list[Row], notes: dict[str, str | None], target: str,
          threshold: Fraction) -> dict:
    validators = [A.ValidatorStake(hotkey=r.hotkey, stake=r.stake, permit=r.permit) for r in rows]
    signals = {hk: n for hk, n in notes.items() if n is not None}
    t = A.tally(target, validators, signals, threshold=float(threshold), block=0)
    stake_of = {r.hotkey: r.stake for r in rows}
    eligible = set(t.eligible)
    total = sum((stake_of[hk] for hk in t.eligible), Fraction(0))
    signed = sum((stake_of[hk] for hk in t.signed), Fraction(0))
    if total == 0:
        raise InputError("no eligible stake: no hotkey has validator_permit and stake > 0")
    ratio = signed / total

    ordered = sorted(rows, key=lambda r: (-r.stake, r.hotkey))
    hotkeys = []
    laggards = []
    for r in ordered:
        view = note_view(notes.get(r.hotkey), target)
        counted = r.hotkey in eligible
        hotkeys.append({"hotkey": r.hotkey, "stake": _number(r.stake),
                        "validator_permit": r.permit, "counted": counted, **view})
        if counted and r.hotkey not in t.signed:
            laggards.append({"hotkey": r.hotkey, "stake": _number(r.stake),
                             "note_state": view["note_state"], "release": view["release"],
                             "cfg": view["cfg"]})
    return {
        "target": target,
        "threshold": float(threshold),
        "hotkeys": hotkeys,
        "signed_stake": _number(signed),
        "total_stake": _number(total),
        "ratio": str(ratio),
        "pct": float(round(100 * ratio, PCT_DECIMALS)),
        "meets_threshold": ratio >= threshold,
        "signed": list(t.signed),
        "eligible": list(t.eligible),
        "laggards": laggards,
    }


def main(argv: list[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
        target = parse_target(args.cfg)
        threshold = parse_threshold(args.threshold)
        rows = parse_metagraph(_load_json(args.metagraph, "metagraph"))
        notes = parse_notes(_load_json(args.notes, "notes"))
        report = track(rows, notes, target, threshold)
    except InputError as e:
        print(f"upgrade_tracker: {e}", file=sys.stderr)
        return EXIT_BAD_INPUT
    print(json.dumps(report, indent=2))
    return EXIT_MEETS if report["meets_threshold"] else EXIT_BELOW


if __name__ == "__main__":
    sys.exit(main())
