"""Note-budget decoder (lane D plan §3.0 R2, §5 row 3 S0b).

``python -m cascade.laned.notes decode|simulate ...`` decodes a readiness
note (``cascade-ready:1:<feature>:<lock>:<activation>[...]``) into segments
with their role and pending/decided state, and simulates adding a
``cfg-<8 hex>`` overlay segment under the 128-byte commitment budget. It
reads no chain and no network; the live notes are fed in by the operator.

Parsing, formatting and the budget constants are the real ones from
``cascade/shared/activation.py`` (``parse_signals``, ``format_signals``,
``NOTE_MAX_BYTES``, ``NOTE_BLOCK_HORIZON``). The trim loop mirrors
``own_signal_payload`` / ``note_trimmed_segments``: the primary segment is
never trimmed; extras go newest first; the ``cfg-`` tail goes only once no
extra is left. ``--worst-case`` judges pending segments at the longest
blocks reachable within the horizon, as the startup gate
``v2b_note_problem`` does.

Exit codes: 0 not refused, 1 refused (a pending segment would be trimmed),
2 usage or input error. Contract: ``tests/accept/r2/README.md``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Sequence

from cascade.shared.activation import (
    NOTE_BLOCK_HORIZON,
    NOTE_MAX_BYTES,
    ReadySignal,
    format_signals,
    parse_signals,
)
from cascade.shared.netconfig import FEATURE_PREFIX

# ``netconfig.cfg_feature_name``: the prefix plus the first 8 lowercase hex
# chars of the overlay body's sha256.
CFG_FEATURE_RE = re.compile(re.escape(FEATURE_PREFIX) + r"[0-9a-f]{8}")


class NotesError(ValueError):
    """Unusable input: exit 2."""


# ── decode ───────────────────────────────────────────────────────────────────


def note_from_hex(text: str) -> str:
    raw = text.strip()
    if raw[:2].lower() == "0x":
        raw = raw[2:]
    try:
        data = bytes.fromhex(raw)
    except ValueError as e:
        raise NotesError(f"--hex is not valid hex: {e}") from None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as e:
        raise NotesError(f"--hex bytes are not UTF-8: {e}") from None


def parse_note(note: str) -> tuple[ReadySignal, ...]:
    sigs = parse_signals(note)
    if not sigs:
        raise NotesError(f"not a well-formed v1 readiness note: {note!r}")
    return sigs


def segment_state(sig: ReadySignal, head_block: int) -> str:
    """``decided`` iff locked and the activation block is reached."""
    return ("decided" if sig.lock_block > 0 and sig.activation_block <= head_block
            else "pending")


def segment_role(index: int, sig: ReadySignal) -> str:
    if index == 0:
        return "primary"
    return "cfg" if sig.feature.startswith(FEATURE_PREFIX) else "extra"


def note_bytes(note: str) -> int:
    return len(note.encode("utf-8"))


def decode_note(note: str, head_block: int) -> dict:
    sigs = parse_note(note)
    return {
        "note": note,
        "bytes": note_bytes(note),
        "max_bytes": NOTE_MAX_BYTES,
        "head_block": head_block,
        "segments": [
            {"feature": s.feature, "lock_block": s.lock_block,
             "activation_block": s.activation_block, "role": segment_role(i, s),
             "state": segment_state(s, head_block)}
            for i, s in enumerate(sigs)
        ],
    }


# ── simulate ─────────────────────────────────────────────────────────────────


def worst_case_block(head_block: int) -> int:
    """``v2b_note_problem``'s worst-case block for ``head_block``."""
    digits = max(7, len(str(max(0, int(head_block)) + NOTE_BLOCK_HORIZON)))
    return 10 ** digits - 1


def simulate_note(note: str, head_block: int, cfg_feature: str, cfg_lock: int = 0,
                  cfg_activation: int = 0, *, worst_case: bool = False) -> dict:
    if not CFG_FEATURE_RE.fullmatch(cfg_feature):
        raise NotesError(f"--cfg must match {CFG_FEATURE_RE.pattern}; got {cfg_feature!r}")
    if cfg_lock < 0 or cfg_activation < 0:
        raise NotesError("--cfg-lock / --cfg-activation must be integers >= 0")
    decoded = decode_note(note, head_block)
    sigs = parse_note(note)
    head, rest = sigs[0], sigs[1:]
    new_cfg = ReadySignal(cfg_feature, lock_block=cfg_lock, activation_block=cfg_activation)
    # State and role from the INPUT blocks (the new cfg- from its CLI blocks).
    state = {s.feature: segment_state(s, head_block) for s in rest}
    state[cfg_feature] = segment_state(new_cfg, head_block)
    # The builder carries exactly one cfg- segment: any existing one is replaced.
    extras = [s for s in rest if segment_role(1, s) == "extra"]
    tail = [new_cfg]
    if worst_case:
        big = worst_case_block(head_block)
        extras = [ReadySignal(s.feature, big, big) if state[s.feature] == "pending" else s
                  for s in extras]
        if state[cfg_feature] == "pending":
            tail = [ReadySignal(cfg_feature, big, big)]
    cfg_out = tail[0]
    try:
        out_note = format_signals(head, *extras, *tail)
    except ValueError as e:
        raise NotesError(str(e)) from None
    trimmed: list[dict] = []
    while note_bytes(out_note) > NOTE_MAX_BYTES and (extras or tail):
        dropped, role = (extras.pop(), "extra") if extras else (tail.pop(), "cfg")
        trimmed.append({"feature": dropped.feature, "role": role,
                        "state": state[dropped.feature]})
        out_note = format_signals(head, *extras, *tail)
    pending_trimmed = [t["feature"] for t in trimmed if t["state"] == "pending"]
    return {
        "input": decoded,
        "cfg": {"feature": cfg_out.feature, "lock_block": cfg_out.lock_block,
                "activation_block": cfg_out.activation_block},
        "worst_case": bool(worst_case),
        "note": out_note,
        "bytes": note_bytes(out_note),
        "fits": note_bytes(out_note) <= NOTE_MAX_BYTES,
        "trimmed": trimmed,
        "pending_trimmed": pending_trimmed,
        "refused": bool(pending_trimmed),
    }


# ── CLI ──────────────────────────────────────────────────────────────────────


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m cascade.laned.notes", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp: argparse.ArgumentParser) -> None:
        src = sp.add_mutually_exclusive_group(required=True)
        src.add_argument("--hex", help="the commitment bytes as hex (optional 0x)")
        src.add_argument("--text", help="the note string itself")
        sp.add_argument("--head-block", type=int, required=True)

    common(sub.add_parser("decode", help="decode a note into segments"))
    sim = sub.add_parser("simulate", help="simulate adding a cfg- segment")
    common(sim)
    sim.add_argument("--cfg", required=True, help="cfg-<8 lowercase hex>")
    sim.add_argument("--cfg-lock", type=int, default=0)
    sim.add_argument("--cfg-activation", type=int, default=0)
    sim.add_argument("--worst-case", action="store_true",
                     help="judge pending segments at their longest (v2b_note_problem)")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.head_block < 0:
            raise NotesError("--head-block must be >= 0")
        note = note_from_hex(args.hex) if args.hex is not None else args.text
        if args.command == "decode":
            out = decode_note(note, args.head_block)
            rc = 0
        else:
            out = simulate_note(note, args.head_block, args.cfg, args.cfg_lock,
                                args.cfg_activation, worst_case=args.worst_case)
            rc = 1 if out["refused"] else 0
    except NotesError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(json.dumps(out))
    return rc


if __name__ == "__main__":
    sys.exit(main())
