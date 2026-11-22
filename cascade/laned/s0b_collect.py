"""S0b live run: collect and decode the mainnet readiness notes (lane D R2x,
plan §5 row 3 S0b, §14.1, §17.1 L4).

``python -P -m cascade.laned.s0b_collect --network finney --netuid 91
--cfg <cfg|none> --out evidence/s0b_notes.json [--dry-run [--fixture F]]``

Read-only. Through :class:`cascade.shared.chain.ChainClient` (no wallet), all
at one head block: ``current_block``, ``block_hash``, ``validator_stakes``
and ``read_plain_commitments``. For every validator with a permit and stake
it decodes the readiness note and simulates adding the ``cfg-`` segment with
the R2 decoder (``cascade.laned.notes``, worst case, as ``v2b_note_problem``
judges it), then writes the ``laned_s0b_notes/1`` evidence.

``--cfg none`` (before S0c has a seq-1 digest) simulates with the probe name
``cfg-ffffffff``: every ``cfg-<8 hex>`` name is the same length, so the trim
result and the note budget headroom do not depend on the target.

``--dry-run`` opens no connection: it prints the fetch plan, and with
``--fixture`` writes the evidence a snapshot would give (``source:
"fixture"``, never ``passed``).

Exit 0 = no pending segment trimmed and at least one note checked; 1 = a
pending segment would be trimmed, or no note; 2 = usage, input or chain error.
Contract: ``tests/accept/r2x/README.md``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from cascade.laned import notes as R2
from cascade.laned.upgrade_tracker import InputError, parse_target
from cascade.shared.activation import NOTE_MAX_BYTES, parse_signals

SCHEMA = "laned_s0b_notes/1"
WP = "R2x"
MODULE = "cascade.laned.s0b_collect"
# ``v2b_note_problem``'s own budget probe; any cfg-<8 hex> name is 12 bytes.
PROBE_CFG = "cfg-ffffffff"
REPO = Path(__file__).resolve().parents[2]

EXIT_CLEAR = 0
EXIT_TRIMMED = 1
EXIT_ERROR = 2


class CollectError(Exception):
    """Usage, input or chain error: exit 2."""


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        self.print_usage(sys.stderr)
        raise CollectError(f"usage: {message}")


def _parser() -> argparse.ArgumentParser:
    p = _Parser(prog=f"python -P -m {MODULE}", description=__doc__.split("\n\n")[0],
                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--network", required=True, help="subtensor network (finney for mainnet)")
    p.add_argument("--netuid", required=True, type=int)
    p.add_argument("--cfg", required=True,
                   help="cfg-<8 hex>, <8 hex>, the 64-hex overlay body digest, or none")
    p.add_argument("--out", help="evidence JSON to write (required unless --dry-run)")
    p.add_argument("--dry-run", action="store_true", help="no chain connection")
    p.add_argument("--fixture", help="with --dry-run: a snapshot of the four reads")
    return p


def parse_cfg(value: str) -> str | None:
    if value.strip().lower() == "none":
        return None
    try:
        return parse_target(value)
    except InputError as e:
        raise CollectError(str(e)) from None


# ── reads ────────────────────────────────────────────────────────────────────


def fetch_plan(network: str, netuid: int) -> list[str]:
    return [
        f"ChainClient(netuid={netuid}, network={network!r})  # read-only, no wallet",
        "ChainClient.current_block()  -> head_block",
        "ChainClient.block_hash(head_block)",
        f"ChainClient.validator_stakes(head_block)  -> netuid {netuid} metagraph at head_block: "
        "hotkey, stake, validator_permit",
        f"ChainClient.read_plain_commitments(head_block)  -> Commitments::CommitmentOf "
        f"query map for netuid {netuid} at head_block: hotkey -> note",
    ]


def read_chain(network: str, netuid: int) -> dict:
    from cascade.shared.chain import ChainClient, ChainError

    client = ChainClient(netuid=netuid, network=network)
    try:
        head = int(client.current_block())
        block_hash = str(client.block_hash(head))
        stakes = client.validator_stakes(head)
        commitments = client.read_plain_commitments(head)
    except ChainError as e:
        raise CollectError(f"chain read failed: {e}") from None
    return {
        "network": network, "netuid": netuid, "head_block": head,
        "block_hash": block_hash.lower(),
        "fetched_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "validators": [{"hotkey": v.hotkey, "stake": float(v.stake),
                        "validator_permit": bool(v.permit)} for v in stakes],
        "commitments": dict(commitments),
    }


def read_fixture(path: str, network: str, netuid: int) -> dict:
    try:
        snap = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as e:
        raise CollectError(f"--fixture {path}: {e}") from None
    need = ("network", "netuid", "head_block", "block_hash", "fetched_at", "validators",
            "commitments")
    if not isinstance(snap, dict) or any(k not in snap for k in need):
        raise CollectError(f"--fixture {path}: want an object with keys {list(need)}")
    if (snap["network"], snap["netuid"]) != (network, netuid):
        raise CollectError(f"--fixture {path} is for {(snap['network'], snap['netuid'])}, "
                           f"not {(network, netuid)}")
    return snap


# ── evidence ─────────────────────────────────────────────────────────────────


def tool_info() -> dict:
    def git(*args: str) -> subprocess.CompletedProcess:
        return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True,
                              check=False)

    sha = git("rev-parse", "HEAD")
    status = git("status", "--porcelain", "--untracked-files=no")
    return {"module": MODULE,
            "git_sha": sha.stdout.strip() if sha.returncode == 0 else None,
            "git_dirty": bool(status.stdout.strip()) if status.returncode == 0 else True}


def _row(v: dict, note: str | None, head: int, cfg_feature: str) -> dict:
    row = {"hotkey": v["hotkey"], "stake": v["stake"], "validator_permit": True,
           "note_state": "missing", "note_hex": None, "decoded": None, "simulate": None,
           "headroom_bytes": None}
    if note is None:
        return row
    row["note_hex"] = note.encode("utf-8").hex()
    if not parse_signals(note):
        row["note_state"] = "not_a_note"
        return row
    sim = R2.simulate_note(note, head, cfg_feature, 0, 0, worst_case=True)
    row.update(note_state="ok", decoded=R2.decode_note(note, head), simulate=sim,
               headroom_bytes=NOTE_MAX_BYTES - sim["bytes"])
    return row


def build_evidence(snap: dict, target: str | None, source: str) -> dict:
    head = snap["head_block"]
    if isinstance(head, bool) or not isinstance(head, int) or head <= 0:
        raise CollectError(f"head_block must be an int > 0, got {head!r}")
    cfg_feature = target if target is not None else PROBE_CFG
    eligible = [v for v in snap["validators"]
                if v.get("validator_permit") is True and float(v.get("stake", 0)) > 0]
    eligible.sort(key=lambda v: (-float(v["stake"]), v["hotkey"]))
    commitments = snap["commitments"]
    try:
        rows = [_row({"hotkey": str(v["hotkey"]), "stake": float(v["stake"])},
                     commitments.get(v["hotkey"]), head, cfg_feature) for v in eligible]
    except R2.NotesError as e:
        raise CollectError(str(e)) from None
    ok = [r for r in rows if r["note_state"] == "ok"]
    trimmed = [{"hotkey": r["hotkey"], "feature": f}
               for r in ok for f in r["simulate"]["pending_trimmed"]]
    low = min(ok, key=lambda r: (r["headroom_bytes"], r["hotkey"])) if ok else None
    clear = not trimmed and bool(ok)
    return {
        "schema": SCHEMA, "wp": WP, "source": source,
        "network": snap["network"], "netuid": snap["netuid"], "head_block": head,
        "block_hash": snap["block_hash"], "fetched_at": snap["fetched_at"],
        "tool": tool_info(),
        "target_cfg": target,
        "simulate_params": {"cfg_feature": cfg_feature,
                            "cfg_source": "target" if target is not None else "probe",
                            "cfg_lock": 0, "cfg_activation": 0, "worst_case": True},
        "hotkeys": rows,
        "counts": {"permitted": len(rows),
                   **{s: sum(r["note_state"] == s for r in rows)
                      for s in ("ok", "missing", "not_a_note")}},
        "headroom": {"min_bytes": low["headroom_bytes"] if low else None,
                     "min_hotkey": low["hotkey"] if low else None},
        "trimmed_pending": trimmed,
        "passed": clear and source == "chain",
    }


def write_evidence(path: str, evidence: dict) -> str:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, out)
    return str(out)


def summary(evidence: dict) -> dict:
    return {"head_block": evidence["head_block"], "target_cfg": evidence["target_cfg"],
            "counts": evidence["counts"], "headroom": evidence["headroom"],
            "trimmed_pending": evidence["trimmed_pending"]}


def run(args: argparse.Namespace) -> int:
    if args.netuid <= 0:
        raise CollectError("--netuid must be > 0")
    target = parse_cfg(args.cfg)
    report: dict = {"dry_run": bool(args.dry_run), "network": args.network,
                    "netuid": args.netuid, "would_fetch": fetch_plan(args.network, args.netuid),
                    "fixture": args.fixture, "out": None, "gate_clear": None, "passed": False}
    if args.fixture and not args.dry_run:
        raise CollectError("--fixture is only valid with --dry-run")
    if args.dry_run and not args.fixture:
        print(json.dumps(report, indent=2))
        return EXIT_CLEAR
    if not args.out:
        raise CollectError("--out is required (the evidence file to write)")
    if args.dry_run:
        evidence = build_evidence(read_fixture(args.fixture, args.network, args.netuid),
                                  target, "fixture")
    else:
        evidence = build_evidence(read_chain(args.network, args.netuid), target, "chain")
    clear = not evidence["trimmed_pending"] and evidence["counts"]["ok"] >= 1
    report.update(out=write_evidence(args.out, evidence), gate_clear=clear,
                  passed=evidence["passed"], **summary(evidence))
    print(json.dumps(report, indent=2))
    return EXIT_CLEAR if clear else EXIT_TRIMMED


def main(argv: list[str] | None = None) -> int:
    try:
        return run(_parser().parse_args(argv))
    except CollectError as e:
        print(f"s0b_collect: {e}", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
