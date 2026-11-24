"""Lane D recurring ops jobs: ``python -P -m cascade.lane_d.ops_jobs <job> ...`` (plan §10.4
per-settlement runbook, §10.7 monitoring, §1.4 house, §1.2 retention, §11.1).

Every job is one idempotent command: re-running it with nothing new is a
no-op, ``--dry-run`` prints what it would write and writes nothing, and it
never reads the chain, rents, deploys or touches a bucket (inputs are local
files the operator or another job already fetched). The systemd TEMPLATES in
``deploy/laned/systemd/`` schedule them; nothing here installs anything.

Jobs:

* ``house-pointer``: re-sign the W5f house pointer when lane A's king
  changes. The lane A king pointer is read from a local JSON (the published
  ``status/v2.json``, ``king.pointer`` by default). The house checkpoint for
  that king (its lane D manifest digest, D47's BF16 cast, and D46's
  ``reveal_ts``) comes from a ``laned_house_source/1`` file the owner prepares
  per king; without one for the CURRENT king the job exits 3 (owner step).
  The pointer is signed with the validator hotkey through
  :func:`cascade.validator.lane_d_house.sign_house_pointer`.
* ``evidence``: validate a new ``laned_readiness/1`` (W1 ``--readiness-out``)
  and/or ``laned_t7/1`` (the T7 pod run, an owner step) and install it
  atomically at the path the validator's ``--lane-d-inputs`` TOML names.
  A document that does not parse, a T7 that is not pod evidence, or one that
  disarms a currently armed context is refused (``--allow-disarm`` overrides
  the last).
* ``vault-ledger``: export the ``laned_vault_ledger/1`` the W18 sweeper reads
  (:func:`cascade.validator.lane_d_status.build_vault_ledger`) from
  ``lane_d_state.json`` and its persisted reports.
* ``king-check``: every crowned persisted report against the signed lane D
  index (and, given local copies, the vault and the mirrors): §11.1 "one
  digest, one identity" via :func:`cascade.lane_d.publish.check_one_digest`.

Exit codes (every job): 0 done / nothing to do / dry-run planned; 1 an input
is unreadable or invalid, or a check found a problem (nothing written);
2 usage; 3 blocked on an owner step.

Log lines (stderr, ``%(asctime)s %(levelname)s %(name)s: %(message)s``, the
cascade-ops follower's format) start with a ``lane_d_ops_*`` tag listed in
:data:`LOG_TAGS`. A JSON summary goes to stdout.

Light imports only (no torch); numpy is loaded by ``evidence`` alone.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from pathlib import Path

log = logging.getLogger("cascade.lane_d.ops")

EXIT_OK, EXIT_FAIL, EXIT_USAGE, EXIT_BLOCKED = 0, 1, 2, 3
HOUSE_SOURCE_SCHEMA = "laned_house_source/1"
HOUSE_SOURCE_FIELDS = ("schema", "king_pointer", "digest", "reveal_ts")
SIDECAR_SUFFIX = ".source.json"

# tag -> level (the follower patterns the owner adds to cascade-ops)
LOG_TAGS: dict[str, str] = {
    "lane_d_ops_house_current": "info",
    "lane_d_ops_house_no_lane_a_king": "info",
    "lane_d_ops_house_resign_planned": "info",
    "lane_d_ops_house_resigned": "warning",
    "lane_d_ops_house_resign_blocked": "error",
    "lane_d_ops_house_failed": "error",
    "lane_d_ops_evidence_current": "info",
    "lane_d_ops_evidence_planned": "info",
    "lane_d_ops_evidence_installed": "warning",
    "lane_d_ops_evidence_refused": "error",
    "lane_d_ops_ledger_current": "info",
    "lane_d_ops_ledger_planned": "info",
    "lane_d_ops_ledger_written": "info",
    "lane_d_ops_ledger_failed": "error",
    "lane_d_ops_king_ok": "info",
    "lane_d_ops_king_unpublished": "critical",
    "lane_d_ops_king_check_failed": "error",
}


class JobError(Exception):
    """A job input is unreadable or invalid: exit 1, nothing written."""


class JobBlocked(Exception):
    """The job needs an owner step first: exit 3, nothing written."""


def _read_json(path: Path, what: str):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise JobError(f"{what} {path} is absent") from None
    except (OSError, UnicodeDecodeError, ValueError) as e:
        raise JobError(f"{what} {path} is unreadable ({type(e).__name__})") from None


def _write_atomic(path: Path, data: bytes) -> None:
    from ..shared.atomic import write_text_atomic

    write_text_atomic(path, data.decode("utf-8"))


def _field(doc, dotted: str):
    cur = doc
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            raise JobError(f"the king source has no {dotted!r}")
        cur = cur[part]
    return cur


# ── house-pointer ────────────────────────────────────────────────────────────


def _house_source(path: Path | None, king: str) -> dict:
    from ..validator.lane_d_house import LaneDHouseError, check_house

    if path is None:
        raise JobBlocked(f"lane A king changed to {king}: no --house-source given (prepare the "
                         f"{HOUSE_SOURCE_SCHEMA} for it: BF16 cast digest and D46 reveal_ts)")
    if not Path(path).exists():
        raise JobBlocked(f"lane A king changed to {king}: {path} is absent (prepare the "
                         f"{HOUSE_SOURCE_SCHEMA} for it)")
    src = _read_json(path, "house source")
    if not isinstance(src, dict) or sorted(src) != sorted(HOUSE_SOURCE_FIELDS) \
            or src.get("schema") != HOUSE_SOURCE_SCHEMA:
        raise JobError(f"{path} is not exactly {HOUSE_SOURCE_SCHEMA} {list(HOUSE_SOURCE_FIELDS)}")
    if src["king_pointer"] != king:
        raise JobBlocked(f"{path} is for lane A king {src['king_pointer']!r}, not the current "
                         f"{king!r} (prepare the house source for the new king)")
    try:
        house = check_house({"digest": src["digest"], "reveal_ts": src["reveal_ts"]})
    except LaneDHouseError as e:
        raise JobError(f"{path}: {e}") from None
    return {"schema": HOUSE_SOURCE_SCHEMA, "king_pointer": king, **house}


def house_pointer(args) -> dict:
    from ..validator.lane_d_house import (
        HOUSE_SCHEMA,
        LaneDHouseError,
        canonical_body,
        load_house_pointer,
        sign_house_pointer,
        verify_house_pointer,
    )

    king = _field(_read_json(args.king_source, "king source"), args.king_field)
    if not isinstance(king, str):
        raise JobError(f"the king source {args.king_field!r} is {type(king).__name__}, not a str")
    pointer_path = Path(args.pointer)
    sidecar_path = pointer_path.with_name(pointer_path.name + SIDECAR_SUFFIX)
    current = None
    if pointer_path.exists():
        try:
            current = load_house_pointer(pointer_path, signer_ss58=args.signer_ss58)
        except LaneDHouseError as e:
            raise JobError(f"the installed house pointer is refused ({e}); fix it by hand "
                           "first, the job never overwrites a pointer it cannot verify") from None
    sidecar = _read_json(sidecar_path, "house sidecar") if sidecar_path.exists() else None
    if not king:
        log.info("lane_d_ops_house_no_lane_a_king lane A has no king; the house stays %s",
              (current or {}).get("digest", "unset"))
        return {"job": "house-pointer", "action": "kept", "king_pointer": ""}
    if current is not None and isinstance(sidecar, dict) and sidecar.get("king_pointer") == king \
            and sidecar.get("digest") == current["digest"] \
            and sidecar.get("reveal_ts") == current["reveal_ts"]:
        log.info("lane_d_ops_house_current house %s (T=%d) is current for lane A king %s",
              current["digest"], current["reveal_ts"], king)
        return {"job": "house-pointer", "action": "current", "king_pointer": king, **current}
    src = _house_source(args.house_source, king)
    house = {"digest": src["digest"], "reveal_ts": src["reveal_ts"]}
    payload = {"schema": HOUSE_SCHEMA, **house, "signer": args.signer_ss58}
    out = {"job": "house-pointer", "king_pointer": king, "previous": current, "house": house,
           "payload": payload, "signed_bytes": canonical_body(payload).decode("ascii")}
    if args.dry_run:
        log.info("lane_d_ops_house_resign_planned would sign house %s (T=%d) for lane A king %s",
              house["digest"], house["reveal_ts"], king)
        return {**out, "action": "planned"}
    if current == house:
        pointer_doc = None                         # same house: only the sidecar is missing
    else:
        from ..shared.chain import open_wallet

        wallet = open_wallet(args.wallet_name, args.wallet_hotkey, args.wallet_path)
        pointer_doc = sign_house_pointer(digest=house["digest"], reveal_ts=house["reveal_ts"],
                                         signer=wallet)
        if pointer_doc["signer"] != args.signer_ss58:
            raise JobError(f"the wallet hotkey {pointer_doc['signer']} is not the pinned house "
                           f"signer {args.signer_ss58}; nothing written")
        verify_house_pointer(pointer_doc, signer_ss58=args.signer_ss58)
        _write_atomic(pointer_path, (json.dumps(pointer_doc, sort_keys=True) + "\n").encode())
    _write_atomic(sidecar_path, (json.dumps(src, sort_keys=True) + "\n").encode())
    log.warning("lane_d_ops_house_resigned house %s (T=%d) signed for lane A king %s",
          house["digest"], house["reveal_ts"], king)
    return {**out, "action": "signed" if pointer_doc else "sidecar_written"}


# ── evidence (readiness / T7) ────────────────────────────────────────────────


def _armed(cfg, readiness: bytes | None, t7: bytes | None) -> list[int]:
    from ..eval.lane_d_scorer import armed_record

    rec = armed_record(cfg.lane_d.contexts, readiness=readiness, t7=t7,
                       min_long_series=cfg.lane_d.min_long_series)
    return list(rec["armed_contexts"])


def _bytes_or_none(path: Path | None) -> bytes | None:
    if path is None or not Path(path).exists():
        return None
    return Path(path).read_bytes()


def evidence(args) -> dict:
    from ..shared.config import load_chain_config

    if args.readiness is None and args.t7 is None:
        raise JobError("give --readiness and/or --t7 (the new documents)")
    cfg = load_chain_config(args.chain_toml)
    pinned_r, pinned_t = _bytes_or_none(args.pinned_readiness), _bytes_or_none(args.pinned_t7)
    new_r = Path(args.readiness).read_bytes() if args.readiness else None
    new_t = Path(args.t7).read_bytes() if args.t7 else None
    if new_t is not None:
        try:
            src = json.loads(new_t.decode("utf-8")).get("source")
        except (UnicodeDecodeError, ValueError, AttributeError):
            src = None
        if src != "pod":
            raise JobError(f"T7 {args.t7}: source {src!r} is not pod evidence (the T7 run is an "
                           "owner step on the eval host)")
    try:
        before = _armed(cfg, pinned_r, pinned_t) if pinned_r is not None or pinned_t is not None \
            else []
    except ValueError as e:
        before = []
        log.warning("lane_d_ops_evidence_refused the PINNED documents do not parse (%s); "
                    "replacing them", e)
    want_r = new_r if new_r is not None else pinned_r
    want_t = new_t if new_t is not None else pinned_t
    try:
        after = _armed(cfg, want_r, want_t)
    except ValueError as e:
        raise JobError(f"the new document does not parse: {e}") from None
    changes = [(p, b) for p, b, old in ((args.pinned_readiness, new_r, pinned_r),
                                       (args.pinned_t7, new_t, pinned_t))
               if b is not None and b != old]
    out = {"job": "evidence", "armed_before": before, "armed_after": after,
           "install": [str(p) for p, _ in changes],
           "sha256": {str(p): hashlib.sha256(b).hexdigest() for p, b in changes}}
    if not changes:
        log.info("lane_d_ops_evidence_current pinned documents already current; armed %s", after)
        return {**out, "action": "current"}
    lost = sorted(set(before) - set(after))
    if (lost or not after) and not args.allow_disarm:
        raise JobError(f"installing would disarm {lost or 'every context'} (armed {before} -> "
                       f"{after}); pass --allow-disarm to install anyway")
    if args.dry_run:
        log.info("lane_d_ops_evidence_planned would install %s; armed %s -> %s",
              out["install"], before, after)
        return {**out, "action": "planned"}
    for path, data in changes:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(path).with_name(f".{Path(path).name}.tmp-{os.getpid()}")
        tmp.write_bytes(data)
        os.replace(tmp, path)
    log.warning("lane_d_ops_evidence_installed installed %s; armed %s -> %s (effective next "
          "settlement)", out["install"], before, after)
    return {**out, "action": "installed"}


# ── vault-ledger ─────────────────────────────────────────────────────────────


def _reports(state_path: Path) -> list[dict]:
    from ..validator.lane_d_state import persisted_report_boundaries, report_path

    return [_read_json(report_path(state_path, b), f"report {b}")
            for b in persisted_report_boundaries(state_path)]


def _state(state_path: Path) -> dict:
    from ..validator.lane_d_state import LaneDStateError, load_state

    if not Path(state_path).is_file():
        raise JobError(f"{state_path} is absent: lane D has not settled here")
    try:
        return load_state(state_path, {})
    except LaneDStateError as e:
        raise JobError(f"{state_path}: {e}") from None


def vault_ledger(args) -> dict:
    from ..shared.config import load_chain_config
    from ..validator.lane_d_state import persisted_report_boundaries
    from ..validator.lane_d_status import build_vault_ledger

    cfg = load_chain_config(args.chain_toml)
    state = _state(args.state)
    reports = _reports(args.state)
    block = args.block
    if block is None:
        bounds = persisted_report_boundaries(args.state)
        if not bounds:
            raise JobError("no persisted report: pass --block")
        block = bounds[-1]           # the last settled boundary: never ahead of the chain
    disputes = _read_json(args.disputes, "disputes") if args.disputes else []
    if not isinstance(disputes, list):
        raise JobError(f"{args.disputes} is not a JSON list of digests")
    try:
        ledger = build_vault_ledger(cfg, state=state, reports=reports, block=int(block),
                                    disputes=disputes)
    except ValueError as e:
        raise JobError(str(e)) from None
    data = (json.dumps(ledger, sort_keys=True, indent=1) + "\n").encode()
    out = {"job": "vault-ledger", "era": ledger["era"], "entries": len(ledger["entries"]),
           "disputes": len(ledger["disputes"]), "out": str(args.out)}
    old = _bytes_or_none(args.out)
    if old == data:
        log.info("lane_d_ops_ledger_current %s is current (era %d)", args.out, ledger["era"])
        return {**out, "action": "current"}
    if args.dry_run:
        log.info("lane_d_ops_ledger_planned would write %s (era %d, %d entries)", args.out,
              ledger["era"], len(ledger["entries"]))
        return {**out, "action": "planned", "ledger": ledger}
    _write_atomic(Path(args.out), data)
    log.info("lane_d_ops_ledger_written wrote %s (era %d, %d entries)", args.out, ledger["era"],
          len(ledger["entries"]))
    return {**out, "action": "written"}


# ── king-check ───────────────────────────────────────────────────────────────


def _mirrors(specs: list[str]) -> dict:
    from .publish import MIRRORS
    from .storage import FilesystemStorage

    out = {}
    for spec in specs:
        name, sep, root = spec.partition("=")
        if not sep or name not in MIRRORS or not root:
            raise JobError(f"--mirror {spec!r}: want NAME=DIR, NAME one of {MIRRORS}")
        out[name] = FilesystemStorage(root)
    return out


def king_check(args) -> dict:
    from ..validator.lane_d_report import verify_report
    from .publish import check_one_digest, verify_index
    from .storage import FilesystemStorage

    index = _read_json(args.index, "lane D index") if Path(args.index).exists() else None
    if index is not None and not verify_index(index, args.signer_ss58):
        raise JobError(f"{args.index} does not verify under {args.signer_ss58}")
    mirrors = _mirrors(args.mirror)
    vault = FilesystemStorage(args.vault_dir) if args.vault_dir else None
    full = vault is not None and bool(mirrors)
    problems: dict[str, list[str]] = {}
    crowned = 0
    for report in _reports(args.state):
        if report.get("crowned") is None:
            continue
        crowned += 1
        b = report.get("boundary")
        if not verify_report(report, args.signer_ss58):
            problems[f"{b}"] = ["report_unverified"]
            continue
        if index is None:
            problems[f"{b}"] = ["index_missing"]
            continue
        if full:
            found = check_one_digest(index=index, report=report, vault=vault, mirrors=mirrors)
        else:                                       # index-only check (no local copies)
            entry = next((k for k in index.get("kings") or ()
                          if isinstance(k, dict) and k.get("digest") == report["crowned"]), None)
            found = (["index_missing"] if entry is None else
                     ["crown_block_mismatch"] if entry.get("crown_block") != b else [])
        if found:
            problems[f"{b}"] = found
    out = {"job": "king-check", "crowned_reports": crowned, "scope": "full" if full else "index",
           "problems": problems}
    if problems:
        log.critical("lane_d_ops_king_unpublished %d crowned settlement(s) disagree with the "
              "published index/mirrors: %s", len(problems), problems)
        raise _CheckFailed(out)
    log.info("lane_d_ops_king_ok %d crowned settlement(s) agree with the index (%s check)",
          crowned, out["scope"])
    return out


class _CheckFailed(Exception):
    def __init__(self, out: dict):
        super().__init__("check failed")
        self.out = out


# ── CLI ──────────────────────────────────────────────────────────────────────

def _log_failed(job: str, e: Exception) -> None:
    if job == "house-pointer":
        log.error("lane_d_ops_house_failed %s; nothing written", e)
    elif job == "evidence":
        log.error("lane_d_ops_evidence_refused %s; nothing written", e)
    elif job == "vault-ledger":
        log.error("lane_d_ops_ledger_failed %s; nothing written", e)
    else:
        log.error("lane_d_ops_king_check_failed %s", e)


def _configure_logging(log_file: Path | None) -> None:
    handler: logging.Handler = (logging.FileHandler(log_file) if log_file
                                else logging.StreamHandler(sys.stderr))
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    log.handlers[:] = [handler]
    log.setLevel(logging.INFO)
    log.propagate = False


def _run(args) -> int:
    _configure_logging(args.log_file)
    job = args.job
    try:
        out = args.job_func(args)
    except JobBlocked as e:              # only house-pointer blocks on an owner step
        log.error("lane_d_ops_house_resign_blocked %s", e)
        print(json.dumps({"job": job, "action": "blocked", "reason": str(e)}, sort_keys=True))
        return EXIT_BLOCKED
    except _CheckFailed as e:
        print(json.dumps({**e.out, "action": "problems"}, sort_keys=True))
        return EXIT_FAIL
    except Exception as e:  # noqa: BLE001 — JobError, a wallet or signing error: exit 1
        _log_failed(job, e)
        print(json.dumps({"job": job, "action": "failed", "reason": str(e)}, sort_keys=True))
        return EXIT_FAIL
    print(json.dumps(out, sort_keys=True, default=str))
    return EXIT_OK


def add_parser(subparsers) -> None:
    p = subparsers.add_parser("jobs", help="lane D recurring ops jobs (idempotent, --dry-run)")
    jobs = p.add_subparsers(dest="job", required=True)

    def common(sp):
        sp.add_argument("--dry-run", action="store_true", help="print the plan; write nothing")
        sp.add_argument("--log-file", type=Path, default=None,
                        help="append log lines here (default stderr / the journal)")
        sp.set_defaults(func=_run)

    h = jobs.add_parser("house-pointer", help="re-sign the house pointer on a lane A king change")
    h.add_argument("--king-source", type=Path, required=True,
                   help="local JSON naming the lane A king pointer (status/v2.json)")
    h.add_argument("--king-field", default="king.pointer", help="dotted field (default king.pointer)")
    h.add_argument("--pointer", type=Path, required=True,
                   help="the house pointer file the --lane-d-inputs TOML names (house_pointer)")
    h.add_argument("--signer-ss58", required=True, help="the pinned house_signer_ss58")
    h.add_argument("--house-source", type=Path, default=None,
                   help=f"the {HOUSE_SOURCE_SCHEMA} JSON prepared for the current lane A king")
    h.add_argument("--wallet-name", default=None)
    h.add_argument("--wallet-hotkey", default=None)
    h.add_argument("--wallet-path", default=None)
    common(h)
    h.set_defaults(job_func=house_pointer)

    e = jobs.add_parser("evidence", help="validate and install laned_readiness/1 / laned_t7/1")
    e.add_argument("--chain-toml", type=Path, required=True)
    e.add_argument("--readiness", type=Path, default=None, help="new laned_readiness/1 (W1)")
    e.add_argument("--t7", type=Path, default=None, help="new laned_t7/1 (pod run)")
    e.add_argument("--pinned-readiness", type=Path, required=True,
                   help="the inputs TOML 'readiness' path")
    e.add_argument("--pinned-t7", type=Path, required=True, help="the inputs TOML 't7' path")
    e.add_argument("--allow-disarm", action="store_true",
                   help="install even if a currently armed context disarms")
    common(e)
    e.set_defaults(job_func=evidence)

    v = jobs.add_parser("vault-ledger", help="export laned_vault_ledger/1 for the W18 sweeper")
    v.add_argument("--chain-toml", type=Path, required=True)
    v.add_argument("--state", type=Path, required=True, help="lane_d_state.json")
    v.add_argument("--out", type=Path, required=True, help="the ledger the sweeper reads")
    v.add_argument("--block", type=int, default=None,
                   help="the block the era is taken at (default: the last persisted boundary)")
    v.add_argument("--disputes", type=Path, default=None, help="JSON list of disputed digests")
    common(v)
    v.set_defaults(job_func=vault_ledger)

    k = jobs.add_parser("king-check", help="crowned reports vs the signed index (and mirrors)")
    k.add_argument("--state", type=Path, required=True, help="lane_d_state.json")
    k.add_argument("--index", type=Path, required=True, help="local champions/lane_d/index.json")
    k.add_argument("--signer-ss58", required=True, help="the pinned report / index signer")
    k.add_argument("--vault-dir", type=Path, default=None, help="local vault copy (full check)")
    k.add_argument("--mirror", action="append", default=[], help="NAME=DIR local mirror copy")
    common(k)
    k.set_defaults(job_func=king_check)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -P -m cascade.lane_d.ops_jobs")
    sub = parser.add_subparsers(dest="command", required=True)
    add_parser(sub)
    args = parser.parse_args(["jobs", *(sys.argv[1:] if argv is None else argv)])
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
